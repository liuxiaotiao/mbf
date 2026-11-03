# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""moe_color_planner 的可部署方案 → B1 分段引擎清单（stages.json）。**纯 Python，不 import torch / vllm。**

    # 控制机上（按文件路径运行；只读同目录的 layout.py / stages.py）
    python3 vllm/p2p/colorplan.py \\
        --result   ../moe_color_planner/result_deploy.json \\
        --instance ../moe_color_planner/instance_v2.json \\
        --experts  ../moe_color_planner/data/expert_sets_90_union.json \\
        --hosts    /home/ubuntu/p2p-framework/task/hosts.txt \\
        --model-dir /home/ubuntu/p2p-framework/weights-q3-30b \\
        --out p2p_channels/stages_color.json
    python3 vllm/p2p/launch.py --channels p2p_channels/stages_color.json start all
    python vllm/p2p/tiered_gateway.py --stages p2p_channels/stages_color.json --model-dir … --port 9100 \
        --mode lr --lr-at front          # 给了 --lr-model 时：LR 在各前段卡本机跑

LR 选择器部署在含第 L₀ 层的节点上：可部署方案里前段卡装满前 L₀ 层，所以就是每张前段卡。
给了 `--lr-model` 时，前段引擎的 hf_overrides 里写 `p2p_lr`（选择器路径、部署的 task 与先验、阈值），
`stage_agent` 在前段本机加载选择器、prompt 算完时本机判 task，只把判定交给网关
（网关 `--mode lr --lr-at front`）。前段卡上要有 p2p-framework（`--p2p-framework`，默认自动找）。

规划默认按网关的「前后段直连」算延迟；规划时用了 `--deploy-via-coordinator` 的话，网关要加
`--via-coordinator`、并跑在规划的协调器节点上 —— 下面会提示。

规划结果必须是 `moe_color_planner.py --deploy` 解出来的（结果里有 `deploy` 段），结构与 B1 运行时一致：

* **前段**：每张前段卡一个引擎 `front-<节点>`，单卡装满前 L₀ 层、全部专家，三个任务共用；
* **后段**：每条后段流水线一个引擎 `back-<task>-<序号>`，PP 跨流水线上的各卡，切层取规划结果，
  驻留专家取 `expert_sets_90_union.json` 里该 task 的 `union`；
* 前段数与后段条数**不必相等**（不经过 layout.py 的一对一配对，直接写 stages.json）。

与 `stages.py` 生成的 stages.json 同构（`kind = "tiered"`、`fronts` / `backs` / `channels`），
`launch.py`、`tiered_gateway.py`、`bench.py` 原样可用；启动命令用的也是 `stages.agent_command`。

运行时怎么选前段 / 后段仍由网关决定（前段按速度、后段在同 task 里挑在飞最少）；规划给出的
每条后段的流量占比写在 `color_plan` 里，用来核对实际分到的请求比例。

`--max-num-seqs`：默认按规划估一条引擎同时在飞的请求数 ≈ 它的 token 速率 × 平均循环时间，
再乘 `--seqs-margin`、向上取到 8 的倍数（下限 16）。

每张卡每步的批量 ≠ 引擎的在飞数：一条请求某一时刻只停在流水线的一级上，所以卡 i 每步约处理
「token 速率 × 卡 i 一步的耗时」条（Little 定律）。规划的每层耗时取的是实例里 batch（默认 64）
下的 profiling 值；估出的每步批量明显偏离它时会提示（偏大 → 实际每步更慢、TPOT 比规划高）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Mapping

__all__ = ["stages_from_color_plan", "main"]

ARCH = "Qwen3MoePartialForCausalLM"
_HERE = Path(__file__).resolve().parent


def _load_sibling(name: str):
    """按文件路径加载同目录的模块（不经过 `vllm` 包，控制机上没有 torch 也能跑）。"""
    spec = importlib.util.spec_from_file_location(f"_p2p_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod            # dataclass（layout.py）解析注解时要在 sys.modules 里找到模块
    spec.loader.exec_module(mod)
    return mod


class ColorPlanError(ValueError):
    """规划结果不能直接部署（不是 --deploy 模式、结构不对、缺地址……）。"""


def _real_layers(lo: int, hi: int, bl: int) -> tuple[int, int]:
    """规划里的块号区间（1-based、闭区间）→ 真实层号区间（1-based、闭区间）。"""
    return (lo - 1) * bl + 1, hi * bl


def _round_seqs(x: float, margin: float, floor: int = 16) -> int:
    return max(floor, int(math.ceil(x * margin / 8.0)) * 8)


def stages_from_color_plan(
    res: Mapping, inst: Mapping, experts: Mapping, hosts: Mapping, *,
    model_dir: str, python: str, dtype: str = "half", attention_backend: str | None = "TRITON_ATTN",
    max_model_len: int = 4096, gpu_mem: float = 0.9, gpu_mem_node: Mapping[str, float] | None = None,
    port: int = 8300, master_port: int = 29600, front_lr: bool = True, eager: bool = False,
    seqs_margin: float = 1.3, front_max_num_seqs: int | None = None,
    back_max_num_seqs: int | None = None, plan_path: str | None = None,
    lr: Mapping | None = None,
) -> tuple[dict, list[str]]:
    """`lr`：前段本机 LR 的配置（写进前段 hf_overrides 的 `p2p_lr`），至少有 `model`；
    `tasks` 不给就用后段覆盖到的 task、先验各 1。"""
    """→ (stages.json 的内容, 提示)。"""
    st = _load_sibling("stages")
    dep = res.get("deploy")
    if not dep:
        raise ColorPlanError("结果里没有 deploy 段 —— 要用 moe_color_planner.py --deploy 求解")
    if dep.get("problems"):
        raise ColorPlanError("规划结果的结构不能部署：\n  " + "\n  ".join(dep["problems"]))
    bl = int(inst.get("block_layers", 1))
    n_layers = int(inst["L"]) * bl
    l0 = int(dep["l0"]) * bl
    n_experts = int(inst.get("n_experts", 128))
    tasks_ex = experts["tasks"]
    gmn = dict(gpu_mem_node or {})
    notes: list[str] = []

    need = {f["node"] for f in dep["fronts"]} | {v for b in dep["backs"] for v in b["nodes"]}
    miss = sorted(need - set(hosts))
    if miss:
        raise ColorPlanError(f"hosts 里没有 {miss} 的地址")
    batch = int(inst.get("batch", 64))
    gtype = {g["id"]: g["type"] for g in inst.get("gpus", [])}
    d_layer = {t: float(v.get("d_layer_ms") or v["d_ms"] / bl) for t, v in inst.get("gpu_types", {}).items()}

    def addr(v: str) -> str:
        return hosts[v].host

    def step_batch(v: str, rate: float, n_lay: int) -> float | None:
        """卡 v 每步约处理几条：token 速率 × 这张卡一步的耗时（Little 定律）。"""
        t = gtype.get(v)
        return None if t not in d_layer else rate * n_lay * d_layer[t] / 1000.0

    def check_batch(name: str, v: str, b: float | None) -> None:
        if b is not None and (b > 1.5 * batch or b < batch / 4):
            notes.append(f"{name} 的 {v} 每步约 {b:.0f} 条，偏离规划用的 batch {batch}"
                         f"（{'偏大：实际每步更慢，TPOT 会比规划高' if b > batch else '偏小：实际每步更快'}）")

    lr_cfg = None
    if lr:
        if not front_lr:
            raise ColorPlanError("前段本机 LR 要前段交出识别统计：不能同时 --no-front-lr")
        if not lr.get("model"):
            raise ColorPlanError("前段本机 LR 要给选择器路径（--lr-model）")
        have = sorted({b["task"] for b in dep["backs"]})
        tasks = dict(lr.get("tasks") or {t: 1.0 for t in have})
        extra = sorted(set(tasks) - set(have))
        if extra:
            raise ColorPlanError(f"LR 的 task {extra} 没有后段（有的是 {have}）")
        lr_cfg = {"model": str(lr["model"]), "tasks": {k: float(v) for k, v in tasks.items()},
                  "p2p_framework": lr.get("p2p_framework"), "classes": lr.get("classes"),
                  "features": lr.get("features"), "feature_fn": lr.get("feature_fn"),
                  "tau_hi": lr.get("tau_hi"), "tau_lo": float(lr.get("tau_lo", 0.40))}

    fronts, backs = [], []
    # ---- 前段：每张前段卡一个引擎，所有 task 共用；LR 选择器也在这里（含第 L₀ 层） ----
    for fr in dep["fronts"]:
        v = fr["node"]
        name = f"front-{v}"
        fov = {"architectures": [ARCH], "p2p_layer_experts": {}, "p2p_channel": name,
               "p2p_miss_policy": "local_topk"}
        if front_lr:
            fov["p2p_stats"] = {"lr_layers": list(range(l0)), "emit_lr": True,
                                "write": False, "prefill_only": True}
        if lr_cfg:
            fov["p2p_lr"] = lr_cfg
        conc = fr["tokens_per_s"] * (fr.get("avg_cycle_ms") or 0.0) / 1000.0
        mns = front_max_num_seqs or _round_seqs(conc, seqs_margin)
        fb = step_batch(v, fr["tokens_per_s"], l0)
        check_batch(name, v, fb)
        gm = gmn.get(v, gpu_mem)
        fronts.append({
            "name": name, "role": "front", "task": None, "nodes": [v],
            "layers": [0, l0], "partition": [l0], "dtype": dtype, "gpu_mem": {v: gm},
            "max_num_seqs": mns, "api": f"http://{addr(v)}:{port}/v1",
            "zmq": f"tcp://{addr(v)}:{port + 1}", "from_channels": [],
            "plan": {"tokens_per_s": fr["tokens_per_s"], "avg_cycle_ms": fr.get("avg_cycle_ms"),
                     "inflight_est": round(conc, 1),
                     "step_batch_est": {v: None if fb is None else round(fb, 1)}},
            "commands": [{"node": v, "host": addr(v), "cmd": st.agent_command(
                python=python, name=name, model_dir=model_dir, role="front", layers=(0, l0),
                task=None, nodes=[v], rank=0, master=addr(v), master_port=master_port,
                partition=None, dtype=dtype, max_model_len=max_model_len, gpu_mem=gm,
                max_num_seqs=mns, attention_backend=attention_backend, port=port,
                overrides=fov, eager=eager)}],
        })

    # ---- 后段：每条流水线一个引擎 ----
    count: dict[str, int] = {}
    for b in dep["backs"]:
        task = b["task"]
        if task not in tasks_ex:
            raise ColorPlanError(f"专家集合文件里没有 task {task!r}")
        union = tasks_ex[task]["union"]
        if len(union) != n_layers:
            raise ColorPlanError(f"{task} 的专家集合有 {len(union)} 层，模型是 {n_layers} 层")
        k = count.get(task, 0)
        count[task] = k + 1
        name = f"back-{task}-{k}"
        nodes = list(b["nodes"])
        ranges = [_real_layers(lo, hi, bl) for lo, hi in b["layers"]]
        nxt = l0 + 1
        for v, (lo, hi) in zip(nodes, ranges):
            if lo != nxt:
                raise ColorPlanError(f"{name}：{v} 从第 {lo} 层开始，上一级到第 {nxt - 1} 层")
            nxt = hi + 1
        if nxt - 1 != n_layers:
            raise ColorPlanError(f"{name} 只覆盖到第 {nxt - 1} 层")
        partition = [hi - lo + 1 for lo, hi in ranges]
        # 驻留专家：0-based 层号；装全的层不列
        le = {}
        for l1 in range(l0 + 1, n_layers + 1):
            es = sorted(int(e) for e in union[l1 - 1])
            if not es or es[0] < 0 or es[-1] >= n_experts or len(set(es)) != len(es):
                raise ColorPlanError(f"{task} 第 {l1} 层的专家集合不合法")
            if len(es) < n_experts:
                le[str(l1 - 1)] = es
        bov = {"architectures": [ARCH], "p2p_layer_experts": le,
               "p2p_miss_policy": "local_topk", "p2p_channel": name}
        conc = b["tokens_per_s"] * b["avg_cycle_ms"] / 1000.0
        mns = back_max_num_seqs or _round_seqs(conc, seqs_margin)
        sb = {}
        for v, (lo, hi) in zip(nodes, ranges):
            sb[v] = step_batch(v, b["tokens_per_s"], hi - lo + 1)
            check_batch(name, v, sb[v])
        master = addr(nodes[0])
        backs.append({
            "name": name, "role": "back", "task": task, "nodes": nodes,
            "layers": [l0, n_layers], "partition": partition, "dtype": dtype,
            "gpu_mem": {v: gmn.get(v, gpu_mem) for v in nodes}, "max_num_seqs": mns,
            "api": f"http://{master}:{port}/v1", "zmq": f"tcp://{master}:{port + 1}",
            "from_channels": [],
            "plan": {"layers_1based": [list(r) for r in ranges], "tokens_per_s": b["tokens_per_s"],
                     "share_of_task": b["share_of_task"], "avg_cycle_ms": b["avg_cycle_ms"],
                     "inflight_est": round(conc, 1),
                     "step_batch_est": {v: None if x is None else round(x, 1) for v, x in sb.items()}},
            "commands": [{"node": v, "host": addr(v), "cmd": st.agent_command(
                python=python, name=name, model_dir=model_dir, role="back",
                layers=(l0, n_layers), task=task, nodes=nodes, rank=r, master=master,
                master_port=master_port, partition=partition, dtype=dtype,
                max_model_len=max_model_len, gpu_mem=gmn.get(v, gpu_mem), max_num_seqs=mns,
                attention_backend=attention_backend, port=port, overrides=bov,
                eager=eager)} for r, v in enumerate(nodes)],
        })

    tasks = sorted({b["task"] for b in backs})
    no_back = sorted(set(tasks_ex) - set(tasks))
    if no_back:
        notes.append(f"这些 task 没有后段：{no_back}（网关会对它们返回 503）")
    via = bool(dep.get("via_coordinator", dep.get("coordinator") is not None))
    coord = dep.get("coordinator") if via else None
    if via and coord and coord in hosts:
        notes.append(f"规划按「经协调器中转」算延迟：网关要加 --via-coordinator，并跑在 {coord}（{addr(coord)}）上")
    gw = ["python", "vllm/p2p/tiered_gateway.py", "--stages", "<本文件>", "--model-dir", model_dir,
          "--port", "9100"]
    if via:
        gw.append("--via-coordinator")
    if lr_cfg:
        gw += ["--mode", "lr", "--lr-at", "front"]
    out = {
        "kind": "tiered", "plan": plan_path, "l0": l0, "n_layers": n_layers, "n_experts": n_experts,
        "model_dir": model_dir, "fronts": [f["name"] for f in fronts],
        "backs": [b["name"] for b in backs], "channels": fronts + backs,
        "color_plan": {"via_coordinator": via, "coordinator": coord,
                       "lr_at": "front" if lr_cfg else "gateway",
                       "gateway_cmd": " ".join(gw),
                       "theta_per_task": res.get("summary", {}).get("theta"),
                       "avg_cycle_ms": {k: v.get("tpot_ms") for k, v in res.get("colors", {}).items()}},
    }
    return out, notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", required=True, type=Path, help="moe_color_planner.py --deploy 的结果 JSON")
    ap.add_argument("--instance", required=True, type=Path, help="求解用的实例（取 block_layers / L / batch）")
    ap.add_argument("--experts", required=True, type=Path, help="expert_sets_90_union.json（取各 task 的 union）")
    ap.add_argument("--hosts", required=True, type=Path, help="节点 id → 地址（task/hosts.txt 格式）")
    ap.add_argument("--model-dir", default="/home/ubuntu/p2p-framework/weights-q3-30b",
                    help="各节点上的 checkpoint 目录（同一路径）")
    ap.add_argument("--python", default="/home/ubuntu/venvs/vllm-p2p/bin/python",
                    help="各节点上 vllm-p2p venv 的 python")
    ap.add_argument("--out", type=Path, default=Path("p2p_channels/stages_color.json"))
    ap.add_argument("--dtype", default="half", choices=("half", "bfloat16"),
                    help="默认 half：有 T4 / RTX6000（sm75，不支持 bf16）")
    ap.add_argument("--attention-backend", default="TRITON_ATTN")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--gpu-mem-node", action="append", default=[], metavar="节点=比例")
    ap.add_argument("--port", type=int, default=8300, help="代理的 HTTP 端口（ZMQ = 它 + 1）")
    ap.add_argument("--master-port", type=int, default=29600)
    ap.add_argument("--seqs-margin", type=float, default=1.3,
                    help="max_num_seqs = 规划估的在飞请求数 × 这个系数（向上取到 8 的倍数）")
    ap.add_argument("--front-max-num-seqs", type=int, default=None, help="所有前段统一用这个值")
    ap.add_argument("--back-max-num-seqs", type=int, default=None, help="所有后段统一用这个值")
    ap.add_argument("--no-front-lr", action="store_true", help="前段不开识别统计（只能 oracle）")
    g = ap.add_argument_group("前段本机 LR（LR 选择器在含第 L₀ 层的前段卡上）")
    g.add_argument("--lr-model", default=None,
                   help="选择器参数文件（前段卡上的路径），如 /home/ubuntu/p2p-framework/selector/"
                        "selector_body_L8_params.npz；不给 = LR 留在网关里跑")
    g.add_argument("--lr-tasks", default=None, help="部署的 task 与先验，如 gsm8k=1,mbpp=1（默认 = 后段的 task，各 1）")
    g.add_argument("--p2p-framework", default=None, help="前段卡上 p2p-framework 的目录（默认自动找）")
    g.add_argument("--lr-classes", default=None)
    g.add_argument("--lr-features", default=None)
    g.add_argument("--lr-feature-fn", default=None)
    g.add_argument("--tau-hi", type=float, default=None)
    g.add_argument("--tau-lo", type=float, default=0.40)
    ap.add_argument("--eager", action="store_true")
    a = ap.parse_args(argv)

    lay = _load_sibling("layout")
    try:
        res = json.loads(a.result.read_text(encoding="utf-8"))
        inst = json.loads(a.instance.read_text(encoding="utf-8"))
        experts = json.loads(a.experts.read_text(encoding="utf-8"))
        hosts = lay.parse_hosts(a.hosts)
        gmn = lay.parse_gpu_mem_node(a.gpu_mem_node)
        lr = None
        if a.lr_model:
            tasks = None
            if a.lr_tasks:
                tasks = {}
                for part in a.lr_tasks.split(","):
                    k, _, v = part.strip().partition("=")
                    if k:
                        tasks[k.strip()] = float(v) if v else 1.0
            lr = {"model": a.lr_model, "tasks": tasks, "p2p_framework": a.p2p_framework,
                  "classes": a.lr_classes, "features": a.lr_features, "feature_fn": a.lr_feature_fn,
                  "tau_hi": a.tau_hi, "tau_lo": a.tau_lo}
        sj, notes = stages_from_color_plan(
            res, inst, experts, hosts, model_dir=a.model_dir, python=a.python, dtype=a.dtype,
            attention_backend=a.attention_backend or None, max_model_len=a.max_model_len,
            gpu_mem=a.gpu_mem, gpu_mem_node=gmn, port=a.port, master_port=a.master_port,
            front_lr=not a.no_front_lr, eager=a.eager, seqs_margin=a.seqs_margin,
            front_max_num_seqs=a.front_max_num_seqs, back_max_num_seqs=a.back_max_num_seqs,
            plan_path=str(a.result), lr=lr)
    except (ColorPlanError, lay.LayoutError, OSError, KeyError) as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(sj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(sj['fronts'])} 个前段，{len(sj['backs'])} 个后段（L₀ = {sj['l0']}，共 {sj['n_layers']} 层）：")
    for c in sj["channels"]:
        p = c["plan"]
        extra = (f"  切层 {p['layers_1based']}  占该任务 {p['share_of_task']:.0%}" if c["role"] == "back" else "")
        sb = ", ".join(f"{v} {x:.0f}" for v, x in (p.get("step_batch_est") or {}).items() if x is not None)
        print(f"  {c['name']:<22} {c['role']:<5} task={c['task'] or '—':<10} 节点 {'+'.join(c['nodes'])}"
              f"  规划 {p['tokens_per_s']:.0f} tokens/s，在飞 ≈ {p['inflight_est']:.0f}"
              f" → max_num_seqs {c['max_num_seqs']}{extra}" + (f"\n{'':26}每步批量 ≈ {sb}" if sb else ""))
    for s in notes:
        print(f"  ! {s}")
    if sj["color_plan"]["lr_at"] == "front":
        print(f"  LR 选择器在各前段卡本机跑（{a.lr_model}）")
    print(f"已写 {a.out}\n拉起：python3 vllm/p2p/launch.py --channels {a.out} start all")
    print(f"网关：{sj['color_plan']['gateway_cmd'].replace('<本文件>', str(a.out))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
