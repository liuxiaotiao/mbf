# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""实测「一层模型」在各型 GPU 上要多少毫秒 —— 给 p2p-framework 的路径规划器用。

规划器（`p2pmoe/planner`）给每个节点一个 `ms_per_layer`（每层每 token 的计算耗时），段的计算时延
= 层数 × ms_per_layer，再和网络时延同量纲相加挑路径。原来它由拓扑里的抽象算力
`compute_capacity` 按反比换算（A30 定为 0.35 ms），而 B1 实测前段每位置的耗时是
A30 4 ms / RTX 6000 15 ms / T4 26 ms（8 层）—— 卡型之间差 3.7–6.5 倍，拓扑里写的是 1.26 / 2.5 倍。
这里用**部署时同一套 vLLM kernel** 直接量。

三个子命令::

    # 1. 在一台节点上量本机的 GPU（vllm-p2p 的 venv；卡要空着）
    python -m vllm.p2p.layer_bench run --model-dir /home/ubuntu/p2p-framework/weights-q3-30b \\
        --out results/layer_bench_N01.json

    # 2. 在控制机上量所有节点（各台同时 ssh 过去跑 run，结果收回来，按卡型取均值）
    python3 vllm/p2p/layer_bench.py cluster --hosts /home/ubuntu/p2p-framework/task/hosts.txt \\
        --out-dir results/layer_bench

    # 3. 把结果写进规划器的拓扑文件（新文件；原文件不动）：每个节点用它这种卡型的均值
    python3 vllm/p2p/layer_bench.py topo --topo /home/ubuntu/p2p-framework/task/topo_fabric_mbps.json \\
        --results results/layer_bench/*.json --out /home/ubuntu/p2p-framework/task/topo_measured.json

怎么量的（`run`）
----------------
不要 checkpoint：拿 `--model-dir` 里的 config.json（层的形状），用 vLLM 的 `load_format=dummy`
（随机权重）起**整模型引擎**，只是把层数改小。同一组参数起两次（默认 2 层、6 层），每次量
「batch B 条请求、上下文 C 个 token 时，一步 decode 的耗时」与「长 P 的 prompt 一次 prefill 的耗时」
（C、P 默认都是 128, 256, …, 1024，`--lengths 起:止:步长`），
两次相减除以层数差 = **一层**的耗时。embed、lm_head、采样、调度这些与层数无关的开销在相减时抵掉了。
kernel 与部署时相同：`Qwen3MoePartialForCausalLM`（每层只建 E 个专家，`--experts` 可量几档）、
fp16、TRITON_ATTN、CUDA Graph（`--eager` 关掉）。每个配置在单独的子进程里起引擎。

* decode 一步：先发 B 条长 C 的随机 prompt 只生成 1 个 token（= prefill + 1 步），再发同样的
  B 条生成 1+K 个，两者之差 / K = 一步 decode（上下文从 C 涨到 C+K，取的是这一段的平均）。
  重复 `--repeats` 次取中位数。同一组 prompt 反复用、引擎开前缀缓存：两次的 prompt 都命中缓存，
  各只重算最后不满一块的几个位置，相减后只剩 K 步 decode。所以 `max_num_batched_tokens`
  不用设成 B×C（`--max-batched-tokens`，默认 16384；B=320、C=1280 时设成 B×C 会在 profile 时 OOM）。
* prefill：B=1、长 P 的 prompt 只生成 1 个 token 的耗时（每条 prompt 都是新随机的，不命中前缀缓存）。

随机权重的路由近似均匀，batch > 1 时碰到的专家数比真实路由（偏斜）略多 —— decode 的单条
（B=1）几乎不受影响，大 batch 的数偏保守。

显存（`memory[E]`）：引擎起好后、量完后各在 worker 里查一次（`collective_rpc`）——
* 每层权重：逐层数参数字节（注意力 + norm / 路由 / 专家，及每个专家多少），是精确值，不是估的；
* 层外权重：embed_tokens + lm_head + 最后的 norm；
* KV：每 token 每层多少字节（按 vLLM 实际分配的 KV 张量算），本次能放多少 token；
* 运行时峰值：PyTorch 分配器的峰值 − 权重 − KV（激活、采样缓冲等），按层数拟合出「每层」与「固定」两部分；
* 整卡非 KV 占用：(显存总量 − 空闲) − KV，同样拟合成每层 + 固定（固定部分含 CUDA 上下文、CUDA Graph 池）。

输出（每台一个 JSON）里 `decode[E][C][B]` / `prefill[E][P]` 是每层毫秒，`memory[E]` 是上面的显存账，
`summary.ms_per_layer` 是给规划器的那个数（默认 E=128、B=1、各上下文长度取均值的 decode）。

多台（`cluster`）：默认量**全部节点**，再按卡型对每一项取**均值**（`summary_by_gpu.json`，另记每台的
`ms_per_layer` 与最快 / 最慢之比，看同型卡之间差多少）；`topo` 默认给每个节点写它这种卡型的均值
（`--per-node` 改用各台自己的）。

`cluster` 与 `topo` 只用标准库，控制机上按文件路径直接跑。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import statistics as st
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

__all__ = ["gpu_key", "fit_per_layer", "parse_lengths", "summarise", "aggregate",
           "topo_with_measurements", "main"]

ARCH = "Qwen3MoePartialForCausalLM"
DEFAULT_LAYERS = (2, 6)
DEFAULT_EXPERTS = (128, 64)
DEFAULT_BATCHES = (1, 2, 4, 8, 16)
DEFAULT_LENGTHS = "128:1024:128"           # prefill 长度与 decode 上下文：128, 256, …, 1024
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "StrictHostKeyChecking=accept-new"]


# --------------------------------------------------------------------------- #
# 纯函数（控制机上也能跑）
# --------------------------------------------------------------------------- #
def gpu_key(name: str) -> str:
    """「NVIDIA A30」「Quadro RTX 6000」「Tesla T4」→ A30 / RTX6000 / T4（与拓扑的 role、hosts 的注释同口径）。"""
    s = str(name or "").upper()
    for p in ("NVIDIA ", "TESLA ", "QUADRO ", "GEFORCE "):
        s = s.replace(p, "")
    return re.sub(r"[\s\-_]", "", s)


def parse_lengths(spec: str) -> list[int]:
    """「128:1024:128」→ [128, 256, …, 1024]；「128,512,1024」→ 原样。"""
    spec = str(spec).strip()
    if ":" in spec:
        parts = [int(x) for x in spec.split(":")]
        if len(parts) == 2:
            parts.append(parts[0])
        a, b, step = parts
        if a <= 0 or b < a or step <= 0:
            raise ValueError(f"长度 {spec!r} 要写成 起:止:步长（正整数，止 ≥ 起）")
        return list(range(a, b + 1, step))
    out = [int(x) for x in spec.split(",") if x.strip()]
    if not out or min(out) <= 0:
        raise ValueError(f"长度 {spec!r} 不对")
    return out


def _mean(xs):
    xs = [float(x) for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def fit_per_layer(points: Mapping[int, float]) -> tuple[float, float]:
    """{层数: 毫秒} → (每层毫秒, 截距)。最小二乘（两个点时就是两点斜率）。"""
    xs = sorted(points)
    if len(xs) < 2:
        raise ValueError("至少要两种层数才能把与层数无关的开销减掉")
    ys = [float(points[x]) for x in xs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx


MiB = 1024 ** 2


def _bytes(mod) -> int:
    return sum(p.numel() * p.element_size() for p in mod.parameters()) if mod is not None else 0


def _storage_bytes(tensors) -> int:
    """KV 张量（可能嵌套在 list 里、可能共享存储）的总字节数，按存储去重。"""
    seen, tot = set(), 0

    def walk(x):
        nonlocal tot
        if isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
            return
        if not hasattr(x, "untyped_storage"):
            return
        st_ = x.untyped_storage()
        key = (st_.data_ptr(), st_.nbytes())
        if key not in seen:
            seen.add(key)
            tot += st_.nbytes()
    walk(tensors)
    return tot


def mem_probe(worker) -> dict:
    """在 worker 进程里跑（`LLM.collective_rpc`）：逐层权重、层外权重、KV、PyTorch 与整卡显存。"""
    import torch

    mr = worker.model_runner
    model = mr.get_model() if hasattr(mr, "get_model") else mr.model
    inner = model.model
    layers = []
    for i in range(int(getattr(inner, "start_layer", 0)), int(getattr(inner, "end_layer", len(inner.layers)))):
        lay = inner.layers[i]
        mlp = getattr(lay, "mlp", None)
        experts = getattr(mlp, "experts", None)
        w13 = getattr(experts, "w13_weight", None)
        n_exp = int(w13.shape[0]) if w13 is not None else 0
        tot, ex, rt = _bytes(lay), _bytes(experts), _bytes(getattr(mlp, "gate", None))
        layers.append({"idx": i, "bytes": tot, "experts_bytes": ex, "router_bytes": rt,
                       "attn_norm_bytes": tot - ex - rt, "n_experts": n_exp})
    params = _bytes(model)
    cc = getattr(worker, "cache_config", None) or getattr(mr, "cache_config", None)
    blocks = int(getattr(cc, "num_gpu_blocks", 0) or 0)
    bsz = int(getattr(cc, "block_size", 0) or 0)
    out = {"layers": layers, "param_bytes": params,
           "non_layer_param_bytes": params - sum(l["bytes"] for l in layers),
           "kv_bytes": _storage_bytes(getattr(mr, "kv_caches", [])), "kv_tokens": blocks * bsz,
           "kv_layers": len(layers)}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        out.update(torch_allocated=torch.cuda.memory_allocated(), torch_reserved=torch.cuda.memory_reserved(),
                   torch_peak_allocated=torch.cuda.max_memory_allocated(),
                   device_total=total, device_free=free)
    return out


def summarise_memory(runs: Sequence[Mapping]) -> dict:
    """同一档专家数、不同层数的几次 → 每层 / 固定 的显存账（MiB）。runs[i]["mem"] = {"after_init",
    "after_run"}（`mem_probe` 的输出）。"""
    rs = [r for r in runs if (r.get("mem") or {}).get("after_init")]
    if not rs:
        return {}

    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(st.median(xs), 3) if xs else None

    per = []
    for r in rs:
        m = r["mem"]["after_init"]
        ls = m["layers"]
        n = max(len(ls), 1)
        per.append({
            "L": int(r["layers"]),
            "w": sum(l["bytes"] for l in ls) / n / MiB,
            "attn": sum(l["attn_norm_bytes"] for l in ls) / n / MiB,
            "router": sum(l["router_bytes"] for l in ls) / n / MiB,
            "exp": sum(l["experts_bytes"] for l in ls) / n / MiB,
            "per_exp": (sum(l["experts_bytes"] for l in ls) / max(sum(l["n_experts"] for l in ls), 1) / MiB),
            "n_exp": sum(l["n_experts"] for l in ls) / n,
            "nonlayer": m["non_layer_param_bytes"] / MiB,
            "kv_tok_layer": (m["kv_bytes"] / (m["kv_tokens"] * m["kv_layers"])
                             if m.get("kv_tokens") and m.get("kv_layers") else None),
            "kv_tokens": m.get("kv_tokens"),
            "device_non_kv": ((m["device_total"] - m["device_free"] - m["kv_bytes"]) / MiB
                              if m.get("device_total") else None),
            "peak_act": (((r["mem"].get("after_run") or m).get("torch_peak_allocated", 0)
                          - m["param_bytes"] - m["kv_bytes"]) / MiB if m.get("device_total") else None),
            "device_total": m.get("device_total"),
        })
    out = {
        "experts_per_layer": med([p["n_exp"] for p in per]),
        "weight_mib_per_layer": med([p["w"] for p in per]),
        "attn_norm_mib_per_layer": med([p["attn"] for p in per]),
        "router_mib_per_layer": med([p["router"] for p in per]),
        "experts_mib_per_layer": med([p["exp"] for p in per]),
        "mib_per_expert": med([p["per_exp"] for p in per]),
        "non_layer_weight_mib": med([p["nonlayer"] for p in per]),
        "kv_kib_per_token_per_layer": (round(med([p["kv_tok_layer"] for p in per]) / 1024, 4)
                                       if med([p["kv_tok_layer"] for p in per]) else None),
        "kv_tokens": {str(p["L"]): p["kv_tokens"] for p in per},
        "device_total_mib": med([p["device_total"] / MiB if p["device_total"] else None for p in per]),
    }
    for key, name in (("device_non_kv", "device_non_kv"), ("peak_act", "activation_peak")):
        pts = {p["L"]: p[key] for p in per if p[key] is not None}
        out[f"{name}_mib"] = {str(k): round(v, 1) for k, v in sorted(pts.items())}
        if len(pts) >= 2:
            sl, ic = fit_per_layer(pts)
            out[f"{name}_mib_per_layer"] = round(sl, 2)
            out[f"{name}_mib_fixed"] = round(ic, 1)
    return out


def summarise(raw: Mapping, *, ref_experts: int = 128, ref_batch: int = 1,
              ref_context: str | int = "mean") -> dict:
    """各子进程的原始耗时 → 每层毫秒。raw = {"runs": [{"layers", "experts", "decode": {C: {B: ms}},
    "prefill": {P: ms}, "mem": …}, …], 其余字段原样带上}。"""
    by_e: dict[int, list] = {}
    for r in raw["runs"]:
        by_e.setdefault(int(r["experts"]), []).append(r)
    decode, prefill, overhead, memory = {}, {}, {}, {}
    for e, runs in sorted(by_e.items(), reverse=True):
        mem = summarise_memory(runs)
        if mem:
            memory[str(e)] = mem
        if len({int(r["layers"]) for r in runs}) < 2:
            continue
        cs = sorted({int(c) for r in runs for c in r["decode"]})
        decode[str(e)], overhead[str(e)] = {}, {}
        for c in cs:
            bs = sorted({int(b) for r in runs for b in (r["decode"].get(str(c)) or {})})
            for b in bs:
                pts = {int(r["layers"]): r["decode"][str(c)][str(b)] for r in runs
                       if str(b) in (r["decode"].get(str(c)) or {})}
                if len(pts) >= 2:
                    sl, ic = fit_per_layer(pts)
                    decode[str(e)].setdefault(str(c), {})[str(b)] = round(sl, 4)
                    overhead[str(e)].setdefault(str(c), {})[str(b)] = round(ic, 3)
        ps = sorted({int(p) for r in runs for p in r["prefill"]})
        prefill[str(e)] = {}
        for p in ps:
            pts = {int(r["layers"]): r["prefill"][str(p)] for r in runs if str(p) in r["prefill"]}
            if len(pts) >= 2:
                prefill[str(e)][str(p)] = round(fit_per_layer(pts)[0], 4)
    ref = ref_ms(decode, ref_experts=ref_experts, ref_batch=ref_batch, ref_context=ref_context)
    out = {k: v for k, v in raw.items() if k != "runs"}
    out.update(decode=decode, prefill=prefill, decode_overhead_ms=overhead, memory=memory,
               runs=raw["runs"],
               summary={"ms_per_layer": ref, "ref_experts": ref_experts, "ref_batch": ref_batch,
                        "ref_context": ref_context,
                        "definition": "decode 一步、每层、毫秒（ref_batch 条请求同时 decode 时每条请求每层"
                                      "等的时间；ref_context = mean 时对各上下文长度取均值）"})
    return out


def ref_ms(decode: Mapping, *, ref_experts: int = 128, ref_batch: int = 1,
           ref_context: str | int = "mean") -> float | None:
    """decode[E][C][B] → 给规划器的一个数。ref_context="mean"：各上下文长度取均值。"""
    by_c = decode.get(str(ref_experts)) or {}
    if str(ref_context) == "mean":
        v = _mean([d.get(str(ref_batch)) for d in by_c.values()])
    else:
        v = (by_c.get(str(ref_context)) or {}).get(str(ref_batch))
    return None if v is None else round(v, 4)


def _nested_mean(objs: Sequence) -> object:
    """若干个同构的嵌套 dict（叶子是数）→ 逐叶子取均值；有的台缺某个叶子时按有的算。"""
    vals = [o for o in objs if o is not None]
    if not vals:
        return None
    if all(isinstance(v, Mapping) for v in vals):
        keys = []
        for v in vals:
            keys += [k for k in v if k not in keys]
        return {k: _nested_mean([v.get(k) for v in vals]) for k in keys}
    nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if nums:
        m = sum(nums) / len(nums)
        return round(m, 4)
    return vals[0]


def aggregate(results: Sequence[Mapping], *, ref_experts: int | None = None, ref_batch: int | None = None,
              ref_context: str | int | None = None) -> dict:
    """多台的结果 → 按卡型取均值：{卡型: {n, nodes, decode, prefill, memory, ms_per_layer, …}}。"""
    by: dict[str, list] = {}
    for r in results:
        by.setdefault(gpu_key(r.get("gpu", "")), []).append(r)
    out = {}
    for g, rs in sorted(by.items()):
        sm = rs[0].get("summary") or {}
        re_ = ref_experts if ref_experts is not None else int(sm.get("ref_experts") or 128)
        rb = ref_batch if ref_batch is not None else int(sm.get("ref_batch") or 1)
        rc = ref_context if ref_context is not None else (sm.get("ref_context") or "mean")
        per = {str(r.get("node") or r.get("host")): ref_ms(r.get("decode") or {}, ref_experts=re_,
                                                          ref_batch=rb, ref_context=rc) for r in rs}
        vals = [v for v in per.values() if v is not None]
        out[g] = {
            "gpu": rs[0].get("gpu"), "n": len(rs), "nodes": sorted(per),
            "ms_per_layer": round(sum(vals) / len(vals), 4) if vals else None,
            "ms_per_layer_nodes": per,
            "slowest_over_fastest": round(max(vals) / min(vals), 3) if vals and min(vals) > 0 else None,
            "ref_experts": re_, "ref_batch": rb, "ref_context": rc,
            "decode": _nested_mean([r.get("decode") for r in rs]),
            "prefill": _nested_mean([r.get("prefill") for r in rs]),
            "decode_overhead_ms": _nested_mean([r.get("decode_overhead_ms") for r in rs]),
            "memory": _nested_mean([r.get("memory") for r in rs]),
        }
    return out


def topo_with_measurements(topo: Mapping, results: Sequence[Mapping], *, ref_batch: int | None = None,
                           ref_experts: int | None = None, ref_context: str | int | None = None,
                           per_node: bool = False, allow_missing: bool = False
                           ) -> tuple[dict, list[str]]:
    """拓扑 JSON + 若干台的实测结果 → 新拓扑（每个节点加 `ms_per_layer`，`compute_capacity` 按实测
    重算成同比例，原值留在 `compute_capacity_orig`）。返回 (新拓扑, 说明行)。

    默认每个节点用它这种卡型（拓扑的 role）所有已测节点的**均值**；`per_node` 时量过的节点用自己的、
    没量过的用卡型均值。某卡型一台都没量时报错（`allow_missing` 时按原 compute_capacity 相对最近的
    已测节点换算，并在说明里标出来）。"""
    agg = aggregate(results, ref_experts=ref_experts, ref_batch=ref_batch, ref_context=ref_context)
    bad = [f"{g}（{a['nodes']}）" for g, a in agg.items() if a["ms_per_layer"] is None]
    if bad:
        any_a = next(iter(agg.values()))
        raise ValueError(f"{bad} 的结果里没有 E={any_a['ref_experts']}、B={any_a['ref_batch']} 的 decode")
    by_node = {}
    for res in results:
        if res.get("node"):
            g = gpu_key(res.get("gpu", ""))
            by_node[str(res["node"])] = (agg[g]["ms_per_layer_nodes"].get(str(res["node"])), res)
    gpu_ms = {g: a["ms_per_layer"] for g, a in agg.items()}
    any_a = next(iter(agg.values())) if agg else {}
    rb_used, re_used, rc_used = any_a.get("ref_batch"), any_a.get("ref_experts"), any_a.get("ref_context")

    out = json.loads(json.dumps(topo))
    notes, missing = [], []
    nodes = out["nodes"]
    for n in nodes:
        g = gpu_key(n.get("role", ""))
        if per_node and n["id"] in by_node and by_node[n["id"]][0] is not None:
            ms, src = by_node[n["id"]][0], f"本机实测（{by_node[n['id']][1].get('gpu')}）"
        elif g in gpu_ms:
            ms, src = gpu_ms[g], f"{g} 均值（{agg[g]['n']} 台实测）"
        else:
            missing.append(n)
            continue
        n["ms_per_layer"] = round(ms, 4)
        n["ms_per_layer_src"] = src
    if missing:
        if not allow_missing:
            raise ValueError(f"{[n['id'] for n in missing]}（卡型 {sorted({n.get('role') for n in missing})}）"
                             "没有实测 —— 每种卡型至少量一台，或给 --allow-missing 按原算力比例换算")
        done = [n for n in nodes if "ms_per_layer" in n]
        if not done:
            raise ValueError("一台实测都没有对上拓扑里的节点（看 role 与 GPU 名）")
        for n in missing:
            # 拿算力最接近的已测节点做基准，按原 compute_capacity 反比换算
            base = min(done, key=lambda d: abs(float(d["compute_capacity"]) - float(n["compute_capacity"])))
            n["ms_per_layer"] = round(base["ms_per_layer"] * float(base["compute_capacity"])
                                      / float(n["compute_capacity"]), 4)
            n["ms_per_layer_src"] = f"未实测：按原算力比例由 {base['id']} 换算"
            notes.append(f"⚠ {n['id']}（{n.get('role')}）没有实测，按原算力比例由 {base['id']} 换算")
    top = max(float(n["compute_capacity"]) for n in nodes)
    fastest = min(n["ms_per_layer"] for n in nodes)
    for n in nodes:
        n.setdefault("compute_capacity_orig", n["compute_capacity"])
        # 旧版规划器只认 compute_capacity：按实测改成同比例，它算出来的相对快慢也对
        n["compute_capacity"] = round(top * fastest / n["ms_per_layer"], 1)
    out["ms_per_layer_meta"] = {
        "source": "vllm/p2p/layer_bench.py", "ref_batch": rb_used, "ref_experts": re_used,
        "ref_context": rc_used, "per_node": bool(per_node),
        "unit": "ms / 层 / token（decode）", "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpus": {g: round(v, 4) for g, v in sorted(gpu_ms.items())},
        "gpus_n": {g: a["n"] for g, a in sorted(agg.items())},
        # 规划器的显存账（ModelSpec 每层权重 / KV、A2b 的引擎开销）可以拿这个对：每层精确值 + 固定开销
        "memory_mib": {g: a["memory"] for g, a in sorted(agg.items())}}
    return out, notes


# --------------------------------------------------------------------------- #
# run：在本机 GPU 上量
# --------------------------------------------------------------------------- #
def _prepare_model_dir(src: Path, dst: Path) -> Path:
    """只拷 config 类文件：dummy 权重不读 safetensors，引擎也就不会去扫几十 GB 的目录。"""
    dst.mkdir(parents=True, exist_ok=True)
    if not (src / "config.json").is_file():
        raise SystemExit(f"{src} 下没有 config.json")
    for name in ("config.json", "generation_config.json"):
        if (src / name).is_file():
            (dst / name).write_bytes((src / name).read_bytes())
    return dst


def _overrides(n_layers: int, n_experts: int, total_experts: int) -> dict:
    ov = {"architectures": [ARCH], "num_hidden_layers": int(n_layers), "p2p_channel": "layer-bench"}
    if n_experts < total_experts:
        ov["p2p_layer_experts"] = {str(l): list(range(n_experts)) for l in range(n_layers)}
    return ov


def _time_generate(llm, prompts, sp) -> float:
    t0 = time.perf_counter()
    llm.generate(prompts, sp, use_tqdm=False)
    return (time.perf_counter() - t0) * 1e3


def measure(llm, *, batches: Sequence[int], contexts: Sequence[int], decode_steps: int,
            prefill_lens: Sequence[int], repeats: int, vocab: int, seed: int = 0,
            log: Callable[[str], None] = lambda s: None) -> dict:
    """一个已经起好的引擎上：{decode: {C: {B: 一步 decode 毫秒}}, prefill: {P: 毫秒}}。"""
    import random

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    rng = random.Random(seed)

    def prompts(n, length):
        return [TokensPrompt(prompt_token_ids=[rng.randrange(100, vocab - 100) for _ in range(length)])
                for _ in range(n)]

    def sp(n_out):
        return SamplingParams(temperature=0.0, max_tokens=n_out, ignore_eos=True, detokenize=False)

    out = {"decode": {}, "prefill": {}}
    for c in contexts:
        row = out["decode"][str(c)] = {}
        for b in batches:
            # 同一组 prompt 反复用：引擎开了前缀缓存，第一次之后这 B 条的 prompt 都命中缓存，
            # 「1 个 token」「1+K 个 token」两次各只重算最后不满一块的那几个位置。所以大 B × 长 C
            # 也不用把 B 条的 prefill 塞进一步（max_num_batched_tokens 不必 = B × C，否则 B=320、
            # C=1280 时 profile 一步 40 万 token 直接 OOM），两次之差仍只剩 K 步 decode。
            ps = prompts(b, c)
            _time_generate(llm, ps, sp(1 + min(decode_steps, 8)))   # 预热这个形状，顺带填好前缀缓存
            ds = []
            for _ in range(repeats):
                t1 = _time_generate(llm, ps, sp(1))
                tk = _time_generate(llm, ps, sp(1 + decode_steps))
                ds.append((tk - t1) / decode_steps)
            row[str(b)] = round(st.median(ds), 4)
        log(f"    decode C={c:<5} " + "  ".join(f"B={b}:{v:.3f}" for b, v in row.items()) + " ms/步")
    for p in prefill_lens:
        _time_generate(llm, prompts(1, p), sp(1))
        ps = [_time_generate(llm, prompts(1, p), sp(1)) for _ in range(repeats)]
        out["prefill"][str(p)] = round(st.median(ps), 4)
        log(f"    prefill P={p:<5} {out['prefill'][str(p)]:.2f} ms")
    return out


def _worker(argv: Sequence[str]) -> int:
    """子进程：起一个 L 层、每层 E 个专家的引擎，量完写 JSON。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--experts", type=int, required=True)
    ap.add_argument("--batches", required=True)
    ap.add_argument("--prefill", required=True)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--decode-steps", type=int, required=True)
    ap.add_argument("--repeats", type=int, required=True)
    ap.add_argument("--gpu-mem", type=float, required=True)
    ap.add_argument("--dtype", default="half")
    ap.add_argument("--attention-backend", default="TRITON_ATTN")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--max-batched-tokens", type=int, default=16384)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    # collective_rpc 要把 mem_probe 这个函数发给 worker（按模块路径引用，worker 里同样 import 得到）
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    from vllm import LLM

    cfg = json.loads((Path(a.model_dir) / "config.json").read_text(encoding="utf-8"))
    batches = [int(x) for x in a.batches.split(",") if x]
    prefill = [int(x) for x in a.prefill.split(",") if x]
    contexts = [int(x) for x in a.contexts.split(",") if x]
    mml = max([max(contexts) + a.decode_steps + 8, *(p + 8 for p in prefill)])
    kw = dict(model=a.model_dir, dtype=a.dtype, load_format="dummy", skip_tokenizer_init=True,
              hf_overrides=_overrides(a.layers, a.experts, int(cfg["num_experts"])),
              max_model_len=mml, max_num_seqs=max(batches), gpu_memory_utilization=a.gpu_mem,
              # 一步最多这么多 token：decode 的量法靠前缀缓存（见 measure），不需要 B 条的 prefill
              # 一步做完；上限压住 profile 时的激活峰值（B × C 很大时不至于 OOM）
              max_num_batched_tokens=max(mml, max(batches),
                                         min(a.max_batched_tokens, max(batches) * max(contexts))),
              enable_prefix_caching=True, enforce_eager=a.eager, seed=0,
              # 与分段引擎相同（stage_agent 关了异步调度）；也让一步 = CPU + GPU 串行，
              # 层数相减才能干净地只剩 GPU 上那几层
              async_scheduling=False)
    if a.attention_backend:
        kw["attention_backend"] = a.attention_backend
    t0 = time.perf_counter()
    llm = LLM(**kw)
    print(f"  [L={a.layers} E={a.experts}] 引擎起好 {time.perf_counter() - t0:.0f}s", flush=True)

    def probe():
        try:
            return llm.collective_rpc(mem_probe)[0]
        except Exception as e:                               # noqa: BLE001 — 显存账查不到不影响计时
            print(f"    （显存查询失败：{type(e).__name__}: {e}）", flush=True)
            return None
    m0 = probe()
    if m0:
        w = sum(l["bytes"] for l in m0["layers"]) / max(len(m0["layers"]), 1) / MiB
        print(f"    显存：每层权重 {w:.1f} MiB，层外权重 {m0['non_layer_param_bytes'] / MiB:.0f} MiB，"
              f"KV {m0['kv_bytes'] / MiB:.0f} MiB（{m0['kv_tokens']} token）", flush=True)
    res = measure(llm, batches=batches, contexts=contexts, decode_steps=a.decode_steps,
                  prefill_lens=prefill, repeats=a.repeats, vocab=int(cfg["vocab_size"]),
                  log=lambda s: print(s, flush=True))
    res.update(layers=a.layers, experts=a.experts, mem={"after_init": m0, "after_run": probe()})
    Path(a.out).write_text(json.dumps(res), encoding="utf-8")
    return 0


def _gpu_info() -> dict:
    try:
        q = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=name,memory.total,memory.used,"
                            "driver_version,clocks.max.sm", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        name, tot, used, drv, clk = [x.strip() for x in q.stdout.strip().split(",")[:5]]
        return {"gpu": name, "mem_total_mb": float(tot), "mem_used_mb": float(used),
                "driver": drv, "sm_clock_max_mhz": clk}
    except Exception as e:                                   # noqa: BLE001
        return {"gpu": "?", "error": f"nvidia-smi: {e}"}


def cmd_run(a) -> int:
    info = _gpu_info()
    if info.get("mem_used_mb", 0) > a.max_used_mb and not a.force:
        print(f"✗ 卡上已经占着 {info['mem_used_mb']:.0f} MiB（> {a.max_used_mb}）—— 别的引擎在跑，"
              "量出来的数不准。先停掉（launch.py … stop all），或 --force", file=sys.stderr)
        return 3
    layers = [int(x) for x in a.layers.split(",")]
    experts = [int(x) for x in a.experts.split(",")]
    prefill = parse_lengths(a.prefill or a.lengths)
    contexts = parse_lengths(a.contexts or a.lengths)
    cfg = json.loads((Path(a.model_dir) / "config.json").read_text(encoding="utf-8"))
    total_e = int(cfg["num_experts"])
    if any(e > total_e or e < int(cfg["num_experts_per_tok"]) for e in experts):
        raise SystemExit(f"--experts 要在 [{cfg['num_experts_per_tok']}, {total_e}] 之间")
    node = a.node or os.environ.get("P2P_NODE") or os.uname().nodename
    print(f"[layer_bench] {node}：{info.get('gpu')}，层数 {layers} × 专家 {experts}，batch {a.batches}，"
          f"decode 上下文 {contexts}，prefill {prefill}，{'eager' if a.eager else 'CUDA Graph'}", flush=True)
    env = dict(os.environ)
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    runs = []
    with tempfile.TemporaryDirectory(prefix="layer_bench_") as td:
        mdir = _prepare_model_dir(Path(a.model_dir), Path(td) / "model")
        for e in experts:
            for l in layers:
                out = Path(td) / f"L{l}_E{e}.json"
                cmd = [sys.executable, "-m", "vllm.p2p.layer_bench", "_worker", "--model-dir", str(mdir),
                       "--layers", str(l), "--experts", str(e), "--batches", a.batches,
                       "--prefill", ",".join(map(str, prefill)),
                       "--contexts", ",".join(map(str, contexts)),
                       "--decode-steps", str(a.decode_steps), "--repeats", str(a.repeats),
                       "--gpu-mem", str(a.gpu_mem), "--dtype", a.dtype,
                       "--attention-backend", a.attention_backend,
                       "--max-batched-tokens", str(a.max_batched_tokens), "--out", str(out)]
                if a.eager:
                    cmd.append("--eager")
                print(f"  → L={l} E={e} …", flush=True)
                p = subprocess.run(cmd, env=env, stdout=None if a.verbose else subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
                if p.returncode != 0 or not out.is_file():
                    tail = "\n".join((p.stdout or "").splitlines()[-30:])
                    print(f"✗ L={l} E={e} 失败（退出码 {p.returncode}）：\n{tail}", file=sys.stderr)
                    return 2
                if not a.verbose:
                    for ln in (p.stdout or "").splitlines():
                        if ln.startswith("  [") or ln.startswith("    "):   # 进度、计时、显存行
                            print(ln, flush=True)
                runs.append(json.loads(out.read_text(encoding="utf-8")))
    try:
        import vllm
        ver = getattr(vllm, "__version__", "?")
    except Exception:                                        # noqa: BLE001
        ver = "?"
    raw = {"node": node, "host": os.uname().nodename, **info, "vllm": ver, "dtype": a.dtype,
           "attention": a.attention_backend, "cuda_graph": not a.eager, "contexts": contexts,
           "prefill_lengths": prefill,
           "decode_steps": a.decode_steps, "repeats": a.repeats, "model": cfg.get("architectures"),
           "model_dir": str(a.model_dir), "layers_measured": layers, "runs": runs}
    res = summarise(raw, ref_experts=a.ref_experts, ref_batch=a.ref_batch, ref_context=a.ref_context)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print_table(res)
    print(f"已写 {a.out}")
    return 0


def print_table(res: Mapping, title: str | None = None) -> None:
    print(f"\n{title or (str(res.get('node')) + '  ' + str(res.get('gpu')))}  （每层毫秒）")
    for e, by_c in (res.get("decode") or {}).items():
        bs = sorted({int(b) for d in by_c.values() for b in d})
        print(f"  E={e:<4} decode  上下文\\batch " + "".join(f"{'B=' + str(b):>9}" for b in bs))
        for c, d in by_c.items():
            print(f"  {'':<7}         {c:>8} " + "".join(
                f"{d[str(b)]:>9.3f}" if str(b) in d else f"{'—':>9}" for b in bs))
        pf = (res.get("prefill") or {}).get(e) or {}
        if pf:
            print(f"  E={e:<4} prefill " + "  ".join(f"P={p}:{v:.2f}" for p, v in pf.items()))
    for e, m in (res.get("memory") or {}).items():
        def g(k, fmt="{:.1f}"):
            v = m.get(k)
            return "—" if v is None else fmt.format(v)
        print(f"  E={e:<4} 显存/层  权重 {g('weight_mib_per_layer')} MiB"
              f"（注意力+norm {g('attn_norm_mib_per_layer')}、路由 {g('router_mib_per_layer', '{:.2f}')}、"
              f"专家 {g('experts_mib_per_layer')} = {g('experts_per_layer', '{:.0f}')} × "
              f"{g('mib_per_expert', '{:.2f}')}）  KV {g('kv_kib_per_token_per_layer', '{:.2f}')} KiB/token")
        print(f"  E={e:<4} 显存/固定 层外权重 {g('non_layer_weight_mib', '{:.0f}')} MiB  "
              f"运行时峰值 每层 {g('activation_peak_mib_per_layer')} + 固定 {g('activation_peak_mib_fixed', '{:.0f}')} MiB  "
              f"整卡非 KV 每层 {g('device_non_kv_mib_per_layer')} + 固定 {g('device_non_kv_mib_fixed', '{:.0f}')} MiB"
              f"  （KV 可放 {m.get('kv_tokens')} token）")
    sm = res.get("summary") or res
    if sm.get("ms_per_layer") is not None:
        rc = sm.get("ref_context")
        print(f"  → 规划器用的 ms_per_layer = {sm.get('ms_per_layer')}（E={sm.get('ref_experts')}、"
              f"B={sm.get('ref_batch')}、{'各上下文均值' if str(rc) == 'mean' else f'上下文 {rc}'} 的 decode）")


# --------------------------------------------------------------------------- #
# cluster：每种卡型挑一台（或指定节点），ssh 过去跑 run
# --------------------------------------------------------------------------- #
def parse_hosts(path: str | Path) -> dict[str, tuple[str, str]]:
    """task/hosts.txt：`节点id  地址[:端口]  … # 卡型 …` → {节点: (地址, 卡型)}。"""
    out = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        body, _, comment = raw.partition("#")
        parts = body.split()
        if len(parts) < 2:
            continue
        host = parts[1].rsplit(":", 1)[0] if parts[1].count(":") == 1 else parts[1]
        out[parts[0]] = (host, comment.split()[0] if comment.split() else "")
    return out


def pick_nodes(hosts: Mapping[str, tuple[str, str]], want: str | None) -> list[str]:
    """`want`：None / "all" = 全部；"one-per-gpu" = 每种卡型第一台；否则逗号分隔的节点名。"""
    if want in (None, "all"):
        return sorted(hosts)
    if want != "one-per-gpu":
        names = [x.strip() for x in want.split(",") if x.strip()]
        bad = [x for x in names if x not in hosts]
        if bad:
            raise SystemExit(f"hosts 里没有 {bad}")
        return names
    seen, out = set(), []
    for n in sorted(hosts):
        g = gpu_key(hosts[n][1]) or n
        if g not in seen:
            seen.add(g)
            out.append(n)
    return out


def remote_command(node: str, a) -> str:
    args = ["run", "--model-dir", a.model_dir, "--node", node, "--out", f"/tmp/layer_bench_{node}.json",
            "--layers", a.layers, "--experts", a.experts, "--batches", a.batches,
            "--lengths", a.lengths, "--decode-steps", str(a.decode_steps),
            "--repeats", str(a.repeats), "--gpu-mem", str(a.gpu_mem),
            "--max-batched-tokens", str(a.max_batched_tokens), "--ref-batch", str(a.ref_batch),
            "--ref-experts", str(a.ref_experts), "--ref-context", str(a.ref_context)]
    if a.prefill:
        args += ["--prefill", a.prefill]
    if a.contexts:
        args += ["--contexts", a.contexts]
    if a.eager:
        args.append("--eager")
    if a.force:
        args.append("--force")
    return (f"cd {shlex.quote(a.repo)} && {shlex.quote(a.python)} -m vllm.p2p.layer_bench "
            + " ".join(shlex.quote(x) for x in args))


def cmd_cluster(a) -> int:
    hosts = parse_hosts(a.hosts)
    nodes = pick_nodes(hosts, a.nodes)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[layer_bench cluster] {len(nodes)} 台：" +
          "，".join(f"{n}（{hosts[n][1] or '?'}）" for n in nodes), flush=True)
    ok: dict[str, bool] = {}
    lock = threading.Lock()

    def one(n):
        host = hosts[n][0]
        cmd = remote_command(n, a)
        t0 = time.perf_counter()
        try:
            p = subprocess.run([*SSH, f"{a.user}@{host}", cmd], capture_output=True, text=True,
                               timeout=a.timeout)
        except subprocess.TimeoutExpired:
            with lock:
                print(f"✗ {n}：{a.timeout:.0f}s 没跑完", flush=True)
                ok[n] = False
            return
        log = (out_dir / f"{n}.log")
        log.write_text(p.stdout + p.stderr, encoding="utf-8")
        got = subprocess.run([*SSH, f"{a.user}@{host}", f"cat /tmp/layer_bench_{n}.json"],
                             capture_output=True, text=True, timeout=60)
        with lock:
            if p.returncode != 0 or got.returncode != 0:
                tail = "\n    ".join((p.stdout + p.stderr).strip().splitlines()[-8:])
                print(f"✗ {n}（{hosts[n][1]}）失败，退出码 {p.returncode}；日志 {log}\n    {tail}", flush=True)
                ok[n] = False
                return
            (out_dir / f"{n}.json").write_text(got.stdout, encoding="utf-8")
            res = json.loads(got.stdout)
            print(f"✓ {n}（{res.get('gpu')}）{time.perf_counter() - t0:.0f}s："
                  f"ms_per_layer = {res['summary']['ms_per_layer']}", flush=True)
            ok[n] = True

    ths = [threading.Thread(target=one, args=(n,)) for n in nodes]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    good = [n for n in nodes if ok.get(n)]
    print(f"\n{len(good)}/{len(nodes)} 台成功，每台的结果在 {out_dir}/<节点>.json")
    if good:
        results = [json.loads((out_dir / f"{n}.json").read_text(encoding="utf-8")) for n in good]
        agg = aggregate(results)
        (out_dir / "summary_by_gpu.json").write_text(json.dumps(agg, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
        print_gpu_summary(agg)
        print(f"\n按卡型的均值已写 {out_dir}/summary_by_gpu.json")
    return 0 if len(good) == len(nodes) else 2


def print_gpu_summary(agg: Mapping) -> None:
    print(f"\n{'卡型':<10}{'台数':>4}{'ms/层（均值）':>14}{'最慢/最快':>10}  各台")
    for g, a in agg.items():
        per = "  ".join(f"{n}:{v}" for n, v in sorted(a["ms_per_layer_nodes"].items()))
        print(f"{g:<10}{a['n']:>4}{a['ms_per_layer'] if a['ms_per_layer'] is not None else '—':>14}"
              f"{a['slowest_over_fastest'] or '—':>10}  {per}")
    for g, a in agg.items():
        print_table(a, title=f"{g} 均值（{a['n']} 台：{','.join(a['nodes'])}）")


# --------------------------------------------------------------------------- #
def cmd_topo(a) -> int:
    topo = json.loads(Path(a.topo).read_text(encoding="utf-8"))
    results = [json.loads(Path(p).read_text(encoding="utf-8")) for p in a.results]
    try:
        out, notes = topo_with_measurements(topo, results, ref_batch=a.ref_batch,
                                            ref_experts=a.ref_experts, ref_context=a.ref_context,
                                            per_node=a.per_node, allow_missing=a.allow_missing)
    except ValueError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2
    if Path(a.out).resolve() == Path(a.topo).resolve():
        print("✗ --out 不能覆盖原拓扑文件（原文件留作对照）", file=sys.stderr)
        return 2
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = out["ms_per_layer_meta"]
    print(f"按 E={meta['ref_experts']}、B={meta['ref_batch']}、上下文 {meta['ref_context']} 的 decode，"
          f"{'各台自己的值' if meta['per_node'] else '卡型均值'}：{meta['gpus']}（台数 {meta['gpus_n']}）")
    print(f"{'节点':<6}{'卡型':<10}{'ms/层':>8}{'原算力':>10}{'新算力':>10}  来源")
    for n in out["nodes"]:
        print(f"{n['id']:<6}{n.get('role', ''):<10}{n['ms_per_layer']:>8.3f}"
              f"{n['compute_capacity_orig']:>10.0f}{n['compute_capacity']:>10.0f}  {n['ms_per_layer_src']}")
    for s in notes:
        print(s)
    print(f"已写 {a.out}")
    return 0


def _common(ap) -> None:
    ap.add_argument("--model-dir", default="/home/ubuntu/p2p-framework/weights-q3-30b",
                    help="取 config.json 的目录（只读 config，不读权重）")
    ap.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)),
                    help="起几次引擎、各几层（至少两种；层多了 T4 显存不够）")
    ap.add_argument("--experts", default=",".join(map(str, DEFAULT_EXPERTS)),
                    help="每层驻留几个专家（128 = 前段全装；后段一般 55–95）")
    ap.add_argument("--batches", default=",".join(map(str, DEFAULT_BATCHES)),
                    help="decode 时同时几条请求")
    ap.add_argument("--lengths", default=DEFAULT_LENGTHS,
                    help="prefill 长度与 decode 上下文：起:止:步长 或 逗号列表（默认 128:1024:128）")
    ap.add_argument("--prefill", default=None, help="只改 prefill 长度（默认同 --lengths）")
    ap.add_argument("--contexts", default=None, help="只改 decode 上下文（默认同 --lengths）")
    ap.add_argument("--decode-steps", type=int, default=32, help="每次多生成几个 token 来量 decode")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-batched-tokens", type=int, default=16384,
                    help="引擎一步最多几个 token（vLLM 的 max_num_batched_tokens；大 batch × 长上下文时"
                         "别设太大，profile 激活会 OOM）")
    ap.add_argument("--eager", action="store_true", help="不用 CUDA Graph（部署时是用的）")
    ap.add_argument("--ref-batch", type=int, default=1, help="summary.ms_per_layer 取哪个 batch")
    ap.add_argument("--ref-experts", type=int, default=128, help="summary.ms_per_layer 取哪档专家数")
    ap.add_argument("--ref-context", default="mean",
                    help="summary.ms_per_layer 取哪个上下文长度；mean = 各长度均值（默认）")
    ap.add_argument("--force", action="store_true", help="卡上有别的进程也照量")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["_worker"]:
        return _worker(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="量本机 GPU（vllm-p2p 的 venv 里）")
    _common(r)
    r.add_argument("--node", default=None, help="节点名（默认主机名）")
    r.add_argument("--out", required=True)
    r.add_argument("--dtype", default="half")
    r.add_argument("--attention-backend", default="TRITON_ATTN")
    r.add_argument("--max-used-mb", type=float, default=1024, help="卡上已占超过这么多就拒绝量")
    r.add_argument("--verbose", action="store_true", help="把子进程的 vLLM 日志也打出来")

    c = sub.add_parser("cluster", help="控制机上：ssh 到各节点跑 run，结果收回来，按卡型取均值")
    _common(c)
    c.add_argument("--hosts", required=True, help="task/hosts.txt（行尾注释里写卡型）")
    c.add_argument("--nodes", default=None,
                   help="默认全部节点；one-per-gpu = 每种卡型第一台；或 N01,N09,N12")
    c.add_argument("--out-dir", default="results/layer_bench")
    c.add_argument("--repo", default="/home/ubuntu/vllm-p2p")
    c.add_argument("--python", default="/home/ubuntu/venvs/vllm-p2p/bin/python")
    c.add_argument("--user", default="ubuntu")
    c.add_argument("--timeout", type=float, default=7200)

    t = sub.add_parser("topo", help="把实测写进规划器的拓扑文件（写新文件）")
    t.add_argument("--topo", required=True, help="原拓扑，如 task/topo_fabric_mbps.json")
    t.add_argument("--results", nargs="+", required=True,
                   help="run / cluster 的每台结果 JSON（<out-dir>/N*.json；不要把 summary_by_gpu.json 也传进来）")
    t.add_argument("--out", required=True)
    t.add_argument("--ref-batch", type=int, default=None, help="默认用结果里的 summary.ref_batch")
    t.add_argument("--ref-experts", type=int, default=None)
    t.add_argument("--ref-context", default=None, help="mean（默认，各上下文均值）或某个长度")
    t.add_argument("--per-node", action="store_true",
                   help="量过的节点用自己的值（默认：每个节点都用它这种卡型的均值）")
    t.add_argument("--allow-missing", action="store_true",
                   help="没量到的卡型按原算力比例换算（默认报错）")

    a = ap.parse_args(argv)
    return {"run": cmd_run, "cluster": cmd_cluster, "topo": cmd_topo}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
