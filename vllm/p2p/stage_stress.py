# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""单卡压测一个分段引擎的一级：不要网络、不要别的节点、不要前段。

满载时后段流水线的中间级出过 Xid 31（GPU 读到未映射的显存）。这个脚本在**一台节点上**把那一级的
层（同样的层、同样的驻留专家、本机的部分权重）起成一个单卡引擎，自己造输入、自己当上下游：

* 中间级（不含最后一层）→ 起成中段（role=mid）：prompt 用随机 hidden 当 prompt_embeds 进去，之后
  每算完一个位置就 p2p_feed 下一个位置的随机 hidden —— 与流水线里这一级每步做的计算相同
  （注意力 + 部分专家 MoE + norm），只是没有 PP 的收发；
* 最后一级（含第 L 层）→ 起成后段（role=back）：每出一个 token 就喂下一个位置的随机 hidden。

在这里也崩 → 问题在这几层的算子（注意力 / MoE / 部分专家）本身；不崩 → 问题在 PP 级之间的收发
或多级流水线的调度。

    # 在 N07 上（权重目录就是这台节点本地的部分 checkpoint）
    cd /home/ubuntu/mbf
    PYTHONPATH=/home/ubuntu/mbf /home/ubuntu/venvs/vllm-p2p/bin/python -m vllm.p2p.stage_stress \\
        --stages p2p_channels/stages_color_m08.json --node N07 --concurrency 300 --minutes 15
    # 定位用：关 CUDA graph、同步报错（出错时调用栈指到真正出错的算子）
    CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=/home/ubuntu/mbf … -m vllm.p2p.stage_stress … --eager

也可以不用清单，直接给层与驻留集：`--layers 22,35 --experts-from <hf_overrides 的 JSON 文件>`。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["StressDriver", "plan_from_stages", "main"]


def plan_from_stages(sj: dict, node: str) -> dict:
    """清单 + 节点 → {engine, role, layers:(a,b), experts:{层: [专家]}, model_dir, task}。"""
    from vllm.p2p.weights import node_specs

    specs = node_specs(sj)
    if node not in specs:
        raise SystemExit(f"清单里没有节点 {node}（有 {sorted(specs)}）")
    sp = specs[node]["spec"]
    if len(sp["layers"]) != 1:
        raise SystemExit(f"{node} 上有 {len(sp['layers'])} 段层（多个引擎同卡），一次压一段：用 --layers 指定")
    a, b = sp["layers"][0]
    n_layers = int(sj["n_layers"])
    if a == 0:
        raise SystemExit(f"{node} 是前段（第 0 层起），这个脚本只压中段 / 后段的层")
    eng = next(c for c in sj["channels"] if any(st["node"] == node for st in c["commands"]))
    return {"engine": eng["name"], "role": "back" if b == n_layers else "mid", "layers": (a, b),
            "experts": {k: v for k, v in sp["experts"].items()}, "model_dir": sj["model_dir"],
            "task": eng.get("task")}


@dataclass
class _R:
    rid: str
    n_prompt: int
    steps: int                 # 要跑几个 decode 位置
    fed: int                   # 已经喂到哪个位置（不含）
    done: int = 0              # 已经算完的 decode 位置数
    t0: float = field(default_factory=time.perf_counter)


class StressDriver:
    """保持 `concurrency` 条请求在飞：prompt 用随机 hidden，之后每算完一个位置喂下一个位置。

    `engine` 是 tiered.Engine（真引擎或测试里的假引擎）。"""

    def __init__(self, engine, *, role: str, hidden: int, concurrency: int, prompt_range,
                 steps_range, scale: float = 1.0, seed: int = 0, dtype=None):
        import torch

        if role not in ("mid", "back"):
            raise ValueError(f"role={role!r}；只压中段 / 后段")
        self.e, self.role, self.H = engine, role, int(hidden)
        self.conc = int(concurrency)
        self.prompt_range, self.steps_range = prompt_range, steps_range
        self.scale = float(scale)
        self.rng = random.Random(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.dtype = dtype or torch.float16
        self.reqs: dict[str, _R] = {}
        self.n_add = self.n_done = self.n_fed = self.n_rows = 0
        self.bad_rows = 0
        self._ids = 0

    def _rows(self, n: int):
        import torch
        return (torch.randn(n, self.H, generator=self.gen) * self.scale).to(self.dtype)

    async def _add(self) -> None:
        from vllm.p2p.tiered import _core_request

        self._ids += 1
        rid = f"s{self._ids}"
        n = self.rng.randint(*self.prompt_range)
        steps = self.rng.randint(*self.steps_range)
        self.reqs[rid] = _R(rid, n, steps, fed=n)
        # 中段从不自己结束（同协调器里的中段）；后段按 steps 个 token 结束
        mt = steps + 1 if self.role == "back" else steps + 2
        await self.e.add(_core_request(rid, prompt=None, embeds=self._rows(n), max_tokens=mt, eos=None,
                                       logprobs=0, ignore_eos=True, stop_ids=(), arrival=time.time()))
        self.n_add += 1

    async def _feed(self, items: list) -> None:
        if items:
            await self.e.fire("p2p_feed", items)

    async def run(self, *, seconds: float | None = None, max_requests: int | None = None,
                  report_every: float = 10.0, log=print) -> dict:
        import torch

        from vllm.p2p.stage import pack_rows

        t_start = time.perf_counter()
        last, last_fed, last_rows = t_start, 0, 0
        while len(self.reqs) < self.conc and (max_requests is None or self.n_add < max_requests):
            await self._add()
        while self.reqs:
            outs = await self.e.outputs()
            feeds, finished = [], []
            for o in outs.outputs:
                r = self.reqs.get(o.request_id)
                if r is None:
                    continue
                if self.role == "mid":
                    if o.p2p_hidden is None:
                        continue
                    h = o.p2p_hidden
                    self.n_rows += h.shape[0]
                    if not torch.isfinite(h).all():
                        self.bad_rows += int((~torch.isfinite(h)).any(dim=-1).sum())
                    end = o.p2p_pos + h.shape[0]
                    if end < r.fed:
                        continue                     # prefill 分块 / 重算的旧位置
                    r.done = end - r.n_prompt
                    if r.done >= r.steps:
                        finished.append(r)
                    else:
                        feeds.append(pack_rows(r.rid, r.fed, self._rows(1)))
                        r.fed += 1
                else:
                    k = len(o.new_token_ids or ())
                    self.n_rows += k
                    r.done += k
                    if o.finish_reason is not None:
                        finished.append(r)
                    elif k:
                        feeds.append(pack_rows(r.rid, r.fed, self._rows(1)))
                        r.fed += 1
            self.n_fed += len(feeds)
            await self._feed(feeds)
            stop = seconds is not None and time.perf_counter() - t_start >= seconds
            if finished:
                if self.role == "mid":
                    await self.e.abort([r.rid for r in finished])
                for r in finished:
                    self.reqs.pop(r.rid, None)
                    self.n_done += 1
                while (not stop and len(self.reqs) < self.conc
                       and (max_requests is None or self.n_add < max_requests)):
                    await self._add()
            if stop and self.reqs:
                await self.e.abort(list(self.reqs))
                self.reqs.clear()
            now = time.perf_counter()
            if now - last >= report_every:
                dt = now - last
                log(f"[{now - t_start:7.0f}s] 在飞 {len(self.reqs):4d}  完成 {self.n_done:6d}  "
                    f"decode 位置/s {(self.n_rows - last_rows) / dt:8.0f}  喂入/s {(self.n_fed - last_fed) / dt:8.0f}"
                    + (f"  非有限行 {self.bad_rows}" if self.bad_rows else ""))
                last, last_fed, last_rows = now, self.n_fed, self.n_rows
        wall = time.perf_counter() - t_start
        return {"added": self.n_add, "done": self.n_done, "rows": self.n_rows, "fed": self.n_fed,
                "bad_rows": self.bad_rows, "wall_s": round(wall, 1),
                "rows_per_s": round(self.n_rows / max(wall, 1e-9), 1)}


def build_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stages", default=None, help="分段引擎清单（取这台节点那一级的层与驻留专家）")
    ap.add_argument("--node", default=None, help="压哪台节点那一级（配合 --stages）")
    ap.add_argument("--layers", default=None, help="不用清单时：0-based 半开区间 lo,hi")
    ap.add_argument("--experts-from", default=None,
                    help="不用清单时：含 p2p_layer_experts 的 JSON（hf_overrides）；不给 = 全专家")
    ap.add_argument("--model", default=None, help="权重目录（默认 = 清单的 model_dir）")
    ap.add_argument("--concurrency", type=int, default=300, help="同时在飞几条（≈ 满载时这一级的在飞数）")
    ap.add_argument("--prompt-min", type=int, default=64)
    ap.add_argument("--prompt-max", type=int, default=1024)
    ap.add_argument("--steps-min", type=int, default=64)
    ap.add_argument("--steps-max", type=int, default=512)
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--scale", type=float, default=1.0, help="随机 hidden 的标准差")
    ap.add_argument("--dtype", default="half")
    ap.add_argument("--attention-backend", default="TRITON_ATTN")
    ap.add_argument("--gpu-mem", type=float, default=0.8)
    ap.add_argument("--kv-bytes", type=float, default=None)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=None, help="默认 = --concurrency")
    ap.add_argument("--eager", action="store_true", help="关 CUDA graph")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


async def _amain(a) -> int:
    from vllm.p2p.tiered import make_engine

    if a.stages:
        if not a.node:
            raise SystemExit("--stages 要配 --node")
        plan = plan_from_stages(json.loads(Path(a.stages).read_text(encoding="utf-8")), a.node)
    else:
        if not a.layers:
            raise SystemExit("不用清单时要给 --layers lo,hi")
        lo, hi = (int(x) for x in a.layers.split(","))
        ov = json.loads(Path(a.experts_from).read_text()) if a.experts_from else {}
        le = ov.get("p2p_layer_experts", ov)
        cfg = json.loads((Path(a.model) / "config.json").read_text())
        plan = {"engine": "stress", "role": "back" if hi == int(cfg["num_hidden_layers"]) else "mid",
                "layers": (lo, hi), "experts": {k: v for k, v in le.items() if lo <= int(k) < hi},
                "model_dir": a.model, "task": None}
    model = a.model or plan["model_dir"]
    cfg = json.loads((Path(model) / "config.json").read_text())
    a.max_num_seqs = a.max_num_seqs or a.concurrency
    lo, hi = plan["layers"]
    print(f"压 {plan['engine']} 的这一级：{plan['role']} 层 [{lo}, {hi})，"
          f"{len(plan['experts'])} 层只装部分专家；权重 {model}；在飞 {a.concurrency}，"
          f"prompt {a.prompt_min}–{a.prompt_max}、decode {a.steps_min}–{a.steps_max} 步；"
          f"显存 {a.gpu_mem}{'，eager' if a.eager else ''}"
          f"{'，CUDA_LAUNCH_BLOCKING=1' if os.environ.get('CUDA_LAUNCH_BLOCKING') == '1' else ''}",
          flush=True)
    eng = make_engine(f"stress-{plan['engine']}", model, role=plan["role"], layers=(lo, hi),
                      layer_experts=plan["experts"], args=a, task=plan["task"])
    try:
        drv = StressDriver(eng, role=plan["role"], hidden=int(cfg["hidden_size"]),
                           concurrency=a.concurrency, prompt_range=(a.prompt_min, a.prompt_max),
                           steps_range=(a.steps_min, a.steps_max), scale=a.scale, seed=a.seed)
        res = await drv.run(seconds=a.minutes * 60)
        print(f"跑完了，没有崩：{json.dumps(res, ensure_ascii=False)}", flush=True)
        return 0
    finally:
        try:
            eng.client.shutdown()
        except Exception:   # noqa: BLE001
            pass


def main(argv=None) -> int:
    a = build_args(argv)
    # 单机单卡：分布式初始化走回环；引擎进程用 spawn（同 tiered.main）
    for k, v in (("VLLM_HOST_IP", "127.0.0.1"), ("GLOO_SOCKET_IFNAME", "lo"),
                 ("NCCL_SOCKET_IFNAME", "lo"), ("VLLM_WORKER_MULTIPROC_METHOD", "spawn")):
        os.environ.setdefault(k, v)
    return asyncio.run(_amain(a))


if __name__ == "__main__":
    sys.exit(main())
