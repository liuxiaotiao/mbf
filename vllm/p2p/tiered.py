# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B-1 分段引擎的协调器 + B0 原型（1 个前段 + 2 个同 task 后段，单卡、微型 checkpoint）。

    # vllm-0.14.0 目录下、装好本源码的环境里（与 parity 同一个）
    python -m vllm.p2p.tiered                         # fp16 + TRITON_ATTN，8 并发
    python -m vllm.p2p.tiered --dtype float32         # 近乎逐位的对拍
    python -m vllm.p2p.tiered --requests 32 --concurrency 16 --max-tokens 128 --out b0.json
    # 多级：前段 [0,4) → 中段 [4,6) ×2 → 后段 [6,8) ×2（4 条路线，中段与后段多对多），开 KV 守卫
    python -m vllm.p2p.tiered --mid-splits 6 --mid-copies 2 --kv-guard --out b0_mid.json

一条请求的一生（协议见 `vllm/p2p/stage.py`）：

    1. 进前段引擎（token prompt）。前段 prefill，交出 prompt 每个位置的残差流 hidden。
    2. prompt 的 hidden 收齐 → **绑定**一个后段 → 以 prompt_embeds 进那个后段。后段 prefill，
       采样出第 1 个 token。绑哪个：先定 task —— oracle 用请求自带的；`route="lr"` 时用前段在
       prompt 算完那一批输出里交出的前 L₀ 层路由统计（`p2p_feat`）跑 `classify` 判出来的 ——
       再在这个 task 的后段里挑负载最轻的。
    3. 循环：后段的 token ──p2p_append──▶ 前段算这一位置的 hidden ──p2p_feed──▶ 后段采下一个。
    4. 后段结束（EOS / 长度）→ 前段那条 abort。

绑定后不再换（换绑不在本阶段范围）；同一个前段、同一 task 的请求可以落到不同的后段。

多级（中段）：绑定的不是一个后段，而是一条**路线**（`Route`）：0 个或多个中段 + 1 个后段。
前段 → 中段₁ → … → 后段，每一段的 prompt hidden 收齐后以 prompt_embeds 进下一段（逐段 prefill），
之后每个位置的 hidden 一段段往下传（直连时由各段代理直接转），后段的 token 回到前段。
同一 task 可以有多条路线，按 `weight`（规划给的在飞份额）分请求；不同路线可以共用中段 / 后段
（分叉、汇合）。路线里同一个引擎只出现一次 —— 一个 token 的一生里不会回到已经离开的段。

B0 验什么
---------
* **正确性**：两个后段默认装**不同**的专家子集；每种驻留集各起一个「整模型」引擎（同一份驻留集的
  Qwen3MoePartial、不分段），每条分段结果对着**它被绑到的那个后段**的整模型逐条比 token 与 logprob。判据同 parity：公共前缀上 max |Δlogprob| ≤ tol，分叉处必须是
  近似平票（整模型自己的 top-k 里两个选择的 logprob 差 ≤ 2·tol）。
* **开销**：同并发下 TTFT / TPOT 与整模型对比；每个 decode 位置拆成「前段腿」（后段出 token →
  前段 hidden 回到协调器）与「后段腿」（hidden 送出 → 下一个 token 回来）。
* **按请求绑定**：两个后段各分到多少条。

数据路径（原型，后面再优化）：前段 worker → 前段 EngineCore 输出 → 协调器 → 后段 EngineCore
utility → SchedulerOutput → 后段 worker。每个 decode 位置一来一回两次 ZMQ。
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import random
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from vllm.p2p.stage import RowStream, pack_rows

__all__ = ["Engine", "TieredCoordinator", "MonoRunner", "compare_runs", "main"]


# --------------------------------------------------------------------------- #
# 一个引擎（EngineCore 的 AsyncMPClient）的薄封装
# --------------------------------------------------------------------------- #
@dataclass
class Engine:
    name: str
    role: str | None                 # "front" / "mid" / "back" / None（整模型）
    client: Any                      # vllm.v1.engine.core_client.AsyncMPClient（测试里是假的）
    task: str | None = None          # 后段服务的 task（绑定时只在同 task 的后段里挑）
    inflight: int = 0
    n_bound: int = 0
    errors: list = field(default_factory=list)
    on_error: Callable[[str], None] | None = None
    # 远端引擎报的每 decode 位置「引擎内毫秒」的滑动平均（选前段时按它算快慢）；0 = 还没数据
    ewma_ms: float = 0.0
    # KV 守卫（协调器 kv_guard=True 时用）：本引擎可用的 KV 块数（None = 不限）、块大小、
    # 已经给在飞请求预留的块数
    kv_blocks: int | None = None
    block_size: int = 16
    kv_reserved: int = 0

    def kv_need(self, n_tokens: int) -> int:
        return -(-int(n_tokens) // self.block_size)

    def kv_fits(self, n_tokens: int) -> bool:
        return self.kv_blocks is None or self.kv_reserved + self.kv_need(n_tokens) <= self.kv_blocks

    def kv_ever_fits(self, n_tokens: int) -> bool:
        return self.kv_blocks is None or self.kv_need(n_tokens) <= self.kv_blocks

    def observe(self, ms: float, alpha: float = 0.05) -> None:
        self.ewma_ms = ms if self.ewma_ms <= 0 else (1 - alpha) * self.ewma_ms + alpha * ms

    async def add(self, req) -> None:
        await self.client.add_request_async(req)

    async def abort(self, ids: list[str]) -> None:
        if ids:
            await self.client.abort_requests_async(ids)

    async def outputs(self):
        return await self.client.get_output_async()

    async def fire(self, method: str, *args) -> None:
        """发一个 utility 调用、**不等结果**（等的话要等对面这一步算完，整条流水线就串行了）。
        失败记进 errors，由协调器统一抛。"""
        fire = getattr(self.client, "p2p_fire", None)
        if fire is not None:                         # 测试用的假引擎
            fut = await fire(method, *args)
        else:
            fut = await _fire_utility(self.client, method, *args)

        def done(f):
            if not f.cancelled() and f.exception() is not None:
                msg = f"{self.name}.{method}: {f.exception()}"
                self.errors.append(msg)
                if self.on_error is not None:
                    self.on_error(msg)
        fut.add_done_callback(done)


async def _fire_utility(client, method: str, *args):
    """`AsyncMPClient._call_utility_async` 去掉最后的 `await future`。"""
    from vllm.v1.engine import EngineCoreRequestType

    call_id = uuid.uuid1().int >> 64
    fut = asyncio.get_running_loop().create_future()
    client.utility_results[call_id] = fut
    msg = (EngineCoreRequestType.UTILITY.value,
           *client.encoder.encode((client.client_index, call_id, method, args)))
    await client._send_input_message(msg, client.core_engine, args)
    client._ensure_output_queue_task()
    return fut


def _core_request(rid: str, *, prompt: list[int] | None, embeds: torch.Tensor | None,
                  max_tokens: int, eos: int | None, logprobs: int, ignore_eos: bool,
                  stop_ids: Sequence[int], arrival: float, extra_args: dict | None = None):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreRequest

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, detokenize=False,
                        logprobs=logprobs or None, ignore_eos=ignore_eos,
                        stop_token_ids=list(stop_ids) or None, extra_args=extra_args or None)
    return EngineCoreRequest(
        request_id=rid, prompt_token_ids=prompt, mm_features=None, sampling_params=sp,
        pooling_params=None, eos_token_id=eos, arrival_time=arrival, lora_request=None,
        cache_salt=None, data_parallel_rank=None, prompt_embeds=embeds)


def _lp_rows(lp) -> list[dict[int, float]]:
    """EngineCoreOutput.new_logprobs（LogprobsLists）→ 每个新 token 一个 {token: logprob}。"""
    if lp is None:
        return []
    ids, vals = lp.logprob_token_ids, lp.logprobs
    return [{int(t): float(v) for t, v in zip(ids[j], vals[j])} for j in range(len(ids))]


@dataclass(eq=False)
class Route:
    """一条路线：前段之后依次经过的引擎（0 个或多个中段 + 最后 1 个后段）。

    `weight`：这条路线该分到的在飞请求份额（同 task 的路线之间按它分；规划器给的是
    「流量 × 循环时间」，即 Little 定律下的在飞数）。"""
    hops: list
    task: str | None = None
    weight: float = 1.0
    name: str = ""
    inflight: int = 0
    n_bound: int = 0

    def __post_init__(self):
        self.hops = list(self.hops)
        if not self.hops:
            raise ValueError("路线至少要有一个后段")
        if self.weight <= 0:
            raise ValueError(f"路线 {self.name or self.label} 的 weight 要 > 0（给的是 {self.weight}）")
        names = [e.name for e in self.hops]
        if len(set(names)) != len(names):
            raise ValueError(f"路线里同一个引擎出现了两次（token 不能回到离开过的段）：{names}")
        for e in self.hops[:-1]:
            if e.role not in (None, "mid"):
                raise ValueError(f"路线 {names}：{e.name} 是 {e.role}，后段之前只能是中段")
        if self.hops[-1].role not in (None, "back"):
            raise ValueError(f"路线 {names}：最后一段 {self.hops[-1].name} 是 {self.hops[-1].role}，要是后段")
        if self.task is None:
            self.task = self.hops[-1].task
        if not self.name:
            self.name = self.label

    @property
    def back(self):
        return self.hops[-1]

    @property
    def mids(self) -> list:
        return self.hops[:-1]

    @property
    def label(self) -> str:
        return " → ".join(e.name for e in self.hops)


# --------------------------------------------------------------------------- #
# 协调器
# --------------------------------------------------------------------------- #
class _Req:
    __slots__ = ("rid", "prompt", "n_prompt", "max_tokens", "task", "front", "back", "stream",
                 "tokens", "lps", "t_arrive", "t_front", "t_bound", "t_first", "t_last", "t_tok",
                 "t_hid", "finish", "done", "events", "info", "eng_f", "eng_b", "feat", "body",
                 "true_task", "route", "hops", "streams", "added", "eng_m", "t_hop", "kv",
                 "t_admit")

    def __init__(self, rid, prompt, max_tokens, task, loop, body=None):
        self.rid, self.prompt, self.max_tokens, self.task = rid, prompt, max_tokens, task
        self.true_task = task                # 请求自带的 task（lr 模式下 task 会换成判出来的）
        self.body = body                     # body token 区间（前段只数这些位置的路由）
        self.feat: dict | None = None        # 前段交出的前 L₀ 层路由统计（prompt 算完时）
        self.n_prompt = len(prompt)
        self.front: Engine | None = None
        self.back: Engine | None = None
        self.stream = RowStream()           # 前段交出的 hidden（= streams[0]）
        # 多级：hops = [前段, 中段…, 后段]（绑定后才完整）；streams[k] = 第 k 段交出的 hidden；
        # added[k] = 第 k 段是否已经收下这条请求（逐段 prefill）
        self.route: Route | None = None
        self.hops: list = []
        self.streams: list[RowStream] = [self.stream]
        self.added: list[bool] = [True]
        self.eng_m: dict[str, list[float]] = {}     # 各中段的「引擎内毫秒」
        self.t_hop: dict[str, float] = {}           # 各中段收齐 prompt hidden、交给下一段的时刻
        self.kv: list = []                          # KV 守卫：[(引擎, 预留块数), …]
        self.t_admit = None                         # KV 守卫：进前段的时刻（之前在协调器排队）
        self.tokens: list[int] = []
        self.lps: list[dict] = []
        self.t_arrive = time.perf_counter()
        self.t_front = self.t_bound = self.t_first = self.t_last = None
        self.t_tok: list[float] = []         # 第 k 个 token 回到协调器的时刻
        self.t_hid: list[float] = []         # 第 k 个 token 的前段 hidden 回到协调器的时刻
        self.finish = None
        self.done: asyncio.Future = loop.create_future()
        # 流式：("tok", [token…]) …，最后 ("done", result)；出错时 ("err", 异常)
        self.events: asyncio.Queue = asyncio.Queue()
        self.info: dict = {}                 # 绑定时的附加信息（B1 的识别结果放这里）
        # 远端引擎（remote.py）报的「引擎内毫秒」：每个 decode 位置前段 / 后段各一个
        self.eng_f: list[float] = []
        self.eng_b: list[float] = []

    def hop_of(self, e) -> int | None:
        """e 是这条请求路线上的第几段（前段 0）；不在路线上 → None。"""
        for k, x in enumerate(self.hops):
            if x is e:
                return k
        return None

    def result(self) -> dict:
        n = len(self.tokens)
        ttft = (self.t_first - self.t_arrive) * 1e3 if self.t_first else None
        hop_ms = {k: round((t - self.t_bound) * 1e3, 3) for k, t in self.t_hop.items()
                  if self.t_bound}
        tpot = ((self.t_last - self.t_first) * 1e3 / (n - 1)) if n > 1 else None
        legs_f = [(h - t) * 1e3 for t, h in zip(self.t_tok, self.t_hid)]
        legs_b = [(t - h) * 1e3 for h, t in zip(self.t_hid, self.t_tok[1:])]
        return {
            "rid": self.rid, "front": self.front.name if self.front else None,
            "back": self.back.name if self.back else None, "task": self.task,
            "mids": [e.name for e in self.hops[1:-1]],
            "route": self.route.name if self.route else None,
            "true_task": self.true_task,
            "n_prompt": self.n_prompt, "tokens": self.tokens, "logprobs": self.lps,
            "finish": self.finish, "ttft_ms": ttft, "tpot_ms": tpot,
            "front_prefill_ms": (self.t_front - self.t_arrive) * 1e3 if self.t_front else None,
            **({"admit_wait_ms": (self.t_admit - self.t_arrive) * 1e3} if self.t_admit else {}),
            "bind_ms": (self.t_bound - self.t_front) * 1e3 if self.t_bound and self.t_front else None,
            **({"mid_prefill_done_ms": hop_ms} if hop_ms else {}),
            "front_leg_ms": legs_f, "back_leg_ms": legs_b,
            "front_engine_ms": self.eng_f, "back_engine_ms": self.eng_b,
            **({"mid_engine_ms": self.eng_m} if self.eng_m else {}),
            **({"info": self.info} if self.info else {}),
        }


class TieredCoordinator:
    """若干前段 + 若干后段。`generate()` / `stream()` 可以并发调用。

    前段：每条请求挑一个前段（任何 task 都能用任何前段；`front_pick` 见 `_pick_front`）。
    后段：前段 prefill 完以后先定 task、再在**这个 task** 的后段里挑 —— `bind`：`least`（在飞最少，
    平手轮流）/ `rr`（轮流）/ 一个函数 `f(req_info: dict, candidates: list[Engine]) -> Engine`。
    task 没有对应后段时用全部后段。

    识别（`classify(feat: dict, req_info: dict) -> verdict dict`，至少有 "task"）：前段在 prompt
    算完的那一批输出里交出前 L₀ 层的逐层路由统计（`EngineCoreOutput.p2p_feat`），绑定前调它。
    `route="task"`（默认）：只记下结果（shadow，绑定仍按请求自带的 task）；`route="lr"`：按判出的
    task 绑定。结果放在 `result()["info"]["verdict"]`。

    `front_lr=True`：LR 在前段代理本机跑（`stage_agent` 的 `p2p_lr`），前段交来的已经是
    `{"verdict": …}`，协调器直接用、不再调 `classify`。

    `kv_guard=True`：按各引擎的 KV 容量（`Engine.kv_blocks`）给每条在飞请求预留最坏情况
    （prompt + max_tokens）的块，装不下就先不进：前段装不下 → `submit` 在协调器排队；路线上
    装不下 → 前段 prefill 完先停在协调器（「停靠」），有请求结束再绑。这样各段都不会被抢占。
    多级时要开：一条请求要所有段都往前走才能结束，各段 KV 都满时会互相等（中段等后段腾地方、
    后段等中段重算），vLLM 自己的抢占解不开。

    `routes`（多级）：`Route` 的列表；给了就不看 `backs`（后段、中段都从路线里取）。不给 = 每个
    后段一条只有它自己的路线（两级，旧行为）。定了 task 以后在这个 task 的路线里挑：`least`
    按 (在飞 + 1) / weight 最小（平手轮流）；`rr` 轮流；函数 `bind(req_info, 候选后段)` 返回一个
    后段，再在以它结尾的候选路线里挑在飞最少的。"""

    def __init__(self, fronts: Engine | Sequence[Engine], backs: Sequence[Engine], *,
                 bind: str | Callable = "least", eos: int | None = None,
                 stop_ids: Sequence[int] = (), ignore_eos: bool = False, logprobs: int = 0,
                 front_pick: str = "speed", direct: bool = False,
                 classify: Callable[[dict, dict], dict] | None = None, route: str = "task",
                 front_lr: bool = False, routes: Sequence[Route] | None = None,
                 kv_guard: bool = False):
        fronts = [fronts] if isinstance(fronts, Engine) else list(fronts)
        if front_pick not in ("speed", "least"):
            raise ValueError(f"front_pick={front_pick!r}；可选 speed / least")
        self.front_pick = front_pick
        if route not in ("task", "lr"):
            raise ValueError(f"route={route!r}；可选 task / lr")
        if route == "lr" and classify is None and not front_lr:
            raise ValueError('route="lr" 要给 classify（或 front_lr=True：前段本机判）')
        self.classify, self.route = classify, route
        self.front_lr = bool(front_lr)
        # 直连（remote.AgentBridge 的 K_ROUTE）：绑定后每 token 的 hidden / token 由前后段代理
        # 直接互发，协调器只收 token（流式、判结束）。要求所有引擎都是远端代理。
        self.direct = bool(direct)
        if not fronts:
            raise ValueError("至少要一个前段")
        if routes is None:
            if not backs:
                raise ValueError("至少要一个后段")
            routes = [Route([b], task=b.task, name=b.name) for b in backs]
        routes = list(routes)
        if not routes:
            raise ValueError("至少要一条路线")
        self.routes = routes
        def uniq(xs):                                  # 按出现顺序去重（Engine 不可哈希，按 id）
            return list({id(e): e for e in xs}.values())
        self.backs = uniq(rt.back for rt in routes)
        self.mids = uniq(e for rt in routes for e in rt.mids)
        clash = {e.name for e in self.backs} & {e.name for e in self.mids}
        if clash or len({e.name for e in (*fronts, *self.mids, *self.backs)}) != \
                len(fronts) + len(self.mids) + len(self.backs):
            raise ValueError(f"引擎名重复，或同一个引擎既当中段又当后段：{sorted(clash)}")
        if direct and not all(hasattr(e.client, "send_route")
                              for e in (*fronts, *self.mids, *self.backs)):
            raise ValueError("direct=True 要所有引擎都是远端代理（RemoteClient）")
        self.fronts = fronts
        self.kv_guard = bool(kv_guard)
        self._kv_ev = asyncio.Event()       # 有预留退回时 set（再换一个新的）
        self._parked: list[_Req] = []       # 前段 prefill 完、路线上 KV 不够、等着绑定的请求
        self._bg: set[asyncio.Task] = set()   # _unpark 等后台任务（close 时一起取消）
        self.bind, self.eos, self.stop_ids = bind, eos, tuple(stop_ids)
        self.ignore_eos, self.logprobs = ignore_eos, logprobs
        self.reqs: dict[str, _Req] = {}
        self._rr: dict[Any, itertools.count] = defaultdict(itertools.count)
        self._tasks: list[asyncio.Task] = []
        self._ids = itertools.count()
        for e in self.engines:
            e.on_error = lambda msg: self._fail_all(RuntimeError(f"分段引擎 utility 调用失败：{msg}"))

    @property
    def engines(self) -> list[Engine]:
        return [*self.fronts, *self.mids, *self.backs]

    @property
    def front(self) -> Engine:                      # 只有一个前段时的简写（B0 脚本与测试用）
        return self.fronts[0]

    # ---- 生命周期 ----
    def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._tasks = [loop.create_task(self._front_loop(f), name=f"p2p-{f.name}")
                       for f in (*self.fronts, *self.mids)]
        self._tasks += [loop.create_task(self._back_loop(b), name=f"p2p-{b.name}")
                        for b in self.backs]

    async def close(self) -> None:
        for t in (*self._tasks, *self._bg):
            t.cancel()
        await asyncio.gather(*self._tasks, *self._bg, return_exceptions=True)
        self._tasks = []
        self._bg.clear()

    def _fail_all(self, exc: BaseException) -> None:
        for r in list(self.reqs.values()):
            if not r.done.done():
                r.done.set_exception(exc)
                r.events.put_nowait(("err", exc))

    def _check_errors(self) -> None:
        errs = [e for x in self.engines for e in x.errors]
        if errs:
            raise RuntimeError("分段引擎 utility 调用失败：\n  " + "\n  ".join(errs[:5]))

    def state(self) -> dict:
        return {"fronts": [{"name": f.name, "inflight": f.inflight,
                            "engine_ms_ewma": round(f.ewma_ms, 2)} for f in self.fronts],
                "backs": [{"name": b.name, "task": b.task, "inflight": b.inflight,
                           "bound": b.n_bound, "engine_ms_ewma": round(b.ewma_ms, 2)}
                          for b in self.backs],
                "mids": [{"name": m.name, "task": m.task, "inflight": m.inflight,
                          "bound": m.n_bound, "engine_ms_ewma": round(m.ewma_ms, 2)}
                         for m in self.mids],
                "routes": [{"name": rt.name, "task": rt.task, "hops": [e.name for e in rt.hops],
                            "weight": rt.weight, "inflight": rt.inflight, "bound": rt.n_bound}
                           for rt in self.routes],
                **({"kv": {e.name: [e.kv_reserved, e.kv_blocks] for e in self.engines},
                    "parked": len(self._parked)} if self.kv_guard else {})}

    # ---- 请求 ----
    async def submit(self, prompt: list[int], max_tokens: int, *, task: str | None = None,
                     rid: str | None = None, body=None) -> _Req:
        """把一条请求送进前段，返回它的状态对象（`events` 队列里陆续有 token）。
        `body`：body token 区间 [[a, b], …]（prompt 下标）—— 前段的识别统计只数这些位置。"""
        self._check_errors()
        rid = rid or f"t{next(self._ids)}"
        if rid in self.reqs:
            raise ValueError(f"请求号 {rid} 正在用")
        body = [(int(a), int(b)) for a, b in body] if body else None
        r = _Req(rid, list(prompt), max_tokens, task, asyncio.get_running_loop(), body=body)
        if self.kv_guard:
            f = await self._admit_front(r)
            if rid in self.reqs:
                self._kv_release(r)
                raise ValueError(f"请求号 {rid} 正在用")
            r.t_admit = time.perf_counter()
        else:
            f = self._pick_front()
        self.reqs[rid] = r
        r.front = f
        r.hops = [f]
        f.inflight += 1
        xa = {"p2p_id": rid}
        if body:
            xa["p2p_body"] = ",".join(f"{a}:{b}" for a, b in body)
        # 前段这条永远不会自己结束（不采样、不查停止条件）：max_tokens 只是 Request 的必填项
        try:
            await f.add(_core_request(
                rid, prompt=list(prompt), embeds=None, max_tokens=max_tokens + 1, eos=None,
                logprobs=0, ignore_eos=True, stop_ids=(), arrival=time.time(), extra_args=xa))
        except BaseException:
            # 送的时候被取消 / 出错：记账退回（stream() 在 submit 返回前不会替我们 cancel）
            if self.reqs.get(rid) is r:
                self.reqs.pop(rid, None)
                f.inflight -= 1
                self._kv_release(r)
                asyncio.ensure_future(f.abort([rid]))
            raise
        return r

    async def cancel(self, r: _Req) -> None:
        """客户端走了 / 超时：路线上各段都 abort（还没收下的段也发：它会丢掉先到的 hidden），
        记账退回。"""
        if r.rid not in self.reqs:
            return
        self.reqs.pop(r.rid, None)
        if r in self._parked:
            self._parked.remove(r)
        for e in r.hops:
            if r.finish is None:
                e.inflight -= 1
        if r.route is not None and r.finish is None:
            r.route.inflight -= 1
        try:
            for e in r.hops:
                await e.abort([r.rid])
        finally:
            # 预留在 abort 发出去之后才退：否则等着的请求可能先于 abort 进引擎，短暂超出预留
            self._kv_release(r)

    async def stream(self, prompt: list[int], max_tokens: int, *, task: str | None = None,
                     rid: str | None = None, timeout: float | None = None, body=None):
        """异步生成器：先产出 ("req", 请求状态对象)，再逐批 ("tok", [token…])，最后
        ("done", result)。`timeout` 是两次产出之间最多等多久（不是整条的总时长）。"""
        r = await self.submit(prompt, max_tokens, task=task, rid=rid, body=body)
        try:
            yield ("req", r)
            while True:
                try:
                    ev = await asyncio.wait_for(r.events.get(), timeout)
                except asyncio.TimeoutError:
                    raise TimeoutError(f"{r.rid} {timeout:.0f}s 没有进展：{self.describe(r.rid)}") \
                        from None
                if ev[0] == "err":
                    raise ev[1]
                yield ev
                if ev[0] == "done":
                    return
        finally:
            if r.finish is None:
                await self.cancel(r)
            self.reqs.pop(r.rid, None)

    async def generate(self, prompt: list[int], max_tokens: int, *, task: str | None = None,
                       rid: str | None = None, timeout: float | None = None, body=None) -> dict:
        async for ev in self.stream(prompt, max_tokens, task=task, rid=rid, timeout=timeout,
                                    body=body):
            if ev[0] == "done":
                return ev[1]
        raise RuntimeError("流在 done 之前结束了")

    def describe(self, rid: str) -> str:
        r = self.reqs.get(rid)
        if r is None:
            return "（已结束）"
        mids = "".join(f" mid={e.name}（收下 {r.added[k]}，hidden 收到 {r.streams[k].next} 位）"
                       for k, e in enumerate(r.hops[1:-1], 1))
        return (f"front={r.front.name if r.front else None} "
                f"back={r.back.name if r.back else None} hidden 收到 {r.stream.next} 位"
                f"（prompt {r.n_prompt}）{mids} token {len(r.tokens)} 个 hidden 回 {len(r.t_hid)} 次")

    # ---- KV 守卫 ----
    def _kv_tokens(self, r: _Req) -> int:
        return r.n_prompt + r.max_tokens

    def _kv_reserve(self, r: _Req, engines) -> None:
        n = self._kv_tokens(r)
        for e in engines:
            nb = e.kv_need(n)
            e.kv_reserved += nb
            r.kv.append((e, nb))

    def _kv_release(self, r: _Req) -> None:
        if not r.kv:
            return
        for e, nb in r.kv:
            e.kv_reserved -= nb
        r.kv = []
        ev, self._kv_ev = self._kv_ev, asyncio.Event()
        ev.set()
        if self._parked:
            t = asyncio.get_running_loop().create_task(self._unpark())
            self._bg.add(t)
            t.add_done_callback(self._bg.discard)

    async def _admit_front(self, r: _Req) -> Engine:
        n = self._kv_tokens(r)
        if not any(f.kv_ever_fits(n) for f in self.fronts):
            raise ValueError(f"{r.rid}：prompt {r.n_prompt} + max_tokens {r.max_tokens} 超过了"
                             "每个前段的 KV 容量")
        while True:
            fits = [f for f in self.fronts if f.kv_fits(n)]
            if fits:
                f = self._pick_front(fits)
                self._kv_reserve(r, [f])
                return f
            await self._kv_ev.wait()

    async def _unpark(self) -> None:
        for r in list(self._parked):
            if r not in self._parked:
                continue                    # 另一个 _unpark 已经绑了
            if r.rid not in self.reqs:
                self._parked.remove(r)
                continue
            rt = self._pick(r)
            if rt is not None:
                self._parked.remove(r)
                try:
                    await self._bind_to(r, rt)
                except asyncio.CancelledError:
                    raise
                except Exception as e:              # noqa: BLE001 —— 只让这一条失败
                    if not r.done.done():
                        r.done.set_exception(e)
                    r.events.put_nowait(("err", e))

    def _pick_front(self, fronts: list[Engine] | None = None) -> Engine:
        """`least`：在飞最少。`speed`（默认）：(在飞 + 1) × 该前段每位置的引擎内耗时最小 ——
        前段的卡型不同（A30 vs RTX 6000，B1 实测每位置 6 ms vs 28 ms），只数在飞会把一半
        请求压到慢卡上。还没数据的前段按已知最快的算（先让它接几条、量出来）。"""
        pool = fronts or self.fronts
        k = next(self._rr["__front__"])
        if self.front_pick == "least":
            low = min(f.inflight for f in pool)
            cands = [f for f in pool if f.inflight == low]
            return cands[k % len(cands)]
        known = [f.ewma_ms for f in self.fronts if f.ewma_ms > 0]
        base = min(known) if known else 1.0

        def cost(f):
            return (f.inflight + 1) * (f.ewma_ms if f.ewma_ms > 0 else base)
        low = min(cost(f) for f in pool)
        cands = [f for f in pool if cost(f) <= low * (1 + 1e-9)]
        return cands[k % len(cands)]

    def _candidates(self, task: str | None) -> list[Route]:
        same = [rt for rt in self.routes if task is not None and rt.task == task]
        return same or self.routes

    def _pick(self, r: _Req) -> Route | None:
        """挑路线。KV 守卫开着时只在各段都装得下的路线里挑；一条都装不下 → None（先停靠）。"""
        cands = self._candidates(r.task)
        if self.kv_guard:
            n = self._kv_tokens(r)
            cands = [rt for rt in cands if all(e.kv_fits(n) for e in rt.hops)]
            if not cands:
                return None
        if callable(self.bind):
            backs = list({id(rt.back): rt.back for rt in cands}.values())
            b = self.bind({"rid": r.rid, "task": r.task, "n_prompt": r.n_prompt,
                           "front": r.front.name if r.front else None}, backs)
            mine = [rt for rt in cands if rt.back is b]
            if not mine:
                if self.kv_guard and any(rt.back is b for rt in self.routes):
                    return None             # 以它结尾的路线现在都装不下：先停靠
                mine = [rt for rt in self.routes if rt.back is b]
            if not mine:
                raise RuntimeError(f"bind 返回的 {getattr(b, 'name', b)} 不是任何路线的后段")
            return min(mine, key=lambda rt: rt.inflight / rt.weight)
        k = next(self._rr[r.task])
        if self.bind == "rr":
            return cands[k % len(cands)]

        def load(rt):
            return (rt.inflight + 1) / rt.weight
        low = min(load(rt) for rt in cands)
        least = [rt for rt in cands if load(rt) <= low * (1 + 1e-9)]
        return least[k % len(least)]

    def _identify(self, r: _Req) -> None:
        """前段出口识别：用前段交出的前 L₀ 层统计判 task。识别失败不能让请求卡住 ——
        记下原因，lr 模式下在全部后段里挑最闲的（**不**偷看请求自带的 task：那是用来记对错的）。"""
        t0 = time.perf_counter()
        info = {"rid": r.rid, "task": r.true_task, "n_prompt": r.n_prompt,
                "front": r.front.name if r.front else None, "body": r.body}
        try:
            if r.feat is None:
                raise RuntimeError("前段没有交出识别统计（前段引擎没开 p2p_stats.emit_lr？）")
            if isinstance(r.feat, dict) and "verdict" in r.feat:
                # 前段本机 LR 判好的：直接用，补上协调器这边才知道的 prompt / body 长度
                v = dict(r.feat["verdict"])
                v["lr_at"] = "front"
                v["n_prompt"] = r.n_prompt
                v["n_body"] = sum(int(b) - int(a) for a, b in r.body) if r.body else None
            elif self.classify is None:
                raise RuntimeError("前段交来的是统计、不是判定（前段代理没配 p2p_lr？），协调器也没有 LR")
            else:
                v = dict(self.classify(r.feat, info))
        except Exception as e:                       # noqa: BLE001
            v = {"task": None, "error": f"{type(e).__name__}: {e}"}
        v["lr_ms"] = round((time.perf_counter() - t0) * 1e3, 3)
        r.info["verdict"] = v
        if self.route == "lr":
            r.task = v.get("task")

    async def _bind(self, r: _Req) -> None:
        if self.classify is not None or self.front_lr:
            self._identify(r)
        rt = self._pick(r)
        if rt is None:
            n = self._kv_tokens(r)
            if not any(all(e.kv_ever_fits(n) for e in x.hops) for x in self._candidates(r.task)):
                err = ValueError(f"{r.rid}：prompt {r.n_prompt} + max_tokens {r.max_tokens} 超过了"
                                 f"task {r.task} 每条路线上某一段的 KV 容量")
                if not r.done.done():
                    r.done.set_exception(err)
                r.events.put_nowait(("err", err))
                return
            self._parked.append(r)          # 路线上 KV 不够：前段那条先停着，有请求结束再绑
            return
        await self._bind_to(r, rt)

    async def _bind_to(self, r: _Req, rt: Route) -> None:
        if self.kv_guard:
            self._kv_reserve(r, rt.hops)
        r.route = rt
        r.back = rt.back
        r.hops = [r.front, *rt.hops]
        r.streams = [r.stream] + [RowStream() for _ in rt.mids]
        r.added = [True] + [False] * len(rt.hops)
        rt.inflight += 1
        rt.n_bound += 1
        for e in rt.hops:
            e.inflight += 1
            e.n_bound += 1
        r.t_bound = time.perf_counter()
        await self._add_hop(r, 1)

    async def _add_hop(self, r: _Req, k: int) -> None:
        """第 k−1 段的 prompt hidden 收齐了 → 以 prompt_embeds 送进第 k 段（逐段 prefill）。"""
        e, prev = r.hops[k], r.hops[k - 1]
        last = k == len(r.hops) - 1
        if self.reqs.get(r.rid) is not r:
            return                          # 已经取消 / 结束（cancel 已向各段发了 abort）
        r.added[k] = True
        if k > 1:
            r.t_hop[prev.name] = time.perf_counter()
        if self.direct and last:
            # 先给后段下路由（与 add 同一条 socket，必在 add 之前到）：它出的 token 直接给前段
            await e.client.send_route(r.rid, r.front.client.address, 0)
            if self.reqs.get(r.rid) is not r:
                return                      # 等 send_route 的时候被取消了：abort 已经先发，不能再 add
        embeds = r.streams[k - 1].take()
        if last:
            req = _core_request(
                r.rid, prompt=None, embeds=embeds, max_tokens=r.max_tokens, eos=self.eos,
                logprobs=self.logprobs, ignore_eos=self.ignore_eos, stop_ids=self.stop_ids,
                arrival=time.time())
        else:
            # 中段同前段：从不自己结束（不采样），max_tokens 只是必填项
            req = _core_request(
                r.rid, prompt=None, embeds=embeds, max_tokens=r.max_tokens + 1, eos=None,
                logprobs=0, ignore_eos=True, stop_ids=(), arrival=time.time())
        await e.add(req)
        if self.direct:
            # 上一段：prompt 之后位置的 hidden 直接给这一段（这一段的这些位置至少要等后段出第一个
            # token，而后段要等这一段 prefill 完，所以 add 一定先到；万一没到，上一段的代理先攒着，
            # 这一段的调度器也会先存下 add 之前到的 hidden）
            await prev.client.send_route(r.rid, e.client.address, r.n_prompt)

    # ---- 两个方向的搬运 ----
    def _note_ms(self, r: _Req, e: Engine, k: int, ms: float) -> None:
        if k == 0:
            r.eng_f.append(ms)
        else:
            r.eng_m.setdefault(e.name, []).append(ms)
        e.observe(ms)

    async def _front_loop(self, f: Engine) -> None:
        """前段与中段（都交出 hidden）的输出：prompt 的行收齐 → 绑定路线 / 送进下一段；
        之后的行（不直连时）→ p2p_feed 给下一段。"""
        try:
            while True:
                outs = await f.outputs()
                meta = getattr(f.client, "last_meta", None) or {}
                feeds: dict[str, tuple[Engine, list]] = {}
                now = time.perf_counter()
                for o in outs.outputs:
                    r = self.reqs.get(o.request_id)
                    if r is None or o.p2p_hidden is None:
                        continue
                    k = r.hop_of(f)
                    if k is None or k >= len(r.hops) - 1 and r.route is not None:
                        continue
                    if k == 0:
                        ft = getattr(o, "p2p_feat", None)
                        if ft is not None and r.feat is None:
                            r.feat = ft           # prompt 算完的那一批：前 L₀ 层的识别统计
                    st = r.streams[k]
                    got = st.add(o.p2p_pos, o.p2p_hidden)
                    if got is None:
                        continue          # 被抢占后重算，送来的是旧位置
                    start, rows = got
                    if r.route is None:               # 前段，还没绑定
                        st.keep(rows)
                        if st.next >= r.n_prompt:
                            r.t_front = now
                            await self._bind(r)
                    elif not r.added[k + 1]:          # 下一段还没收下：先攒 prompt 的行
                        st.keep(rows)
                        if st.next >= r.n_prompt:
                            await self._add_hop(r, k + 1)
                    elif self.direct:
                        continue          # 直连：代理已经直接发给下一段了（抢占重算的旧行）
                    else:
                        if k == 0:
                            r.t_hid.append(now)
                        if o.request_id in meta:
                            self._note_ms(r, f, k, meta[o.request_id])
                        nx = r.hops[k + 1]
                        feeds.setdefault(nx.name, (nx, []))[1].append(pack_rows(r.rid, start, rows))
                if self.direct and meta:
                    # 直连转走的那些位置：输出没回来，只回来了「引擎内毫秒」
                    seen = {o.request_id for o in outs.outputs}
                    for rid, ms in meta.items():
                        r = self.reqs.get(rid)
                        if rid in seen or r is None or r.route is None:
                            continue
                        k = r.hop_of(f)
                        if k is not None:
                            self._note_ms(r, f, k, ms)
                for nx, items in feeds.values():
                    await nx.fire("p2p_feed", items)
        except asyncio.CancelledError:
            raise
        except BaseException as e:      # noqa: BLE001 — 引擎死了：让所有在等的请求都知道
            self._fail_all(e)
            raise

    async def _back_loop(self, b: Engine) -> None:
        try:
            while True:
                outs = await b.outputs()
                meta = getattr(b.client, "last_meta", None) or {}
                appends: dict[str, tuple[Engine, list]] = {}
                aborts: dict[str, tuple[Engine, list]] = {}
                released: list[_Req] = []
                now = time.perf_counter()
                for o in outs.outputs:
                    r = self.reqs.get(o.request_id)
                    if r is None or r.back is not b:
                        continue
                    f = r.front
                    if o.new_token_ids:
                        new = [int(t) for t in o.new_token_ids]
                        r.tokens.extend(new)
                        r.lps.extend(_lp_rows(o.new_logprobs))
                        r.t_first = r.t_first or now
                        r.t_last = now
                        r.t_tok.extend([now] * len(new))
                        if len(r.tokens) > 1 and o.request_id in meta:
                            r.eng_b.append(meta[o.request_id])      # 第 1 个 token 是 prefill，不算
                            b.observe(meta[o.request_id])
                        r.events.put_nowait(("tok", new))
                    if o.finish_reason is not None:
                        r.finish = str(o.finish_reason)
                        released.append(r)              # KV 预留在 abort 发出去之后再退
                        for e in r.hops:
                            e.inflight -= 1
                        if r.route is not None:
                            r.route.inflight -= 1
                        for e in r.hops[:-1]:                 # 前段与各中段
                            aborts.setdefault(e.name, (e, []))[1].append(r.rid)
                        res = r.result()
                        res["stop_reason"] = o.stop_reason
                        if not r.done.done():
                            r.done.set_result(res)
                        r.events.put_nowait(("done", res))
                        self.reqs.pop(r.rid, None)
                    elif o.new_token_ids and not self.direct:
                        appends.setdefault(f.name, (f, []))[1].append(
                            [r.rid, [int(t) for t in o.new_token_ids]])
                for f, items in appends.values():
                    await f.fire("p2p_append", items)
                for e, ids in aborts.values():
                    await e.abort(ids)
                for r in released:
                    self._kv_release(r)
        except asyncio.CancelledError:
            raise
        except BaseException as e:      # noqa: BLE001
            self._fail_all(e)
            raise


class MonoRunner:
    """整模型引擎上的同一套请求（对拍与开销的参照）。"""

    def __init__(self, eng: Engine, *, eos=None, stop_ids=(), ignore_eos=False, logprobs=0):
        self.eng, self.eos, self.stop_ids = eng, eos, tuple(stop_ids)
        self.ignore_eos, self.logprobs = ignore_eos, logprobs
        self.reqs: dict[str, _Req] = {}
        self._task: asyncio.Task | None = None
        self._ids = itertools.count()

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._loop())

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def generate(self, prompt, max_tokens, *, task=None, rid=None, timeout=None) -> dict:
        rid = rid or f"m{next(self._ids)}"
        r = self.reqs[rid] = _Req(rid, list(prompt), max_tokens, task, asyncio.get_running_loop())
        await self.eng.add(_core_request(
            rid, prompt=list(prompt), embeds=None, max_tokens=max_tokens, eos=self.eos,
            logprobs=self.logprobs, ignore_eos=self.ignore_eos, stop_ids=self.stop_ids,
            arrival=time.time()))
        try:
            return await asyncio.wait_for(asyncio.shield(r.done), timeout)
        finally:
            self.reqs.pop(rid, None)

    async def _loop(self) -> None:
        try:
            while True:
                outs = await self.eng.outputs()
                now = time.perf_counter()
                for o in outs.outputs:
                    r = self.reqs.get(o.request_id)
                    if r is None:
                        continue
                    if o.new_token_ids:
                        r.tokens.extend(int(t) for t in o.new_token_ids)
                        r.lps.extend(_lp_rows(o.new_logprobs))
                        r.t_first = r.t_first or now
                        r.t_last = now
                    if o.finish_reason is not None:
                        r.finish = str(o.finish_reason)
                        if not r.done.done():
                            r.done.set_result(r.result())
        except asyncio.CancelledError:
            raise
        except BaseException as e:      # noqa: BLE001
            for r in self.reqs.values():
                if not r.done.done():
                    r.done.set_exception(e)
            raise


# --------------------------------------------------------------------------- #
# 对拍
# --------------------------------------------------------------------------- #
def compare_runs(ref: Sequence[dict], got: Sequence[dict], *, tol: float) -> dict:
    """逐条比较（同序）。ref 是整模型，got 是分段。"""
    rows = []
    for a, b in zip(ref, got):
        ta, tb = a["tokens"], b["tokens"]
        cp = 0
        while cp < min(len(ta), len(tb)) and ta[cp] == tb[cp]:
            cp += 1
        dl = [abs(a["logprobs"][i][ta[i]] - b["logprobs"][i][tb[i]])
              for i in range(cp) if i < len(a["logprobs"]) and i < len(b["logprobs"])
              and ta[i] in a["logprobs"][i] and tb[i] in b["logprobs"][i]]
        tie = None
        if cp < min(len(ta), len(tb)) and cp < len(a["logprobs"]):
            top = a["logprobs"][cp]
            tie = (top[ta[cp]] - top[tb[cp]]) if tb[cp] in top else float("inf")
        rows.append({"rid": b["rid"], "back": b.get("back"), "same": ta == tb,
                     "common_prefix": cp, "len": [len(ta), len(tb)],
                     "max_lp_diff": max(dl) if dl else 0.0, "tie_gap": tie})
    n = len(rows)
    ok_lp = all(r["max_lp_diff"] <= tol for r in rows)
    ok_tie = all(r["tie_gap"] is None or r["tie_gap"] <= 2 * tol for r in rows)
    return {"n": n, "same": sum(r["same"] for r in rows),
            "max_lp_diff": max((r["max_lp_diff"] for r in rows), default=0.0),
            "diverged": [r for r in rows if not r["same"]], "ok": ok_lp and ok_tie,
            "ok_lp": ok_lp, "ok_tie": ok_tie, "rows": rows}


def _p(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2)


def summarise(res: Sequence[dict], wall_s: float) -> dict:
    toks = sum(len(r["tokens"]) for r in res)
    fl = [x for r in res for x in r.get("front_leg_ms", ())]
    bl = [x for r in res for x in r.get("back_leg_ms", ())]
    return {
        "n": len(res), "wall_s": round(wall_s, 3), "tok_per_s": round(toks / wall_s, 1),
        "ttft_ms_p50": _p([r["ttft_ms"] for r in res], .5),
        "ttft_ms_p90": _p([r["ttft_ms"] for r in res], .9),
        "tpot_ms_p50": _p([r["tpot_ms"] for r in res], .5),
        "tpot_ms_p90": _p([r["tpot_ms"] for r in res], .9),
        "front_prefill_ms_p50": _p([r.get("front_prefill_ms") for r in res], .5),
        "front_leg_ms_p50": _p(fl, .5), "back_leg_ms_p50": _p(bl, .5),
        "by_back": _count(r.get("back") for r in res),
    }


def _count(xs) -> dict:
    out: dict = defaultdict(int)
    for x in xs:
        out[str(x)] += 1
    return dict(sorted(out.items()))


async def closed_loop(gen, prompts: Sequence[list[int]], *, max_tokens: int, concurrency: int,
                      tasks: Sequence[str | None] | None = None, timeout: float = 300.0,
                      tag: str = "r"):
    """固定并发：同时最多 concurrency 条在飞；结果按 prompts 的顺序返回。
    请求号 `<tag><i>`：每一轮用不同的 tag（同一个号在引擎里结束后再复用虽然合法，但前一轮的
    迟到输出会被认到新请求上）。"""
    sem = asyncio.Semaphore(concurrency)
    out: list[dict | None] = [None] * len(prompts)

    async def one(i):
        async with sem:
            out[i] = await gen(prompts[i], max_tokens,
                               task=tasks[i] if tasks else None, rid=f"{tag}{i}",
                               timeout=timeout)

    t0 = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(len(prompts))))
    return out, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# B0 原型
# --------------------------------------------------------------------------- #
def tiny_config(n_layers: int, dtype: str) -> dict:
    from vllm.p2p.parity import PARITY_CFG

    torch_dtype = {"half": "float16", "float16": "float16", "bfloat16": "bfloat16",
                   "float32": "float32", "float": "float32"}[dtype]
    return dict(PARITY_CFG, num_hidden_layers=n_layers, torch_dtype=torch_dtype)


def back_residents(n_layers: int, split: int, n_experts: int, keep: int, seed: int) -> dict:
    """后段各层一份随机驻留集（前段全装 —— 与 p2p-framework 的部署一致）。"""
    rng = random.Random(seed)
    return {str(l): sorted(rng.sample(range(n_experts), keep)) for l in range(split, n_layers)}


def make_engine(name: str, model: str, *, role: str | None, layers: tuple[int, int] | None,
                layer_experts: dict, args, task: str | None = None) -> Engine:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.p2p.layout import ARCH
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.core_client import EngineCoreClient
    from vllm.v1.executor import Executor

    ov: dict = {"architectures": [ARCH], "p2p_layer_experts": layer_experts}
    if role:
        ov["p2p_stage"] = {"role": role, "layers": list(layers)}
    kw = dict(model=model, dtype=args.dtype, hf_overrides=ov, skip_tokenizer_init=True,
              enable_prompt_embeds=(role in ("mid", "back")), async_scheduling=False,
              enable_prefix_caching=False, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
              enforce_eager=args.eager, seed=0)
    if args.attention_backend:
        kw["attention_backend"] = args.attention_backend
    if args.kv_bytes:
        kw["kv_cache_memory_bytes"] = int(args.kv_bytes)
    cfg = EngineArgs(**kw).create_engine_config(UsageContext.ENGINE_CONTEXT)
    t0 = time.perf_counter()
    client = EngineCoreClient.make_async_mp_client(cfg, Executor.get_class(cfg), log_stats=False)
    nb = cfg.cache_config.num_gpu_blocks
    print(f"  {name:<6} {role or 'mono':<5} 层 {list(layers) if layers else 'all'}  "
          f"起好了（{time.perf_counter() - t0:.1f}s，KV {nb} 块）", flush=True)
    return Engine(name, role, client, task=task, kv_blocks=(nb - 1) if nb else None,
                  block_size=cfg.cache_config.block_size)


def _prompts(n: int, vocab: int, lo: int, hi: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(3, vocab) for _ in range(rng.randint(lo, hi))] for _ in range(n)]


async def _run(args) -> int:
    wd = Path(args.workdir or tempfile.mkdtemp(prefix="p2p_tiered_"))
    model = args.model
    n_layers, n_experts, vocab, eos = args.n_layers, 16, 512, 2
    if model is None:
        from vllm.p2p.parity import _use_p2p_framework

        _use_p2p_framework(args.p2p_framework)
        from p2pmoe.sim.fake_checkpoint import write_fake_checkpoint

        ckpt = wd / f"ckpt_L{n_layers}_{args.dtype}"
        if not (ckpt / "config.json").exists():
            write_fake_checkpoint(ckpt, tiny_config(n_layers, args.dtype), seed=11)
        model = str(ckpt)
    else:
        cfg = json.loads((Path(model) / "config.json").read_text())
        n_layers, n_experts = int(cfg["num_hidden_layers"]), int(cfg["num_experts"])
        vocab, eos = int(cfg["vocab_size"]), cfg.get("eos_token_id")
        eos = eos[0] if isinstance(eos, list) else eos
    split = args.split
    if not 0 < split < n_layers:
        raise SystemExit(f"--split 要在 (0, {n_layers}) 里")
    # 多级：--mid-splits 给的切点把 [split, n_layers) 再切开，前面几段是中段（全专家），最后一段是后段
    cuts = sorted({int(x) for x in str(args.mid_splits or "").split(",") if x.strip()})
    if any(not split < c < n_layers for c in cuts):
        raise SystemExit(f"--mid-splits 的切点要在 ({split}, {n_layers}) 里")
    bounds = [split, *cuts, n_layers]
    back_lo = bounds[-2]
    # 每个后段一份驻留集（默认两条后段装不同的子集；前段 / 中段全装 —— 与 p2p-framework 的部署一致）
    les = [({} if args.full_experts else
            {k: v for k, v in back_residents(n_layers, split, n_experts, args.keep_experts,
                                             seed=5 if args.same_experts else 5 + i).items()
             if int(k) >= back_lo})
           for i in range(args.n_back)]
    distinct = sorted({json.dumps(x, sort_keys=True) for x in les})
    mono_of = [distinct.index(json.dumps(x, sort_keys=True)) for x in les]   # 后段 i → 整模型 j
    mid_desc = "".join(f"、中段 [{a}, {b}) ×{args.mid_copies}" for a, b in zip(bounds, bounds[1:-1]))
    print(f"checkpoint: {model}\n  {n_layers} 层，前段 [0, {split}){mid_desc}、后段 [{back_lo}, {n_layers})；"
          f"{args.n_back} 个后段，每层驻留 "
          f"{'全部' if args.full_experts else args.keep_experts}/{n_experts} 个专家"
          f"（{len(distinct)} 种驻留集）；dtype={args.dtype} "
          f"attention={args.attention_backend or 'auto'}{' eager' if args.eager else ''}")

    prompts = _prompts(args.requests, vocab, args.prompt_min, args.prompt_max, seed=3)
    tol = args.tol if args.tol is not None else (1e-3 if args.dtype in ("float32", "float")
                                                 else 0.02)
    common = dict(eos=eos, ignore_eos=not args.honor_eos, logprobs=5)
    report: dict = {"model": model, "split": split, "n_layers": n_layers, "dtype": args.dtype,
                    "requests": args.requests, "concurrency": args.concurrency,
                    "max_tokens": args.max_tokens, "layer_experts": les, "bind": args.bind}

    print("起引擎（逐个起：每个都要在其余几个已占的显存之外做 profile）…", flush=True)
    engines: list[Engine] = []
    try:
        for j, x in enumerate(distinct):
            engines.append(make_engine(f"mono{j}", model, role=None, layers=None,
                                       layer_experts=json.loads(x), args=args))
        monos = list(engines)
        front = make_engine("front", model, role="front", layers=(0, split), layer_experts={},
                            args=args)
        engines.append(front)
        mid_rows = []                       # 每个中段区间的若干份（并列：路线在这里分叉 / 汇合）
        for k, (a, b) in enumerate(zip(bounds, bounds[1:-1])):
            row = []
            for cpy in range(args.mid_copies):
                row.append(make_engine(f"mid{k}{chr(97 + cpy)}", model, role="mid", layers=(a, b),
                                       layer_experts={}, args=args, task="t0"))
                engines.append(row[-1])
            mid_rows.append(row)
        backs = []
        for i in range(args.n_back):
            backs.append(make_engine(f"back{i}", model, role="back", layers=(back_lo, n_layers),
                                     layer_experts=les[i], args=args, task="t0"))
            engines.append(backs[-1])

        mrs = [MonoRunner(m, **common) for m in monos]
        for m in mrs:
            m.start()
        # 路线：每个后段 × 每个中段区间任取一份（中段区间之间、中段与后段之间都是多对多）
        routes = [Route([*ms, b], task="t0") for b in backs
                  for ms in itertools.product(*mid_rows)]
        co = TieredCoordinator(front, [], routes=routes, bind=args.bind, kv_guard=args.kv_guard,
                               **common)
        co.start()
        # 预热：各一条，排除首轮分配
        for m in mrs:
            await m.generate(prompts[0], 4, rid="warm-m", timeout=args.timeout)
        await co.generate(prompts[0], 4, rid="warm-t", timeout=args.timeout)
        for e in (*backs, *[m for row in mid_rows for m in row]):
            e.n_bound = 0
        for rt in routes:
            rt.n_bound = 0

        phases = sorted({1, args.concurrency})
        refs_c1: list = []
        for c in phases:
            print(f"\n并发 {c}：整模型 ×{len(mrs)} …", flush=True)
            refs, w_ref = [], None
            for j, m in enumerate(mrs):
                ref, w = await closed_loop(m.generate, prompts, max_tokens=args.max_tokens,
                                           concurrency=c, timeout=args.timeout, tag=f"c{c}-m{j}-")
                refs.append(ref)
                w_ref = w if w_ref is None else w_ref       # 时延参照取第一个整模型
            print(f"并发 {c}：分段（1 前段 + {len(backs)} 后段，bind={args.bind}）…", flush=True)
            got, w_got = await closed_loop(co.generate, prompts, max_tokens=args.max_tokens,
                                           concurrency=c, timeout=args.timeout, tag=f"c{c}-t")
            # 每条分段结果对「它被绑到的那个后段」的整模型
            ref_for = [refs[mono_of[int(r["back"].removeprefix("back"))]][i]
                       for i, r in enumerate(got)]
            cmp = compare_runs(ref_for, got, tol=tol)
            # 噪声底：同一个整模型引擎，并发 1 与并发 c 之间的差（批的组成不同 → kernel 的分块 /
            # 累加顺序不同；fp16 下随机小模型的 logits 很平，这个差不小）。并发 > 1 时分段对
            # 整模型的差只要求与噪声底同量级（≤ 3× 底，且不低于 tol），同 parity 的判据。
            if c == 1:
                refs_c1 = refs
                floor = None
            else:
                floors = [compare_runs(a, b, tol=tol) for a, b in zip(refs_c1, refs)]
                floor = {"max_lp_diff": max(f["max_lp_diff"] for f in floors),
                         "same": sum(f["same"] for f in floors),
                         "n": sum(f["n"] for f in floors)}
                lim = max(tol, 3 * floor["max_lp_diff"])
                cmp["ok_lp"] = cmp["max_lp_diff"] <= lim
                # 分叉处的「近似平票」也按噪声底放宽：整模型自己换个批就会在差这么多的地方分叉
                cmp["ok_tie"] = all(r["tie_gap"] is None or r["tie_gap"] <= max(2 * tol, lim)
                                    for r in cmp["rows"])
                cmp["ok"] = cmp["ok_lp"] and cmp["ok_tie"]
                cmp["lp_limit"] = lim
            differ = (sum(a["tokens"] != b["tokens"] for a, b in zip(refs[0], refs[1]))
                      if len(refs) > 1 else None)
            sm, sg = summarise(refs[0], w_ref), summarise(got, w_got)
            report[f"c{c}"] = {"mono": sm, "tiered": sg, "compare":
                               {k: v for k, v in cmp.items() if k != "rows"},
                               "rows": cmp["rows"], "mono_sets_differ": differ,
                               "noise_floor": floor}
            _print_phase(c, sm, sg, cmp, cmp.get("lp_limit", tol))
            if floor is not None:
                print(f"  噪声底（同一整模型，并发 1 vs 并发 {c}）：{floor['same']}/{floor['n']} 条逐 token "
                      f"相同，公共前缀上 max|Δlogprob| = {floor['max_lp_diff']:.2e} → 分段的上限取 "
                      f"max(tol, 3×底) = {cmp['lp_limit']:.2e}")
            if differ is not None:
                print(f"  两种驻留集的整模型输出不同的有 {differ}/{len(prompts)} 条"
                      f"（>0 说明「绑到哪个后段」确实决定了输出，上面的对拍是对着各自的那个比的）")
        ok = all(report[f"c{c}"]["compare"]["ok"] for c in phases)
        bound = {b.name: b.n_bound for b in backs}
        spread = len(backs) == 1 or all(v > 0 for v in bound.values())
        print(f"\n按请求绑定：{bound}  {'✓ 每个后段都分到了' if spread else '✗ 有后段没分到'}")
        if mid_rows:
            rb = {rt.name: rt.n_bound for rt in routes}
            spread = spread and all(v > 0 for v in rb.values())
            print(f"按路线：{rb}  {'✓ 每条路线都走过' if all(rb.values()) else '✗ 有路线没走过'}")
            report["routes"] = rb
        print(f"B0 正确性：{'✓' if ok else '✗'}（tol={tol}）")
        report["bound"], report["ok"] = bound, ok and spread
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
            print(f"明细 → {args.out}")
        await co.close()
        for m in mrs:
            await m.close()
        return 0 if report["ok"] else 1
    finally:
        for e in engines:
            try:
                e.client.shutdown()
            except Exception:   # noqa: BLE001
                pass


def _print_phase(c, sm, sg, cmp, tol) -> None:
    def f(x):
        return "—" if x is None else f"{x:g}"
    print(f"  {'':<8}{'TTFT p50':>10}{'TTFT p90':>10}{'TPOT p50':>10}{'TPOT p90':>10}"
          f"{'tok/s':>9}{'墙钟 s':>9}")
    for name, s in (("整模型", sm), ("分段", sg)):
        print(f"  {name:<8}{f(s['ttft_ms_p50']):>10}{f(s['ttft_ms_p90']):>10}"
              f"{f(s['tpot_ms_p50']):>10}{f(s['tpot_ms_p90']):>10}{f(s['tok_per_s']):>9}"
              f"{f(s['wall_s']):>9}")
    print(f"  分段每个 decode 位置：前段腿 p50 {f(sg['front_leg_ms_p50'])} ms、"
          f"后段腿 p50 {f(sg['back_leg_ms_p50'])} ms；前段 prefill p50 "
          f"{f(sg['front_prefill_ms_p50'])} ms；分到 {sg['by_back']}")
    print(f"  对拍：{cmp['same']}/{cmp['n']} 条逐 token 相同；公共前缀上 max|Δlogprob| = "
          f"{cmp['max_lp_diff']:.2e}（≤ {tol} {'✓' if cmp['ok_lp'] else '✗'}）；"
          f"分叉处都是近似平票 {'✓' if cmp['ok_tie'] else '✗'}")
    for r in cmp["diverged"][:4]:
        gap = r["tie_gap"]
        print(f"    {r['rid']}（{r['back']}）第 {r['common_prefix']} 个 token 分叉，"
              f"整模型自己的两个候选差 {('%.4f' % gap) if gap is not None else '—'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=None, help="现成的 checkpoint；不给 = 生成微型随机 checkpoint")
    ap.add_argument("--p2p-framework", default=None, help="微型 checkpoint 生成器在那边")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--n-layers", type=int, default=8, help="微型 checkpoint 的层数")
    ap.add_argument("--split", type=int, default=4, help="前段层数 L0（前段 [0, L0)）")
    ap.add_argument("--n-back", type=int, default=2, help="同 task 后段的个数")
    ap.add_argument("--mid-splits", default=None,
                    help="多级：再切几刀（逗号分隔，如 6 或 5,6），切出来的前几段是中段（全专家）")
    ap.add_argument("--mid-copies", type=int, default=1,
                    help="多级：每个中段区间起几份（>1 时路线在中段之间分叉 / 汇合）")
    ap.add_argument("--kv-guard", action="store_true", help="协调器按 KV 容量预留（多级建议开）")
    ap.add_argument("--keep-experts", type=int, default=8, help="后段每层驻留几个专家（共 16）")
    ap.add_argument("--full-experts", action="store_true", help="后段也全装")
    ap.add_argument("--same-experts", action="store_true",
                    help="两个后段装同一份子集（默认各装不同的，各对各自的整模型）")
    ap.add_argument("--bind", default="least", choices=("least", "rr"))
    ap.add_argument("--dtype", default="half", choices=("half", "float16", "bfloat16", "float32"))
    ap.add_argument("--attention-backend", default="TRITON_ATTN")
    ap.add_argument("--gpu-mem", type=float, default=0.12,
                    help="每个引擎的显存比例（默认 5 个引擎：2 整模型 + 1 前段 + 2 后段）")
    ap.add_argument("--kv-bytes", type=float, default=None, help="直接给每个引擎的 KV 字节数")
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt-min", type=int, default=8)
    ap.add_argument("--prompt-max", type=int, default=96)
    ap.add_argument("--honor-eos", action="store_true", help="遇到 EOS 就停（默认生成满 max-tokens）")
    ap.add_argument("--tol", type=float, default=None, help="logprob 容差（默认 fp16 0.02、fp32 1e-3）")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    # 所有引擎都在这台机器上（每个 world_size=1）：分布式初始化一律走回环。
    # 不设的话 vLLM 自己挑地址（可能是公网 IPv6、可能是 hostname 解析出的 127.0.1.1），
    # 在有防火墙 / 多网卡的节点上会卡在 TCPStore 连接上（10-13 实测）。已设的不覆盖。
    #
    # 引擎进程用 spawn 起，不用 vLLM 默认的 fork：本进程先用 torch 在 CPU 上生成了微型
    # checkpoint，torch 的 OpenMP 线程池已经起来了；fork 出来的子进程里第一次走到并行的
    # CPU 算子（runner 初始化时 torch.zeros 一个 ~1 MB 的 pinned 缓冲）就永远等不到那些
    # 线程 —— 10-13 实测卡在 CpuGpuBuffer.__init__（py-spy）。
    for k, v in (("VLLM_HOST_IP", "127.0.0.1"), ("GLOO_SOCKET_IFNAME", "lo"),
                 ("NCCL_SOCKET_IFNAME", "lo"), ("VLLM_WORKER_MULTIPROC_METHOD", "spawn")):
        os.environ.setdefault(k, v)
    return asyncio.run(_run(a))


if __name__ == "__main__":
    sys.exit(main())
