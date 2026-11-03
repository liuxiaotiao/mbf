# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""多级分段（前段 → 中段… → 后段）：**真的 vLLM 调度器**（三种角色）+ 真的协调器 / 代理，
GPU 上的那一步换成确定性的假 runner。不需要 GPU。

假模型（因果、逐位置，与 test_tiered.py 同一套前段 / 后段，再加中段）：

    前段    h[p] = [tok[p], p, (tok[0] + … + tok[p]) mod 1000, 0, 0, …]
    中段 j  g[p] = h[p]，但第 2+j 列 = (Σ_{q ≤ p} (j+2)·h[q][0] + h[p][1+j]) mod 1000
    后段    next(p) = (Σ_{q ≤ p} 3·x[q][0] + x[q][2] + 5·x[q][3] + 7·x[q][4]) mod 509 + 3

中段的输出依赖 ≤ p 的全部输入（像注意力），且两个中段的顺序换了结果就不同，所以：
* 走不同路线的请求有不同的「整模型」答案，逐 token 对拍能看出每条请求真的按它的路线走了；
* 中段漏算 / 错位 / 抢占后重算用错输入都会让 token 对不上。
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
W = 256
V = 509
EOS = 11


def front_rows(t: list[int]) -> torch.Tensor:
    out = torch.zeros(len(t), W, dtype=torch.float16)
    acc = 0
    for p, x in enumerate(t):
        acc += x
        out[p, 0], out[p, 1], out[p, 2] = x, p % 1000, acc % 1000
    return out


def mid_rows(j: int, inp: torch.Tensor) -> torch.Tensor:
    """中段 j 对输入行 [0, n) 的输出（逐位置、因果）。"""
    out = inp.clone()
    acc = 0
    for p in range(inp.shape[0]):
        acc += (j + 2) * int(inp[p, 0].item())
        out[p, 2 + j] = (acc + int(inp[p, 1 + j].item())) % 1000
    return out


def next_tok(rows: torch.Tensor) -> int:
    r = rows.double()
    return int((r[:, 0] * 3 + r[:, 2] + 5 * r[:, 3] + 7 * r[:, 4]).sum().item()) % V + 3


def mono(prompt: list[int], max_tokens: int, eos: int | None, mids: list[int]) -> list[int]:
    t, out = list(prompt), []
    for _ in range(max_tokens):
        x = front_rows(t)
        for j in mids:
            x = mid_rows(j, x)
        nt = next_tok(x)
        out.append(nt)
        t.append(nt)
        if nt == eos:
            break
    return out


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def configs(tmp_path_factory):
    os.environ["PYTHONPATH"] = f"{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    from vllm.config import CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
    from vllm.p2p.parity import PARITY_CFG

    d = tmp_path_factory.mktemp("ckpt")
    (d / "config.json").write_text(json.dumps(dict(PARITY_CFG, torch_dtype="float16")))
    out = {}
    for role, layers in (("front", [0, 1]), ("mid", [1, 2]), ("back", [2, 4]), ("mono", None)):
        ov = {"architectures": ["Qwen3MoePartialForCausalLM"]}
        if layers:
            ov["p2p_stage"] = {"role": role, "layers": layers}
        mc = ModelConfig(model=str(d), dtype="float16", seed=0, skip_tokenizer_init=True,
                         max_model_len=256, enable_prompt_embeds=(role in ("mid", "back")),
                         hf_overrides=ov)
        sc = SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=64, max_model_len=256,
                             enable_chunked_prefill=True, is_encoder_decoder=False,
                             async_scheduling=False)
        cc = CacheConfig(block_size=16, gpu_memory_utilization=0.9, cache_dtype="auto",
                         enable_prefix_caching=False)
        out[role] = VllmConfig(model_config=mc, scheduler_config=sc, cache_config=cc,
                               device_config=DeviceConfig("cpu"))
    return out


def make_sched(vc, num_blocks: int):
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.structured_output import StructuredOutputManager

    kv = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(["layer"], FullAttentionSpec(block_size=16, num_kv_heads=1,
                                                      head_size=1, dtype=torch.float32))])
    vc.cache_config.num_gpu_blocks = num_blocks
    return Scheduler(vllm_config=vc, kv_cache_config=kv, log_stats=False,
                     structured_output_manager=StructuredOutputManager(vc), block_size=16)


def _computed(so) -> dict[str, int]:
    c = {nr.req_id: nr.num_computed_tokens for nr in so.scheduled_new_reqs}
    c.update(zip(so.scheduled_cached_reqs.req_ids, so.scheduled_cached_reqs.num_computed_tokens))
    return c


class FrontRunner:
    def __init__(self):
        self.toks: dict[str, list[int]] = {}

    def step(self, so):
        from vllm.v1.outputs import ModelRunnerOutput

        for rid in so.finished_req_ids:
            self.toks.pop(rid, None)
        for nr in so.scheduled_new_reqs:
            self.toks[nr.req_id] = list(nr.prompt_token_ids)
        cr = so.scheduled_cached_reqs
        assert len(cr.new_token_ids) == len(cr.req_ids)
        for i, rid in enumerate(cr.req_ids):
            new, known = cr.new_token_ids[i], self.toks[rid]
            extra = cr.num_computed_tokens[i] + len(new) - len(known)
            if extra > 0:
                known.extend(new[-extra:])
        comp, ids, hid = _computed(so), list(so.num_scheduled_tokens), {}
        for rid in ids:
            n, s0, t = so.num_scheduled_tokens[rid], comp[rid], self.toks[rid]
            assert s0 + n <= len(t)
            hid[rid] = (s0, front_rows(t)[s0:s0 + n])
        return ModelRunnerOutput(req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
                                 p2p_hidden=hid)


class _FedInputs:
    """中段 / 后段共用：prompt_embeds + p2p_feed 进来的行（与 runner 的 RowBuffer 同一套记账）。"""

    def __init__(self):
        from vllm.p2p.stage import RowBuffer

        self.RowBuffer = RowBuffer
        self.inp: dict = {}
        self.n_prompt: dict[str, int] = {}

    def take(self, so):
        for rid in so.finished_req_ids:
            self.inp.pop(rid, None)
            self.n_prompt.pop(rid, None)
        for nr in so.scheduled_new_reqs:
            assert nr.prompt_token_ids is None and nr.prompt_embeds is not None
            b = self.inp[nr.req_id] = self.RowBuffer(W, torch.float16)
            b.put(0, nr.prompt_embeds)
            self.n_prompt[nr.req_id] = nr.prompt_embeds.shape[0]
        for rid, items in (so.p2p_feed or {}).items():
            for pos, rows in items:
                if rid in self.inp:
                    assert pos >= self.n_prompt[rid], "prompt 的行不该走 p2p_feed"
                    self.inp[rid].put(pos, rows)


class MidRunner(_FedInputs):
    def __init__(self, j: int):
        super().__init__()
        self.j = j
        self.placeholders = 0

    def step(self, so):
        from vllm.p2p.stage import PLACEHOLDER_TOKEN
        from vllm.v1.outputs import ModelRunnerOutput

        self.take(so)
        cr = so.scheduled_cached_reqs
        # 中段和前段一样：调度器把（占位）token 随调度结果送下来
        assert len(cr.new_token_ids) == len(cr.req_ids), "中段要随调度结果收到占位 token"
        for toks in cr.new_token_ids:
            assert all(t == PLACEHOLDER_TOKEN for t in toks)
            self.placeholders += len(toks)
        comp, ids, hid = _computed(so), list(so.num_scheduled_tokens), {}
        for rid in ids:
            n, s0 = so.num_scheduled_tokens[rid], comp[rid]
            rows = self.inp[rid].get(0, s0 + n)           # 没送到的位置被排上 → IndexError
            hid[rid] = (s0, mid_rows(self.j, rows)[s0:s0 + n])
        return ModelRunnerOutput(req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
                                 p2p_hidden=hid)


class BackRunner(_FedInputs):
    def __init__(self):
        super().__init__()
        self.n_tok: dict[str, int] = {}

    def step(self, so):
        from vllm.v1.outputs import ModelRunnerOutput

        for rid in so.finished_req_ids:
            self.n_tok.pop(rid, None)
        self.take(so)
        for nr in so.scheduled_new_reqs:
            self.n_tok[nr.req_id] = nr.prompt_embeds.shape[0]
        comp, ids, sampled = _computed(so), list(so.num_scheduled_tokens), []
        for rid in ids:
            n, s0 = so.num_scheduled_tokens[rid], comp[rid]
            rows = self.inp[rid].get(0, s0 + n)
            if s0 + n >= self.n_tok[rid]:
                sampled.append([next_tok(rows)])
                self.n_tok[rid] += 1
            else:
                sampled.append([])
        return ModelRunnerOutput(req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
                                 sampled_token_ids=sampled)


class FakeCore:
    """EngineCore + AsyncMPClient 的替身（同 test_tiered.py）。"""

    def __init__(self, sched, runner, seed: int):
        self.sched, self.runner = sched, runner
        self.q: asyncio.Queue = asyncio.Queue()
        self.wake = asyncio.Event()
        self.rng = random.Random(seed)
        self.task = None
        self.aborted: list[str] = []
        self.preempted = 0

    def _ensure(self):
        if self.task is None:
            self.task = asyncio.get_running_loop().create_task(self._loop())

    async def add_request_async(self, req):
        from vllm.v1.request import Request

        self.sched.add_request(Request.from_engine_core_request(req, None))
        self.wake.set()
        self._ensure()

    async def abort_requests_async(self, ids):
        from vllm.v1.request import RequestStatus

        self.aborted += ids
        self.sched.finish_requests(ids, RequestStatus.FINISHED_ABORTED)

    async def p2p_fire(self, method, *args):
        fut = asyncio.get_running_loop().create_future()
        try:
            fut.set_result(getattr(self.sched, method)(*args))
        except Exception as e:    # noqa: BLE001
            fut.set_exception(e)
        self.wake.set()
        return fut

    async def get_output_async(self):
        self._ensure()
        out = await self.q.get()
        if isinstance(out, BaseException):
            raise out
        return out

    async def _loop(self):
        try:
            while True:
                if not self.sched.has_requests():
                    self.wake.clear()
                    await self.wake.wait()
                so = self.sched.schedule()
                self.preempted += len(so.preempted_req_ids or ())
                outs = self.sched.update_from_output(so, self.runner.step(so))
                for eco in outs.values():
                    if eco.outputs:
                        self.q.put_nowait(eco)
                if so.total_num_scheduled_tokens == 0:
                    self.wake.clear()
                    try:
                        await asyncio.wait_for(self.wake.wait(), 0.05)
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(self.rng.random() * 0.002)
        except BaseException as e:   # noqa: BLE001
            self.q.put_nowait(e)
            raise

    def close(self):
        if self.task:
            self.task.cancel()

    shutdown = close


# 拓扑：两个前段；task t 三条路线（分叉 / 汇合 / 跨过中段），task u 一条
#   t: m1 → back-t      t: m1 → m2 → back-t      t: m2 → back-t2      u: m2 → back-u
SPEC = [("front-A", "front", None, None), ("front-B", "front", None, None),
        ("mid-1", "mid", "t", 1), ("mid-2", "mid", None, 2),
        ("back-t", "back", "t", None), ("back-t2", "back", "t", None), ("back-u", "back", "u", None)]
ROUTES = [("t", ["mid-1", "back-t"], 1.0), ("t", ["mid-1", "mid-2", "back-t"], 2.0),
          ("t", ["mid-2", "back-t2"], 1.0), ("u", ["mid-2", "back-u"], 1.0)]
MIDS_OF = {"mid-1": 1, "mid-2": 2}


def _cores(configs, blocks: dict[str, int], seed0: int = 1):
    """blocks：按引擎名或角色给 KV 块数（名字优先）。"""
    cores = {}
    for k, (name, role, _task, j) in enumerate(SPEC):
        runner = (FrontRunner() if role == "front" else MidRunner(j) if role == "mid"
                  else BackRunner())
        cores[name] = FakeCore(make_sched(configs[role], blocks.get(name, blocks.get(role))),
                               runner, seed0 + k)
    return cores


def _engines(cores, *, guard: bool = False) -> dict:
    from vllm.p2p.tiered import Engine

    out = {}
    for name, role, task, _ in SPEC:
        nb = cores[name].sched.kv_cache_manager.block_pool.get_num_free_blocks()
        out[name] = Engine(name, role, cores[name], task=task,
                           kv_blocks=nb if guard else None, block_size=16)
    return out


def _coordinator(engines: dict, **kw):
    from vllm.p2p.tiered import Route, TieredCoordinator

    routes = [Route([engines[n] for n in hops], task=task, weight=w) for task, hops, w in ROUTES]
    fronts = [engines[n] for n, role, _, _ in SPEC if role == "front"]
    return TieredCoordinator(fronts, [], routes=routes, **kw)


def _prompts(n, seed=0):
    rng = random.Random(seed)
    return [[rng.randrange(3, V) for _ in range(rng.randint(3, 40))] for _ in range(n)]


def _check(prompts, tasks, res, max_tokens, eos):
    for p, task, r in zip(prompts, tasks, res):
        want = mono(p, max_tokens, eos, [MIDS_OF[m] for m in r["mids"]])
        assert r["tokens"] == want, (r["rid"], r["route"])
        assert r["task"] == task
        assert r["route"] in {" → ".join(h) for t, h, _ in ROUTES if t == task}


def test_multihop_matches_monolithic_under_preemption(configs) -> None:
    """mid-1 的 KV 很紧（会被抢占、重算），其余各段宽裕（不会互相等死，见下一条）。"""
    from vllm.p2p.tiered import closed_loop

    n = 24
    prompts = _prompts(n)
    tasks = ["t" if i % 4 else "u" for i in range(n)]

    async def go():
        cores = _cores(configs, {"front": 64, "mid": 64, "back": 64, "mid-1": 5, "front-A": 6})
        engines = _engines(cores)
        co = _coordinator(engines, eos=None, ignore_eos=True)
        co.start()
        try:
            res, _ = await closed_loop(co.generate, prompts, max_tokens=20, concurrency=10,
                                       tasks=tasks, timeout=60)
            await asyncio.sleep(0.05)
            return res, cores, co
        finally:
            await co.close()
            for c in cores.values():
                c.close()

    res, cores, co = asyncio.run(go())
    _check(prompts, tasks, res, 20, None)
    for r in res:
        assert r["finish"] == "length" and len(r["front_leg_ms"]) == 19
    used = {r["route"] for r in res}
    assert used == {" → ".join(h) for _, h, _ in ROUTES}, used        # 四条路线都走过
    assert sum(c.preempted for c in cores.values() if isinstance(c.runner, MidRunner)) > 0, \
        "中段 KV 给得很紧，应该被抢占过（否则没测到中段重算）"
    assert cores["front-A"].preempted > 0
    for c in cores.values():
        assert not c.sched.requests, "各段都收干净了"
        assert not c.sched._p2p_early
    assert all(e.inflight == 0 for e in co.engines) and all(rt.inflight == 0 for rt in co.routes)
    # 中段的序列靠占位 token 长：每条经过它的请求 19 个 decode 位置（抢占重算会再送）
    assert cores["mid-1"].runner.placeholders > 0 and cores["mid-2"].runner.placeholders > 0
    # 结束时前段与路线上各中段都被 abort
    for r in res:
        assert r["rid"] in cores[r["front"]].aborted
        for m in r["mids"]:
            assert r["rid"] in cores[m].aborted


def test_multihop_kv_guard_no_deadlock(configs) -> None:
    """各段 KV 都很紧：不加守卫时多级会互相等死（中段等后段腾地方、后段等中段重算）；
    加了 KV 守卫，按最坏情况预留，谁都不会被抢占，请求在协调器排队 / 停靠。"""
    from vllm.p2p.tiered import closed_loop

    n = 24
    prompts = _prompts(n)
    tasks = ["t" if i % 4 else "u" for i in range(n)]

    async def go():
        cores = _cores(configs, {"front": 10, "mid": 7, "back": 6})
        engines = _engines(cores, guard=True)
        co = _coordinator(engines, eos=None, ignore_eos=True, kv_guard=True)
        co.start()
        peak = {}

        async def watch():
            while True:
                for e in co.engines:
                    assert e.kv_reserved <= e.kv_blocks, (e.name, e.kv_reserved, e.kv_blocks)
                    peak[e.name] = max(peak.get(e.name, 0), e.kv_reserved)
                peak["parked"] = max(peak.get("parked", 0), len(co._parked))
                await asyncio.sleep(0.002)
        w = asyncio.get_running_loop().create_task(watch())
        try:
            res, _ = await closed_loop(co.generate, prompts, max_tokens=20, concurrency=12,
                                       tasks=tasks, timeout=30)
            await asyncio.sleep(0.05)
            return res, cores, co, peak
        finally:
            w.cancel()
            await co.close()
            for c in cores.values():
                c.close()

    res, cores, co, peak = asyncio.run(go())
    _check(prompts, tasks, res, 20, None)
    assert not any(c.preempted for c in cores.values()), "守卫按最坏情况预留，不该有抢占"
    assert all(e.kv_reserved == 0 for e in co.engines) and not co._parked
    assert peak["parked"] > 0, "后段只有 5 块可用，应该有请求在协调器停靠过"
    assert any(r.get("admit_wait_ms", 0) > 1 for r in res), "前段装不下的请求在协调器排过队"
    for c in cores.values():
        assert not c.sched.requests


def test_kv_guard_rejects_oversized(configs) -> None:
    async def go():
        cores = _cores(configs, {"front": 64, "mid": 64, "back": 64, "back-u": 3})
        co = _coordinator(_engines(cores, guard=True), eos=None, ignore_eos=True, kv_guard=True)
        co.start()
        try:
            with pytest.raises(ValueError, match="前段的 KV 容量"):
                await co.generate([5] * 2000, 8, task="t", timeout=5)
            with pytest.raises(ValueError, match="每条路线"):
                await co.generate([5] * 40, 20, task="u", timeout=5)   # back-u 只有 2 块
            r = await co.generate([5] * 40, 20, task="t", timeout=5)
            return r, cores, co
        finally:
            await co.close()
            for c in cores.values():
                c.close()

    r, cores, co = asyncio.run(go())
    assert len(r["tokens"]) == 20
    assert all(e.kv_reserved == 0 for e in co.engines)


def test_multihop_eos_and_cancel(configs) -> None:
    prompts = _prompts(10, seed=3)

    async def go():
        cores = _cores(configs, {"front": 64, "mid": 64, "back": 64})
        engines = _engines(cores)
        co = _coordinator(engines, eos=EOS, ignore_eos=False)
        co.start()
        try:
            res = await asyncio.gather(*(co.generate(p, 40, task="t", rid=f"e{i}", timeout=30)
                                         for i, p in enumerate(prompts)))
            # 中途取消：拿到 3 个 token 就走
            got = []
            async for ev in co.stream(prompts[0], 40, task="t", rid="cx"):
                if ev[0] == "tok":
                    got += ev[1]
                    if len(got) >= 3:
                        break
            await asyncio.sleep(0.1)
            return res, got, cores, co
        finally:
            await co.close()
            for c in cores.values():
                c.close()

    res, got, cores, co = asyncio.run(go())
    _check(prompts, ["t"] * len(prompts), res, 40, EOS)
    for r in res:
        assert r["finish"] == ("stop" if r["tokens"][-1] == EOS and len(r["tokens"]) < 40
                               else "length")
    assert len(got) >= 3
    for c in cores.values():
        assert not c.sched.requests
    assert all(e.inflight == 0 for e in co.engines) and all(rt.inflight == 0 for rt in co.routes)


def test_mid_scheduler_placeholders_early_feed_and_gone(configs) -> None:
    """单看中段调度器：feed 让序列变长（占位 token）、add 之前先到的 feed 存着、结束后迟到的丢掉。"""
    from vllm.p2p.stage import PLACEHOLDER_TOKEN, pack_rows
    from vllm.p2p.tiered import _core_request
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus

    def add(s, rid, n):
        req = _core_request(rid, prompt=None, embeds=torch.zeros(n, W, dtype=torch.float16),
                            max_tokens=8, eos=None, logprobs=0, ignore_eos=True, stop_ids=(),
                            arrival=0.0)
        s.add_request(Request.from_engine_core_request(req, None))

    def emit(s, so):
        ids = list(so.num_scheduled_tokens)
        return s.update_from_output(so, ModelRunnerOutput(
            req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
            p2p_hidden={r: (0, torch.zeros(so.num_scheduled_tokens[r], W)) for r in ids}))

    one = lambda: torch.ones(1, W, dtype=torch.float16)            # noqa: E731
    s = make_sched(configs["mid"], 32)
    assert s.p2p_role == "mid"
    add(s, "a", 5)
    so = s.schedule()
    assert so.num_scheduled_tokens == {"a": 5}
    outs = emit(s, so)
    (o,) = outs[0].outputs
    assert o.p2p_pos == 0 and o.p2p_hidden.shape[0] == 5 and not o.new_token_ids
    assert s.schedule().num_scheduled_tokens == {}             # 没 feed：不排
    assert s.p2p_feed([pack_rows("a", 5, one())]) == 1
    req = s.requests["a"]
    assert req.num_tokens == 6 and list(req.output_token_ids) == [PLACEHOLDER_TOKEN]
    so = s.schedule()
    assert so.num_scheduled_tokens == {"a": 1}
    # 调度器把占位 token 随调度结果送给 runner（runner 不采样，不知道序列长了）
    assert so.scheduled_cached_reqs.new_token_ids == [[PLACEHOLDER_TOKEN]]
    emit(s, so)
    # 重复送（上一段抢占后重算）不让序列多长
    s.p2p_feed([pack_rows("a", 5, one())])
    assert req.num_tokens == 6
    with pytest.raises(RuntimeError, match="前段"):
        s.p2p_append([["a", [1]]])

    # add 之前先到的 hidden：存着，add 时补上
    s.p2p_feed([pack_rows("b", 3, one())])
    assert "b" in s._p2p_early
    add(s, "b", 3)
    assert "b" not in s._p2p_early and s.requests["b"].p2p_ready == 4
    assert s.requests["b"].num_tokens == 4

    # add 之前先到、add 之后又没排上的那几行：留到它第一次被排上的那一步再送（runner 才认识它）
    s.max_num_running_reqs = 2                 # a、b 在跑，d 排不上
    s.p2p_feed([pack_rows("d", 2, one())])
    add(s, "d", 2)
    so = s.schedule()
    assert "d" not in so.num_scheduled_tokens and "d" not in (so.p2p_feed or {})
    assert "d" in s._p2p_pending
    s.max_num_running_reqs = 16
    so = s.schedule()
    assert so.num_scheduled_tokens.get("d") and [p for p, _ in so.p2p_feed["d"]] == [2]
    # 结束了的请求：迟到的 hidden 直接丢，不进 early
    s.finish_requests(["a"], RequestStatus.FINISHED_ABORTED)
    s.schedule()
    assert s.p2p_feed([pack_rows("a", 6, one())]) == 0 and "a" not in s._p2p_early
    # abort 一条还没 add 的：先到的丢掉，之后的也不收
    s.p2p_feed([pack_rows("c", 4, one())])
    s.finish_requests(["c"], RequestStatus.FINISHED_ABORTED)
    s.p2p_feed([pack_rows("c", 5, one())])
    assert "c" not in s._p2p_early


def test_route_validation() -> None:
    from vllm.p2p.tiered import Engine, Route, TieredCoordinator

    f = Engine("f", "front", object())
    m = Engine("m", "mid", object(), task="t")
    b = Engine("b", "back", object(), task="t")
    with pytest.raises(ValueError, match="出现了两次"):
        Route([m, m, b])
    with pytest.raises(ValueError, match="只能是中段"):
        Route([b, Engine("b2", "back", object())])
    with pytest.raises(ValueError, match="要是后段"):
        Route([m])
    with pytest.raises(ValueError, match="weight"):
        Route([b], weight=0)
    rt = Route([m, b])
    assert rt.task == "t" and rt.name == "m → b" and rt.back is b and rt.mids == [m]
    co = TieredCoordinator(f, [], routes=[rt, Route([b])])
    assert co.mids == [m] and co.backs == [b]
    with pytest.raises(ValueError, match="既当中段又当后段"):
        TieredCoordinator(f, [], routes=[Route([m, b]), Route([Engine("m", "back", object())])])


def test_pick_follows_route_weights() -> None:
    """least：同 task 的路线按 (在飞 + 1) / weight 分 —— 在飞数正比于 weight。"""
    from vllm.p2p.tiered import Engine, Route, TieredCoordinator, _Req

    f = Engine("f", "front", object())
    b1, b2 = Engine("b1", "back", object(), task="t"), Engine("b2", "back", object(), task="t")
    co = TieredCoordinator(f, [], routes=[Route([b1], weight=1.0), Route([b2], weight=3.0)])
    loop = asyncio.new_event_loop()
    try:
        for i in range(40):
            r = _Req(f"r{i}", [1], 4, "t", loop)
            co._pick(r).inflight += 1
    finally:
        loop.close()
    assert [rt.inflight for rt in co.routes] == [10, 30]


# --------------------------------------------------------------------------- #
# 每个引擎隔着真的 ZMQ（remote.AgentBridge ↔ RemoteClient）；直连时中段代理把 hidden 直接
# 转给路线上的下一段
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("direct", [False, True], ids=["via-coordinator", "direct"])
def test_multihop_remote_agents(configs, direct) -> None:
    pytest.importorskip("zmq")
    from vllm.p2p.remote import AgentBridge, RemoteClient
    from vllm.p2p.tiered import Engine, closed_loop

    prompts = _prompts(16, seed=5)
    tasks = ["t" if i % 3 else "u" for i in range(len(prompts))]

    async def go():
        cores = _cores(configs, {"front": 64, "mid": 64, "back": 64}, seed0=40)
        port0 = 27000 + random.Random().randrange(2000)
        bridges, runs, clients, engines = [], [], [], {}
        for k, (name, role, task, _) in enumerate(SPEC):
            cores[name]._ensure()
            br = AgentBridge(cores[name], f"tcp://127.0.0.1:{port0 + k}",
                             {"name": name, "role": role, "task": task, "layers": [0, 1]})
            bridges.append(br)
            runs.append(asyncio.get_running_loop().create_task(br.run()))
        for k, (name, role, task, _) in enumerate(SPEC):
            cl = RemoteClient(f"tcp://127.0.0.1:{port0 + k}")
            info = await cl.connect(timeout=10, direct=direct)
            assert info["role"] == role
            clients.append(cl)
            engines[name] = Engine(name, role, cl, task=task)
        co = _coordinator(engines, eos=EOS, ignore_eos=False, direct=direct)
        co.start()
        try:
            res, _ = await closed_loop(co.generate, prompts, max_tokens=14, concurrency=8,
                                       tasks=tasks, timeout=30)
            await asyncio.sleep(0.2)
            n_direct = {br.info["name"]: br.n_direct for br in bridges}
            assert not any(br.routes or br.held for br in bridges)
            return res, n_direct, cores
        finally:
            await co.close()
            for t in runs:
                t.cancel()
            for c in cores.values():
                c.close()
            for cl in clients:
                cl.shutdown()

    res, n_direct, cores = asyncio.run(go())
    _check(prompts, tasks, res, 14, EOS)
    for c in cores.values():
        assert not c.sched.requests
    if direct:
        # 前段 → 第一段、中段 → 下一段、后段 → 前段 都直接互发了
        assert all(n_direct[n] > 0 for n in ("mid-1", "mid-2", "back-t", "back-u")), n_direct
        assert n_direct["front-A"] + n_direct["front-B"] > 0
    else:
        assert not any(n_direct.values())
    for r in res:
        if len(r["tokens"]) > 2 and not direct:
            assert all(r["mid_engine_ms"].get(m) for m in r["mids"]), r


def test_gateway_reports_mids(configs) -> None:
    """分段网关（tiered_gateway.make_app）在多级下：路由记录 / 响应头 / /p2p/channels 带中段与路线。"""
    pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer

    from vllm.p2p.tiered_gateway import make_app

    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(97 + i % 26) for i in ids)

    prompts = _prompts(6, seed=9)

    async def go():
        cores = _cores(configs, {"front": 64, "mid": 64, "back": 64})
        co = _coordinator(_engines(cores, guard=True), eos=None, ignore_eos=True, kv_guard=True)
        co.start()
        stages = {n: {"nodes": [f"N-{n}"]} for n, *_ in SPEC}
        client = TestClient(TestServer(make_app(co, stages=stages, tokenizer=Tok())))
        await client.start_server()
        try:
            outs = []
            for i, p in enumerate(prompts):
                r = await client.post("/v1/completions", json={
                    "prompt": p, "max_tokens": 6, "temperature": 0.0, "p2p_task": "t",
                    "p2p_id": f"g{i}"})
                assert r.status == 200, await r.text()
                outs.append((dict(r.headers), await r.json()))
            nob = await client.post("/v1/completions", json={
                "prompt": prompts[0], "max_tokens": 4, "temperature": 0.0, "p2p_task": "zzz"})
            assert nob.status == 503
            recs = (await (await client.get("/p2p/records")).json())["records"]
            chans = await (await client.get("/p2p/channels")).json()
            return outs, recs, chans
        finally:
            await client.close()
            await co.close()
            for c in cores.values():
                c.close()

    outs, recs, chans = asyncio.run(go())
    for (hdr, body), p in zip(outs, prompts):
        met = body["p2p_metrics"]
        want = mono(p, 6, None, [MIDS_OF[m] for m in met["mids"]])
        assert body["choices"][0]["text"] == Tok().decode(want)
        assert hdr.get("X-P2P-Mids", "") == ",".join(met["mids"])
        assert met["route"] and met["admit_wait_ms"] is not None
    by = {r["id"]: r for r in recs}
    withmid = [r for r in by.values() if r.get("mids")]
    assert withmid and all(r["mid_nodes"] == [f"N-{m}" for m in r["mids"]] for r in withmid)
    assert {m["name"] for m in chans["mids"]} == {"mid-1", "mid-2"}
    assert len(chans["routes"]) == 4 and "kv" in chans


class MonoRunner:
    """整模型（前段 → 中段 1 → 后段 的合成）的假 runner：自己缓存采样出的 token。"""

    def __init__(self, mids: list[int]):
        self.mids = mids
        self.toks: dict[str, list[int]] = {}

    def step(self, so):
        from vllm.v1.outputs import ModelRunnerOutput

        for rid in so.finished_req_ids:
            self.toks.pop(rid, None)
        for nr in so.scheduled_new_reqs:
            self.toks[nr.req_id] = list(nr.prompt_token_ids)
        comp, ids, sampled = _computed(so), list(so.num_scheduled_tokens), []
        for rid in ids:
            n, s0, t = so.num_scheduled_tokens[rid], comp[rid], self.toks[rid]
            if s0 + n >= len(t):
                x = front_rows(t)
                for j in self.mids:
                    x = mid_rows(j, x)
                nt = next_tok(x)
                sampled.append([nt])
                t.append(nt)
            else:
                sampled.append([])
        return ModelRunnerOutput(req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
                                 sampled_token_ids=sampled)


def test_b0_script_with_mid_stages(configs, tmp_path, monkeypatch, capsys) -> None:
    """`python -m vllm.p2p.tiered --mid-splits 2 --mid-copies 2 --kv-guard`：前段 [0,1) →
    中段 [1,2) ×2 → 后段 [2,4) ×2，4 条路线（中段与后段之间多对多），逐条对整模型。"""
    from vllm.p2p import tiered

    made = []

    def fake_make_engine(name, model, *, role, layers, layer_experts, args, task=None):
        runner = {"front": FrontRunner, "back": BackRunner}.get(role)
        runner = runner() if runner else (MidRunner(1) if role == "mid" else MonoRunner([1]))
        made.append((name, role, layers, json.dumps(layer_experts, sort_keys=True)))
        core = FakeCore(make_sched(configs[role or "mono"], 64), runner, len(made))
        return tiered.Engine(name, role, core, task=task, kv_blocks=63, block_size=16)

    monkeypatch.setattr(tiered, "make_engine", fake_make_engine)
    ckpt = Path(configs["mono"].model_config.model)
    out = tmp_path / "b0.json"
    rc = tiered.main(["--model", str(ckpt), "--split", "1", "--mid-splits", "2", "--mid-copies", "2",
                      "--kv-guard", "--requests", "12", "--concurrency", "4", "--max-tokens", "10",
                      "--prompt-max", "30", "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0, text
    rep = json.loads(out.read_text())
    assert rep["ok"] and rep["c4"]["compare"]["same"] == 12
    assert len(rep["routes"]) == 4 and min(rep["routes"].values()) > 0
    names = [m[0] for m in made]
    assert names == ["mono0", "mono1", "front", "mid0a", "mid0b", "back0", "back1"]
    assert [m[2] for m in made if m[1] == "mid"] == [(1, 2), (1, 2)]
    assert [m[2] for m in made if m[1] == "back"] == [(2, 4), (2, 4)]
    # 后段的驻留集只列它自己的层（≥ 2）；中段全专家
    for nm, role, _, le in made:
        if role == "back":
            assert all(int(k) >= 2 for k in json.loads(le))
        if role == "mid":
            assert json.loads(le) == {}
    assert "按路线" in text and "B0 正确性：✓" in text
