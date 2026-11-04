# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 · 闭环压测（与 p2p-framework `deploy_15.sh continuous` 同一口径）。**按文件路径运行。**

    # 控制机上，vllm-p2p 的 venv；网关已起（gateway.py）
    python vllm/p2p/bench.py --gateway http://127.0.0.1:9000 \\
        --cases /home/ubuntu/p2p-framework/TestSamples/cases.jsonl \\
        --model-dir /home/ubuntu/p2p-framework/weights-q3-30b \\
        --max-tokens 2048 --inflight 5 --warmup 1 --smi --out results/vllm_e3.json

    # 结果与 p2p-framework 的 results/*.json 同构，可以直接
    cd /home/ubuntu/p2p-framework && ./deploy_15.sh report /home/ubuntu/vllm-p2p/results/vllm_e3.json

与本框架**同一套**测试与指标 —— 能复用的全部 import p2p-framework 的原函数，不重写：

* 测试集读取 `p2pmoe.deploy.control.load_cases`、请求池顺序 `continuous.interleave`、
  闭环驱动 `continuous.run_closed_loop`（在途恒为 N，一条完成补一条）、汇总 `continuous.run_metrics`；
* prompt 用 `TextIO.encode_body` 构造（与选择器训练时、与本框架开 LR 时同一口径，
  `enable_thinking=False`），**以 token id 发给 vLLM**（`/v1/completions`），所以两边模型看到的
  token 逐个相同；body 区间一起发给网关（LR / A5 只数 body）；
* 贪心（temperature=0），`max_tokens` 同本框架，停止条件是模型自己的 EOS（generation_config）。

逐请求记录的字段与本框架对齐（`total_ms`、`prefill_ms`=TTFT、`n_generated` …）。几处口径：

| 字段 | 本框架 | 这里 |
|---|---|---|
| `queue_ms` | 等空闲前段 | 网关里的路由耗时（lr 模式 = 识别排队 + 识别计算；oracle ≈ 0） |
| `queue_back_ms` | 等空闲后段 | vLLM 引擎里的排队（`p2p_metrics.queued_ms`，引擎自己量的） |
| `compute_ms` / 逐节点占比 | 节点埋点 | 拿不到（vLLM 不按节点报计算时长）→ 0，`traces_complete=False` |

`--smi`：压测期间在每台节点后台跑 `nvidia-smi -lms`，结束后收回来算 GPU 利用率 / 显存 / 功耗。
`--stats`：通道开了 A5 钩子（`layout.py --stats`）时，收回各节点的逐请求统计，拼上 miss 率与
前 L₀ 层的 LR 特征；给了 `--lr-model` 还会用钩子特征重算一遍识别，与识别服务的结果逐条对。
"""

from __future__ import annotations

import argparse
import http.client
import json
import logging
import re
import shlex
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

try:                                        # 按文件路径运行：同目录直接 import
    from tiered_stats import (continuous_report, cumulative_stats, mark_stats, request_stats,
                              saturate_report, segment_stats, summary_csv, summary_table)
except ImportError:
    from vllm.p2p.tiered_stats import (continuous_report, cumulative_stats, mark_stats,
                                       request_stats, saturate_report, segment_stats,
                                       summary_csv, summary_table)
try:
    import launch as _launch
except ImportError:                         # pragma: no cover  python -m vllm.p2p.bench
    from vllm.p2p import launch as _launch

log = logging.getLogger("p2p.bench")

__all__ = ["Rec", "build_row", "stop_reason_of", "parse_sse_line", "summarise_smi",
           "join_stats", "main"]


def use_p2p_framework(path: str | None) -> Path:
    try:
        from identifier import use_p2p_framework as f       # 同目录
    except ImportError:                                      # pragma: no cover
        from vllm.p2p.identifier import use_p2p_framework as f
    return f(path)


# --------------------------------------------------------------------------- #
# 一条请求
# --------------------------------------------------------------------------- #
@dataclass
class Rec:
    i: int
    rid: str
    true_task: str | None
    payload: object
    ids: list[int]
    body: list[tuple[int, int]] | None
    case_id: str | None = None
    prompt: str = ""
    t0: float = 0.0
    t0_wall: float = 0.0
    done: threading.Event = field(default_factory=threading.Event)
    t_first: float | None = None
    t_last: float | None = None
    chunk_t: list[float] = field(default_factory=list)
    text: str = ""
    finish_reason: str | None = None
    stop_reason: object = None
    usage: dict | None = None
    engine: dict | None = None
    headers: dict = field(default_factory=dict)
    status: int | None = None
    error: str | None = None
    marks_want: tuple = ()
    """生成到这些 token 数时打点（--marks）。"""
    mark_t: dict = field(default_factory=dict)
    """{k: (时刻, 当时实际已生成的 token 数)} —— 一块里可能不止一个 token，实际数 ≥ k。"""


def parse_sse_line(line: bytes) -> dict | None:
    """`data: {...}` → dict；`data: [DONE]`、空行、注释 → None。"""
    s = line.strip()
    if not s.startswith(b"data:"):
        return None
    s = s[5:].strip()
    if not s or s == b"[DONE]":
        return None
    return json.loads(s)


def stop_reason_of(finish_reason: str | None, stop_reason) -> str:
    """vLLM 的 finish_reason → 本框架的 stop_reason（eos / stop_string / max_tokens）。"""
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "stop":
        return "stop_string" if isinstance(stop_reason, str) else "eos"
    return finish_reason or ""


def post_stream(base: str, payload: dict, rec: Rec, timeout: float) -> None:
    """流式 POST /v1/completions，逐块记时间。在线程里跑。"""
    u = urllib.parse.urlsplit(base)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    try:
        conn.request("POST", "/v1/completions", body=json.dumps(payload),
                     headers={"Content-Type": "application/json", "X-Request-Id": rec.rid})
        resp = conn.getresponse()
        rec.status = resp.status
        rec.headers = {k: v for k, v in resp.getheaders() if k.lower().startswith("x-p2p")}
        if resp.status != 200:
            rec.error = f"HTTP {resp.status}: {resp.read()[:500].decode('utf-8', 'replace')}"
            return
        for line in resp:
            d = parse_sse_line(line)
            if d is None:
                continue
            now = time.perf_counter()
            if d.get("error"):
                rec.error = json.dumps(d["error"], ensure_ascii=False)[:500]
                break
            ch = d.get("choices") or []
            if ch:
                c = ch[0]
                if rec.t_first is None:
                    rec.t_first = now
                rec.chunk_t.append(now)
                rec.t_last = now
                rec.text += c.get("text") or ""
                if c.get("finish_reason"):
                    rec.finish_reason = c["finish_reason"]
                    rec.stop_reason = c.get("stop_reason")
            if d.get("usage"):
                rec.usage = d["usage"]
                # 每块都带累计的 completion_tokens（continuous_usage_stats）：第一次 ≥ k 时打点
                n = int(d["usage"].get("completion_tokens") or 0)
                for k in rec.marks_want:
                    if n >= k and k not in rec.mark_t:
                        rec.mark_t[k] = (now, n)
            if d.get("p2p_metrics"):
                rec.engine = d["p2p_metrics"]
    except Exception as e:                                    # noqa: BLE001
        rec.error = f"{type(e).__name__}: {e}"
    finally:
        conn.close()
        if rec.t_last is None:
            rec.t_last = time.perf_counter()
        rec.done.set()


def http_json(url: str, data: dict | None = None, timeout: float = 60) -> dict:
    req = urllib.request.Request(
        url, data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json"}, method="GET" if data is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def http_text(url: str, timeout: float = 10) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


# --------------------------------------------------------------------------- #
# 结果行（与 p2p-framework 的 results/*.json 同构）
# --------------------------------------------------------------------------- #
def _median(xs):
    xs = sorted(xs)
    if not xs:
        return 0.0
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])


def mark_rows(rec: Rec) -> dict:
    """打点 → {k: {n, t_ms（提交→第 k 个 token，含排队）, since_first_ms（首 token→第 k 个）,
    tpot_ms（到这里为止的平均每 token）, tok_per_s（k / 含排队用时）, decode_tok_per_s}}。"""
    out = {}
    for k in sorted(rec.mark_t):
        t, n = rec.mark_t[k]
        e = (t - rec.t0) * 1e3
        sf = (t - rec.t_first) * 1e3 if rec.t_first is not None else None
        out[str(k)] = {
            "n": n, "t_ms": round(e, 2),
            "since_first_ms": round(sf, 2) if sf is not None else None,
            "tpot_ms": round(sf / (n - 1), 3) if sf and n > 1 else None,
            "tok_per_s": round(n / (e / 1e3), 3) if e > 0 else None,
            "decode_tok_per_s": round((n - 1) / (sf / 1e3), 3) if sf and n > 1 else None,
        }
    return out


# --------------------------------------------------------------------------- #
# 满载连续测试（--saturate）：逐步加压到满载，之后一条完成补一条、始终保持满载
# --------------------------------------------------------------------------- #
_SYNTH_WORDS = ("system data model value layer token graph network node memory route stream batch cache "
                "result number function matrix vector signal energy market river city garden window paper "
                "music story travel winter summer planet ocean forest mountain question answer reason "
                "method table chair light water stone bridge train school doctor friend morning").split()
_SYNTH_HEAD = {"gsm8k": "Solve the following math word problem step by step.",
               "mbpp": "Write a Python function for the following task.",
               "no_robots": "Write a helpful, natural response to the following request."}


def synthetic_cases(tasks: Sequence[str], per_task: int, prompt_len: int,
                    seed: int = 0) -> list[tuple[str, str]]:
    """合成请求 → [(task, prompt)]：每个 task per_task 条、各条内容不同（不会命中前缀缓存），
    约 prompt_len 个 token（常用英文单词，带前导空格时一般一个词一个 token）。
    只用来测满载吞吐 / TPOT：内容与真实任务无关；配合网关 --ignore-eos 让每条都生成满 max_tokens。"""
    import random
    rng = random.Random(seed)
    out = []
    for u in tasks:
        head = _SYNTH_HEAD.get(u, "Respond to the following text.")
        for j in range(per_task):
            body = " ".join(rng.choice(_SYNTH_WORDS) for _ in range(max(1, prompt_len - 16)))
            out.append((u, f"{head} [{u}-{j}] {body}"))
    return out


def saturation_caps(live, channels: Mapping[str, Mapping], *, by_task: bool,
                    per_back: int = 0) -> dict:
    """满载的在途上限。按 task（by_task）：{task: (Σ 该 task 各后段的槽位, 后段数)}；
    全局：{None: (Σ 所有后段的槽位, 后段数)}。一个后段的槽位 = --per-back，没给就用它的 max_num_seqs。"""
    caps: dict = {}
    for c in sorted(live):
        ch = channels.get(c) or {}
        slots = int(per_back or ch.get("max_num_seqs") or 0)
        if slots <= 0:
            raise SystemExit(f"后段 {c} 没写 max_num_seqs（vLLM 默认 256，不能当满载）—— 用 --per-back 指定"
                             "每个后段同时跑几条")
        k = ch.get("task") if by_task else None
        cap, nb = caps.get(k, (0, 0))
        caps[k] = (cap + slots, nb + 1)
    return caps


def capacity_mix(pool: Sequence, n: int, caps: Mapping) -> list[int]:
    """按各 task 的满载槽位分配请求数（槽位多的 task 多给），每个 task 在自己的样本里循环取，
    task 之间轮流排开。返回 pool 的下标序列（长 n）。这样各 task 的后段差不多同时做完，满载段最长。"""
    by: dict = defaultdict(list)
    for i, (u, _) in enumerate(pool):
        by[u].append(i)
    keys = [k for k in caps if by.get(k)]
    tot = sum(caps[k][0] for k in keys)
    want = {k: n * caps[k][0] / tot for k in keys}
    cnt = {k: int(want[k]) for k in keys}
    for k in sorted(keys, key=lambda k: want[k] - cnt[k], reverse=True)[: n - sum(cnt.values())]:
        cnt[k] += 1
    taken = {k: 0 for k in keys}
    out = []
    for _ in range(sum(cnt.values())):        # 按比例交错：每次取「进度」最落后的 task
        k = min((k for k in keys if taken[k] < cnt[k]), key=lambda k: ((taken[k] + 1) / cnt[k], keys.index(k)))
        out.append(by[k][taken[k] % len(by[k])])
        taken[k] += 1
    return out


def ramp_level(t: float, *, step: int, every: float) -> int:
    """加压进度：第 t 秒时每个后段允许的在途条数（从 1 起，每 every 秒加 step）。"""
    if every <= 0:
        return 1 << 30
    return 1 + max(0, int(step)) * int(max(0.0, t) // every)


def run_ramp_loop(order: Sequence[int], key_of, caps: Mapping, submit, *, step: int, every: float,
                  timeout_s: float, on_done, poll_s: float = 0.02, sample_s: float = 1.0,
                  on_level=None):
    """逐步加压的闭环：key（task 或全局）k 的在途上限 = min(槽位_k, 后段数_k × ramp_level(t))；
    到了上限以后一条完成补一条，直到这个 key 的请求发完。

    返回 (完成, 超时, 墙钟起点, 墙钟终点, info)。info 里 t_full_s = 所有 key 第一次同时坐满的时刻，
    t_end_s = 第一个 key 发完最后一条（之后它的后段开始空出来）的时刻 —— 两者之间就是满载段；
    curve = 每 sample_s 秒的 (t, 总在途, 总上限)。"""
    queues: dict = {k: [] for k in caps}
    for i in order:
        queues.setdefault(key_of(i), []).append(i)
    for k in list(queues):
        if k not in caps or not queues[k]:
            if queues[k] and k not in caps:
                raise ValueError(f"请求的 key {k!r} 没有后段槽位")
            queues.pop(k)
    cap = {k: caps[k][0] for k in queues}
    nb = {k: caps[k][1] for k in queues}
    head = {k: 0 for k in queues}
    live: list = []
    lkey: dict = {}
    done, dead, curve = [], [], []
    t_full = t_end = None
    n = len(order)
    last_level, next_sample = None, 0.0
    t_start = time.perf_counter()
    while any(head[k] < len(queues[k]) for k in queues) or live:
        now = time.perf_counter() - t_start
        lv = ramp_level(now, step=step, every=every)
        limit = {k: min(cap[k], nb[k] * lv) for k in queues}
        if lv != last_level and on_level and last_level is not None and \
                any(min(cap[k], nb[k] * last_level) < cap[k] for k in queues):
            on_level(now, lv, limit)
        last_level = lv
        cnt = defaultdict(int)
        for r in live:
            cnt[lkey[id(r)]] += 1
        for k in queues:
            while head[k] < len(queues[k]) and cnt[k] < limit[k]:
                r = submit(queues[k][head[k]])
                head[k] += 1
                live.append(r)
                lkey[id(r)] = k
                cnt[k] += 1
            if t_end is None and t_full is not None and head[k] >= len(queues[k]):
                t_end = time.perf_counter() - t_start
        if t_full is None and all(cnt[k] >= cap[k] for k in queues):
            t_full = time.perf_counter() - t_start
            if t_end is None and any(head[k] >= len(queues[k]) for k in queues):
                t_end = t_full
        if now >= next_sample:
            curve.append((round(now, 2), len(live), sum(limit.values())))
            next_sample += sample_s
        time.sleep(poll_s)
        still = []
        tn = time.perf_counter()
        for r in live:
            if r.done.is_set():
                done.append(r)
                on_done(r, len(done), n)
            elif tn - r.t0 > timeout_s:
                dead.append(r)
            else:
                still.append(r)
        live = still
    t1 = time.perf_counter()
    info = {"caps": {str(k): cap[k] for k in queues}, "backs": {str(k): nb[k] for k in queues},
            "target": sum(cap.values()), "t_full_s": None if t_full is None else round(t_full, 3),
            "t_end_s": None if t_end is None else round(t_end, 3), "curve": curve}
    return done, dead, t_start, t1, info


def window_tokens(rows: Sequence[Mapping], t_a: float, t_b: float) -> float:
    """[t_a, t_b]（距压测开始的秒数）里生成的 token 数：每条请求的 token 按首 token → 最后一个 token
    均匀摊开，算落在窗口里的部分（首 token 那一个算在首 token 时刻）。"""
    tot = 0.0
    for r in rows:
        n = int(r.get("n_generated") or 0)
        if n <= 0:
            continue
        tf = float(r["t_submit_s"]) + float(r.get("prefill_ms") or 0) / 1e3
        td = float(r["t_done_s"])
        if t_a <= tf <= t_b:
            tot += 1
        if n > 1 and td > tf:
            ov = max(0.0, min(td, t_b) - max(tf, t_a))
            tot += (n - 1) * ov / (td - tf)
    return tot


def build_row(rec: Rec, *, mode: str, gw: Mapping | None, channels: Mapping[str, Mapping],
              t_run0: float) -> dict:
    gw = gw or {}
    chname = gw.get("channel") or rec.headers.get("X-P2P-Channel")
    ch = channels.get(chname) or {}
    nodes = list(ch.get("nodes") or [])
    total = (rec.t_last - rec.t0) * 1000
    ttft = ((rec.t_first or rec.t_last) - rec.t0) * 1000
    gaps = [(b - a) * 1000 for a, b in zip(rec.chunk_t, rec.chunk_t[1:])]
    n_gen = int((rec.usage or {}).get("completion_tokens") or len(rec.chunk_t))
    n_prompt = int((rec.usage or {}).get("prompt_tokens") or len(rec.ids))
    v = gw.get("verdict")
    sh = gw.get("shadow") if isinstance(gw.get("shadow"), dict) and "task" in (gw.get("shadow") or {}) else None
    eng = rec.engine or {}
    row = {
        "req": rec.rid,
        "prompt": rec.prompt,
        "text": rec.text,
        "stop_reason": stop_reason_of(rec.finish_reason, rec.stop_reason),
        "true_task": rec.true_task,
        "task": gw.get("task") or rec.headers.get("X-P2P-Task"),
        # oracle / shadow 不识别（路由按标注），correct 恒真没有信息量 → None
        "correct": (None if mode != "lr" else bool(gw.get("task") == rec.true_task)),
        "confidence": round(float((v or {}).get("conf", 0.0)), 4),
        "zone": (v or {}).get("zone", ""),
        **({"raw_scores": v["raw"]} if v and v.get("raw") else {}),
        **({"shadow": sh} if sh else {}),
        **({"n_body": sum(b - a for a, b in rec.body)} if rec.body is not None else {}),
        "rebinds": 0, "alarms": 0,
        # 分段网关（tiered_gateway.py）：前段、后段是逐请求挑的，路由记录里直接给节点
        "front": ("+".join(gw["front_nodes"]) if gw.get("front_nodes") else
                  (nodes[0] if nodes else "")),
        "back": ("+".join(gw["back_nodes"]) if gw.get("back_nodes") else "+".join(nodes[1:])),
        **({"front_engine": gw["front"]} if gw.get("front") else {}),
        # 多级：路线上的中段引擎与节点
        **({"mids": gw["mids"], "mid_nodes": "+".join(gw.get("mid_nodes") or []),
            "route": gw.get("route")} if gw.get("mids") else {}),
        **({"tiered": {k: gw.get(k) for k in ("front_prefill_ms", "bind_ms")}}
           if gw.get("front_nodes") else {}),
        "channel": chname,
        "n_prompt": n_prompt, "n_generated": n_gen,
        "total_ms": round(total, 2),
        "queue_ms": round(float(gw.get("route_ms") or 0.0), 2),
        "queue_back_ms": round(float(eng.get("queued_ms") or 0.0), 2),
        "prefill_ms": round(ttft, 2),
        "decode_ms": round(total - ttft, 2),
        "per_token_p50_ms": round(_median(gaps), 2),
        "compute_ms": 0.0,
        "network_and_overhead_ms": 0.0,
        "traces_complete": False,
        "utilisation": 0.0,
        "token_ms": [round(x, 2) for x in gaps],
        "marks": mark_rows(rec),
        "t_submit_s": round(rec.t0 - t_run0, 3),
        "t_done_s": round(rec.t_last - t_run0, 3),
        **({"case_id": rec.case_id} if rec.case_id else {}),
        "engine": eng or None,
        "identify": ({k: v.get(k) for k in ("queue_ms", "compute_ms", "server_ms", "rtt_ms",
                                            "n_body", "n_prompt")} if v else None),
        "nodes": [],
        "missing_traces": [],
    }
    return row


def summarise(rows: Sequence[dict], mode: str) -> dict:
    from p2pmoe.deploy.control import _shadow_acc

    return {
        "n": len(rows),
        "total_ms_p50": round(_median([r["total_ms"] for r in rows]), 2),
        "per_token_ms_p50": round(_median([r["per_token_p50_ms"] for r in rows]), 2),
        "utilisation_mean": 0.0,
        "rebinds": 0,
        "accuracy": (None if mode != "lr" else round(
            sum(bool(r["correct"]) for r in rows) / max(len(rows), 1), 4)),
        "shadow_accuracy": _shadow_acc(list(rows)),
        "errors": 0,
    }


# --------------------------------------------------------------------------- #
# nvidia-smi 采样（每台节点后台跑，结束收回）
# --------------------------------------------------------------------------- #
SMI_Q = "timestamp,utilization.gpu,memory.used,memory.total,power.draw"


def smi_start_cmd(tag: str) -> str:
    f = f"/tmp/p2p_smi_{tag}.csv"
    return (f"nohup nvidia-smi --id=0 --query-gpu={SMI_Q} --format=csv,noheader,nounits "
            f"-lms 1000 > {f} 2>/dev/null < /dev/null & echo $!")


def smi_stop_cmd(tag: str, pid: str) -> str:
    f = f"/tmp/p2p_smi_{tag}.csv"
    return f"kill {int(pid)} 2>/dev/null; sleep 0.2; cat {f}; rm -f {f}"


def summarise_smi(csv_text: str) -> dict:
    util, mem, tot, pw = [], [], [], []
    for ln in csv_text.splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 5:
            continue

        def num(x):
            try:
                return float(x)
            except ValueError:
                return None
        u, m, t, p = (num(x) for x in parts[1:5])
        if u is not None and 0 <= u <= 100:          # 防御：列错位时宁可丢样本也不出 584% 这种数
            util.append(u)
        if m is not None:
            mem.append(m)
        if t is not None:
            tot.append(t)
        if p is not None:
            pw.append(p)
    if not util:
        return {"error": "没有样本"}
    s = sorted(util)
    return {
        # vLLM 不报逐节点的 forward 时长 → 0（p2p-framework 的 report 要这个键）
        "compute_util": 0.0,
        "gpu_util_mean": round(sum(util) / len(util), 1),
        "gpu_util_p95": round(s[min(len(s) - 1, int(0.95 * len(s)))], 1),
        "mem_used_max_mb": max(mem) if mem else None,
        "mem_used_mean_mb": round(sum(mem) / len(mem), 1) if mem else None,
        "mem_total_mb": tot[0] if tot else None,
        "power_mean_w": round(sum(pw) / len(pw), 1) if pw else None,
        "n_samples": len(util),
    }


# --------------------------------------------------------------------------- #
# vLLM /metrics（各通道 API 进程的计数器，压测前后各取一次求差）
# --------------------------------------------------------------------------- #
_METRICS = ("vllm:num_preemptions_total", "vllm:request_queue_time_seconds_sum",
            "vllm:request_queue_time_seconds_count", "vllm:request_prefill_time_seconds_sum",
            "vllm:request_decode_time_seconds_sum", "vllm:prompt_tokens_total",
            "vllm:generation_tokens_total", "vllm:request_success_total")
_PROM = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)")


def parse_prom(text: str) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for ln in text.splitlines():
        if ln.startswith("#"):
            continue
        m = _PROM.match(ln)
        if m and m.group(1) in _METRICS:
            try:
                out[m.group(1)] += float(m.group(3))
            except ValueError:
                pass
    return dict(out)


def metrics_delta(a: Mapping[str, float], b: Mapping[str, float]) -> dict:
    d = {k: round(b.get(k, 0.0) - a.get(k, 0.0), 4) for k in _METRICS}
    n = d.get("vllm:request_queue_time_seconds_count") or 0
    if n:
        d["queue_ms_mean"] = round(1000 * d["vllm:request_queue_time_seconds_sum"] / n, 2)
    return d


# --------------------------------------------------------------------------- #
# A5：收回各节点的逐请求统计
# --------------------------------------------------------------------------- #
def stats_path(channel: str, rank: int, d: str = "/tmp/p2p_stats") -> str:
    return f"{d}/{channel}_pp{rank}.jsonl"


def join_stats(rows: list[dict], stats: Mapping[str, list[dict]]) -> int:
    """把 {p2p_id: [各级记录]} 拼进结果行。返回拼上的条数。"""
    n = 0
    for r in rows:
        recs = sorted(stats.get(r["req"], []), key=lambda x: x["pp_rank"])
        if not recs:
            continue
        n += 1
        stages = []
        ntl = mtl = 0
        mm = 0.0
        dntl = dmtl = 0
        for s in recs:
            pre, dec = s["prefill"], s["decode"]
            ntl += pre["n_token_layer"] + dec["n_token_layer"]
            mtl += pre["miss_token_layer"] + dec["miss_token_layer"]
            mm += pre["miss_mass"] + dec["miss_mass"]
            dntl += dec["n_token_layer"]
            dmtl += dec["miss_token_layer"]
            stages.append({
                "pp_rank": s["pp_rank"], "layers": [s["layers"][0], s["layers"][-1]]
                if s["layers"] else [], "n_seen": s["n_seen"],
                "prefill_miss_rate": round(pre["miss_token_layer"] / pre["n_token_layer"], 4)
                if pre["n_token_layer"] else None,
                "decode_miss_rate": round(dec["miss_token_layer"] / dec["n_token_layer"], 4)
                if dec["n_token_layer"] else None,
                "decode_miss_seq": s.get("decode_miss_seq"),
            })
        r["stats"] = {"miss_rate": round(mtl / ntl, 4) if ntl else None,
                      "decode_miss_rate": round(dmtl / dntl, 4) if dntl else None,
                      "miss_mass": round(mm, 4), "stages": stages}
        front = recs[0]
        if front.get("lr_layers"):
            r["_front_layers"] = front["lr_layers"]
            r["_front_body"] = front.get("body")
    return n


def a5_check(rows: list[dict], gw_records: Mapping[str, dict], clf) -> dict:
    """钩子统计 → LR，与识别服务（同一条请求）的结论逐条对。"""
    import numpy as np

    from p2pmoe.runtime.lr_classifier import verdict_label
    from p2pmoe.runtime.model import LayerStat, MoEStats

    agree = n = 0
    dmax = 0.0
    cnt_eq = cnt_n = 0
    diff_by_layer: dict[int, list[float]] = defaultdict(list)
    bad = []
    for r in rows:
        fl = r.get("_front_layers")
        g = gw_records.get(r["req"]) or {}
        ref = g.get("verdict") or (g.get("shadow") if isinstance(g.get("shadow"), dict) else None)
        if not fl or not ref or "task" not in ref:
            continue
        st = MoEStats(hist=np.zeros(clf.n_experts),
                      layers={int(l): LayerStat.from_wire(s) for l, s in fl.items()})
        v = clf.predict_stats(st)
        n += 1
        ok = verdict_label(v) == (ref.get("label") or verdict_label(ref))
        agree += ok
        if not ok:
            bad.append(r["req"])
        if v.raw and ref.get("raw"):
            dmax = max(dmax, max(abs(v.raw[k] - ref["raw"].get(k, 0.0)) for k in v.raw))
        if ref.get("layers"):
            for l, s in ref["layers"].items():
                if l in fl:
                    a, b = LayerStat.from_wire(s), LayerStat.from_wire(fl[l])
                    cnt_n += 1
                    cnt_eq += bool(np.array_equal(a.count, b.count))
                    # 有多大比例的 top-k 选择不一样：Σ|Δcount| / 2 / (body token × k)
                    tot = max(float(a.count.sum()), 1.0)
                    diff_by_layer[int(l)].append(float(np.abs(a.count - b.count).sum()) / 2 / tot)
    out = {"n": n, "label_agree": agree, "max_abs_prob_diff": round(dmax, 4),
           "disagree": bad[:20]}
    if cnt_n:
        out["layer_count_equal"] = f"{cnt_eq}/{cnt_n}"
        # 逐层：两边 top-k 选择不同的比例（均值）。fp16 下两套实现（vLLM 融合 kernel vs
        # p2p-framework 逐专家 matmul）的舍入不同，并列附近的选择会换位，越深的层差得越多
        out["topk_diff_by_layer"] = {l: round(sum(v) / len(v), 4)
                                     for l, v in sorted(diff_by_layer.items())}
    return out


def write_front_dump(path: Path, rows: list[dict], gw_records: Mapping[str, dict], *,
                     l0: int, n_experts: int, deployed: Sequence[str]) -> int:
    """钩子的前 L₀ 层统计 → p2p-framework `--dump-front-stats` 同格式（给 lr_tool check）。"""
    from p2pmoe.runtime.lr_classifier import verdict_label

    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            fl = r.get("_front_layers")
            if not fl:
                continue
            g = gw_records.get(r["req"]) or {}
            on = g.get("verdict") or (g.get("shadow") if isinstance(g.get("shadow"), dict) else None)
            on = on if on and "task" in on else None
            f.write(json.dumps({
                "req": r["req"], "true_task": r["true_task"],
                **({"case_id": r["case_id"]} if r.get("case_id") else {}),
                "lr_task": on["task"] if on else None,
                "lr_label": (on.get("label") or verdict_label(on)) if on else None,
                "lr_conf": on.get("conf") if on else None,
                "l0": l0, "n_experts": n_experts, "layer_base": 1,
                "deployed": list(deployed), "n_prompt": r["n_prompt"], "prompt": r["prompt"],
                **({"body": r["_front_body"], "n_body": r.get("n_body")}
                   if r.get("_front_body") else {}),
                "layers": fl, "source": "vllm-a5-hook",
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gateway", default="http://127.0.0.1:9000")
    ap.add_argument("--channels", default="/home/ubuntu/vllm-p2p/p2p_channels/channels.json")
    ap.add_argument("--cases", default=None, type=Path,
                    help="带 task 标注的测试集（TestSamples/cases.jsonl 或 qwen3-30b/sel_cases.jsonl）；"
                         "用 --synthetic 时不需要")
    ap.add_argument("--synthetic", action="store_true",
                    help="不用真实测试集，按网关上的 task 生成合成请求（只测吞吐 / TPOT）；"
                         "配合网关 --ignore-eos，每条都生成满 --max-tokens")
    ap.add_argument("--synth-per-task", type=int, default=64, help="--synthetic：每个 task 几条不同的 prompt")
    ap.add_argument("--synth-prompt-len", type=int, default=256, help="--synthetic：prompt 约几个 token")
    ap.add_argument("--synth-seed", type=int, default=0)
    ap.add_argument("--model-dir", required=True, help="有 tokenizer.json 的目录（控制机本地）")
    ap.add_argument("--p2p-framework", default=None)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--offline", action="store_true",
                    help="离线满载：所有请求一次全部提交（在途 = 请求数），由各引擎自己连续批处理、排队；"
                         "看的是吞吐与总耗时。各后段的 --max-num-seqs 要放大才真的满载")
    ap.add_argument("--saturate", action="store_true",
                    help="满载连续测试：从每个后段 1 条起，每 --ramp-every 秒每个后段多放 --ramp-step 条，"
                         "加到满载（每个后段 = 它的 max_num_seqs 或 --per-back）后一条完成补一条、"
                         "始终保持满载，直到请求发完；另报满载段（坐满之后提交、发完之前完成）的统计")
    ap.add_argument("--saturate-by", choices=("auto", "task", "global"), default="auto",
                    help="满载按 task 分别保持（每个 task 在途 = 它各后段的槽位和，按真实标注放请求，"
                         "路由照旧由网关定）还是只保持总在途；auto = any 模式 global、其余 task")
    ap.add_argument("--per-back", type=int, default=0,
                    help="满载时每个后段同时几条；0 = 用 stages/channels 里它的 max_num_seqs")
    ap.add_argument("--ramp-step", type=int, default=1, help="加压：每次每个后段多放几条")
    ap.add_argument("--ramp-every", type=float, default=15.0, help="加压：每隔几秒加一次；0 = 直接满载")
    ap.add_argument("--mix", choices=("capacity", "uniform"), default="capacity",
                    help="--saturate-by task 时各 task 的请求数：capacity = 按满载槽位成比例（各 task "
                         "差不多同时发完，满载段最长；样本不够就循环）；uniform = 测试集原样（三类各一份）")
    ap.add_argument("--inflight", type=int, default=0,
                    help="同时在途条数；默认 = 通道数（与本框架默认 = 后段通道数同口径）")
    ap.add_argument("--requests", type=int, default=0, help="0 = 测试集全部")
    ap.add_argument("--order", choices=("interleave", "file"), default="interleave")
    ap.add_argument("--warmup", type=int, default=1, help="每条通道先打几条预热（不计入结果）")
    ap.add_argument("--marks", default="64,128,256,512,1024,2048",
                    help="每条请求生成到这些 token 数时打点（结果里 requests[i].marks，汇总里有一张表）")
    ap.add_argument("--req-timeout", type=float, default=3600.0)
    ap.add_argument("--thinking", choices=("off", "on", "default"), default="off")
    ap.add_argument("--smi", action="store_true", help="压测期间在各节点采 nvidia-smi")
    ap.add_argument("--stats", action="store_true", help="收回 A5 钩子的逐请求统计（通道要 --stats 起）")
    ap.add_argument("--stats-dir", default="/tmp/p2p_stats")
    ap.add_argument("--lr-model", default=None, help="给了就用钩子统计重算识别，与识别服务逐条对")
    ap.add_argument("--lr-classes", default=None)
    ap.add_argument("--user", default="ubuntu")
    ap.add_argument("--out", type=Path, default=Path("results/vllm_continuous.json"))
    ap.add_argument("--label", default="", help="写进结果的备注（如 E3 / E4）")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    use_p2p_framework(a.p2p_framework)
    from p2pmoe.deploy.continuous import interleave, run_closed_loop, run_metrics
    from p2pmoe.deploy.control import load_cases
    from p2pmoe.runtime.text import TextIO

    cj = json.loads(Path(a.channels).read_text(encoding="utf-8"))
    channels = {c["name"]: c for c in cj["channels"]}
    marks = tuple(sorted({int(x) for x in str(a.marks).split(",") if x.strip()}))
    try:
        gw_info = http_json(f"{a.gateway}/p2p/channels")
    except OSError as e:
        raise SystemExit(
            f"连不上网关 {a.gateway}（{e}）—— 网关没起或已经退出。先起网关（要一直开着，比如另开一个终端"
            "或用 nohup 放后台）：\n  /home/ubuntu/venvs/vllm-p2p/bin/python vllm/p2p/gateway.py "
            "--mode shadow --identifier http://<识别服务所在节点>:8100 --ident-layers") from None
    mode = gw_info["mode"]
    live = {c["name"] for c in gw_info["channels"]}
    tasks = sorted({channels[c]["task"] for c in live if c in channels})
    log.info("网关 %s（%s 模式），通道 %s", a.gateway, mode, sorted(live))

    textio = TextIO.from_model_dir(a.model_dir, chat=True)
    if a.thinking != "default":
        textio.template_kw["enable_thinking"] = a.thinking == "on"
    if a.synthetic:
        cases = synthetic_cases(tasks, a.synth_per_task, a.synth_prompt_len, a.synth_seed)
        log.info("合成请求：%d 个 task × %d 条，prompt 约 %d token（内容与任务无关，只测吞吐 / TPOT）",
                 len(tasks), a.synth_per_task, a.synth_prompt_len)
        if not gw_info.get("ignore_eos"):
            log.warning("  网关没开 --ignore-eos：合成 prompt 的输出长度不可控（可能很早遇到 EOS）")
    elif a.cases is None:
        raise SystemExit("要给 --cases 测试集，或者用 --synthetic 生成合成请求")
    else:
        cases = load_cases(a.cases, tasks)
    if mode in ("oracle", "shadow"):
        # 按标注路由时，网关上没有通道的 task 只会 503 —— 直接跳过，免得压测里混进一堆失败
        # （网关 --only 只开了部分通道、或分段网关只起了部分后段时）
        skip = [u for u, _ in cases if u is not None and u not in tasks]
        if skip:
            log.info("网关上没有 %s 的通道：这 %d 条跳过", sorted(set(skip)), len(skip))
            cases = [(u, p) for u, p in cases if u is None or u in tasks]
    if not cases:
        raise SystemExit(f"{a.cases or '合成请求'} 里一条都没有")
    pool = interleave(cases) if a.order == "interleave" else list(cases)
    n = len(pool) if a.requests <= 0 else a.requests
    inflight = n if a.offline else (a.inflight or len(live))
    seq = [i % len(pool) for i in range(n)]          # 第 j 条请求用 pool[seq[j]]
    caps = None
    if a.saturate:
        if a.offline:
            raise SystemExit("--saturate 与 --offline 只能二选一")
        by_task = a.saturate_by == "task" or (a.saturate_by == "auto" and mode != "any")
        caps = saturation_caps(live, channels, by_task=by_task, per_back=a.per_back)
        if by_task:
            miss = sorted({str(u) for u, _ in pool if u not in caps})
            if miss:
                log.info("满载按 task：%s 没有后段，这些样本不发", miss)
                pool = [x for x in pool if x[0] in caps]
                n = len(pool) if a.requests <= 0 else a.requests
            seq = (capacity_mix(pool, n, caps) if a.mix == "capacity"
                   else [i % len(pool) for i in range(n)])
        inflight = sum(c for c, _ in caps.values())
        log.info("[满载] 按 %s 保持；满载在途 %s（共 %d），从每个后段 1 条起、每 %gs 每个后段 +%d",
                 "task" if by_task else "全局",
                 {str(k): f"{c}（{b} 个后段）" for k, (c, b) in caps.items()}, inflight,
                 a.ramp_every, a.ramp_step)
        if by_task and a.mix == "capacity":
            cnt = defaultdict(int)
            for i in seq:
                cnt[pool[i][0]] += 1
            log.info("  请求按满载槽位分给各 task：%s（样本不够的循环用）", dict(sorted(cnt.items())))
        if n < 2 * inflight:
            log.warning("  请求只有 %d 条、满载在途 %d：坐满后很快就发完，满载段会很短 —— "
                        "加 --requests（比如 %d，样本循环用）", n, inflight, 4 * inflight)

    def encode(payload) -> tuple[list[int], list[tuple[int, int]], str, str | None]:
        if isinstance(payload, dict):
            segs, system, cid = payload["segments"], payload.get("system"), payload.get("id")
        else:
            segs, system, cid = [(payload, False)], None, None
        bp = textio.encode_body(segs, system=system)
        return bp.ids, bp.body, "".join(t for t, _ in segs), cid

    def make_rec(i: int, tag: str, task: str | None, payload, max_tokens: int) -> Rec:
        ids, body, prompt, cid = encode(payload)
        rec = Rec(i=i, rid=f"{tag}{i}", true_task=task, payload=payload, ids=ids, body=body,
                  case_id=cid, prompt=prompt, marks_want=marks)
        req = {"prompt": ids, "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
               # continuous_usage_stats：每块带累计生成数，生成到 --marks 那几档时打点
               "stream_options": {"include_usage": True, "continuous_usage_stats": True},
               "skip_special_tokens": True,
               "p2p_id": rec.rid, "p2p_body": [list(x) for x in body]}
        if task:
            req["p2p_task"] = task
        rec.t0, rec.t0_wall = time.perf_counter(), time.time()
        threading.Thread(target=post_stream, args=(a.gateway, req, rec, a.req_timeout),
                         daemon=True).start()
        return rec

    hosts = {c["name"]: [(st["node"], st["host"]) for st in c.get("commands", [])]
             for c in cj["channels"]}
    # 分段网关还有前段引擎（不属于任何「通道」）：它们的节点也要采 smi
    fronts_live = [f["name"] for f in gw_info.get("fronts") or []]
    mids_live = [m["name"] for m in gw_info.get("mids") or []]       # 多级：中段引擎
    all_hosts = sorted({nh for c in (*live, *fronts_live, *mids_live) for nh in hosts.get(c, [])})
    if caps and fronts_live:
        fcap = sum(int((channels.get(f) or {}).get("max_num_seqs") or 0) for f in fronts_live)
        if fcap and fcap < inflight:
            log.warning("[满载] 前段 max_num_seqs 合计 %d < 满载在途 %d：每个 token 都要过前段，前段会先满、"
                        "后段坐不满 —— stages.py 加大 --front-max-num-seqs", fcap, inflight)

    # ---- 预热：每条通道至少一条（task 轮着来，网关按最闲分到同 task 的各条） ---- #
    if a.warmup:
        by_task = defaultdict(list)
        for u, p in cases:
            by_task[u].append(p)
        warm = []
        k = 0
        for c in sorted(live):
            u = channels[c]["task"]
            for _ in range(a.warmup):
                if by_task.get(u):
                    warm.append((u, by_task[u][k % len(by_task[u])]))
                    k += 1
        log.info("预热 %d 条（不计入结果）……", len(warm))
        ws = [make_rec(j, "warm", u, p, 16) for j, (u, p) in enumerate(warm)]
        for w in ws:
            w.done.wait(a.req_timeout)
        bad = [w for w in ws if w.error]
        if bad:
            log.error("预热失败 %d 条，第一条：%s", len(bad), bad[0].error)
            return 2
    try:
        http_json(f"{a.gateway}/p2p/reset", {})
    except Exception:                                         # noqa: BLE001
        pass

    # ---- 资源采样与引擎计数器 ---- #
    tag = f"{int(time.time())}"
    smi_pids = {}
    if a.smi:
        for node, host in all_hosts:
            r = _launch._ssh(host, a.user, smi_start_cmd(tag), timeout=30)
            if r.returncode == 0 and r.stdout.strip().isdigit():
                smi_pids[(node, host)] = r.stdout.strip()
            else:
                log.warning("  %s 上 nvidia-smi 起不来：%s", node, r.stderr.strip()[:200])
    if a.stats:
        for c in live:
            for k, (node, host) in enumerate(hosts.get(c, [])):
                _launch._ssh(host, a.user, f"mkdir -p {shlex.quote(a.stats_dir)} && : > "
                             f"{shlex.quote(stats_path(c, k, a.stats_dir))}", timeout=30)
    m0 = {}
    for c in live:
        try:
            m0[c] = parse_prom(http_text(channels[c]["api"].rsplit("/v1", 1)[0] + "/metrics"))
        except Exception:                                     # noqa: BLE001
            pass

    # ---- 闭环 ---- #
    log.info("[连续测试] %d 条（池 %d，%s），在途 %d，max_tokens %d，单条超时 %.0fs",
             n, len(pool), a.order, inflight, a.max_tokens, a.req_timeout)

    def submit(i):
        u, p = pool[seq[i]]
        return make_rec(i, "req", u, p, a.max_tokens)

    def done(rec, k, total):
        mark = "✗ " + rec.error[:120] if rec.error else ""
        log.info("  [%d/%d] %s %s → %s  %.0fms  首token %.0fms  %s",
                 k, total, rec.rid, rec.true_task, rec.headers.get("X-P2P-Channel", "?"),
                 (rec.t_last - rec.t0) * 1000, ((rec.t_first or rec.t_last) - rec.t0) * 1000,
                 mark)

    sat = None
    if caps:
        def level(t, lv, limit):
            log.info("  [加压] 第 %.0fs：每个后段 ≤ %d 条，在途上限 %d / 满载 %d", t, lv,
                     sum(limit.values()), inflight)

        def key_of(j):
            k = pool[seq[j]][0]
            return k if k in caps else None

        recs, dead, t0, t1, sat = run_ramp_loop(range(n), key_of, caps, submit, step=a.ramp_step,
                                                every=a.ramp_every, timeout_s=a.req_timeout,
                                                on_done=done, on_level=level)
        if sat["t_full_s"] is None:
            log.warning("[满载] 一直没坐满（请求不够或加压太慢）—— 满载段为空")
        else:
            log.info("[满载] 第 %.0fs 坐满，第 %.0fs 第一个 task 发完 → 满载段 %.0fs", sat["t_full_s"],
                     sat["t_end_s"], sat["t_end_s"] - sat["t_full_s"])
    else:
        recs, dead, t0, t1 = run_closed_loop(n, submit, inflight=inflight, timeout_s=a.req_timeout,
                                             on_done=done)
    wall = t1 - t0

    # ---- 收尾：计数器、smi、网关记录、A5 统计 ---- #
    mdelta = {}
    for c in live:
        try:
            mdelta[c] = metrics_delta(m0.get(c, {}), parse_prom(
                http_text(channels[c]["api"].rsplit("/v1", 1)[0] + "/metrics")))
        except Exception as e:                                # noqa: BLE001
            mdelta[c] = {"error": str(e)}
    usage = {}
    for (node, host), pid in smi_pids.items():
        r = _launch._ssh(host, a.user, smi_stop_cmd(tag, pid), timeout=60)
        usage[node] = summarise_smi(r.stdout) if r.returncode == 0 else {"error": r.stderr[:200]}
        if r.returncode == 0:
            usage[node]["raw_head"] = r.stdout.splitlines()[:2]      # 核对列用
    for node in sorted(usage):
        v = usage[node]
        log.info("  GPU %s：利用率均值 %s%%  p95 %s%%  显存峰值 %s/%s MB  功耗 %s W  （%s 个样本）",
                 node, v.get("gpu_util_mean"), v.get("gpu_util_p95"), v.get("mem_used_max_mb"),
                 v.get("mem_total_mb"), v.get("power_mean_w"), v.get("n_samples"))
    gw = http_json(f"{a.gateway}/p2p/records")
    gwr = {g["id"]: g for g in gw["records"]}

    ok = [r for r in recs if not r.error]
    failed = [{"req": r.rid, "true_task": r.true_task, "error": r.error} for r in recs if r.error]
    failed += [{"req": r.rid, "true_task": r.true_task, "error": "timeout"} for r in dead]
    rows = [build_row(r, mode=mode, gw=gwr.get(r.rid), channels=channels, t_run0=t0)
            for r in sorted(ok, key=lambda r: r.i)]

    a5 = None
    if a.stats:
        # 最后一批请求的「结束」要等下一步调度才送到各级（vLLM 的 finished_req_ids），
        # 等一会儿再收；空闲时调度器也会补一步把它们送出去
        time.sleep(3.0)
        got: dict[str, list[dict]] = defaultdict(list)
        for c in live:
            for k, (node, host) in enumerate(hosts.get(c, [])):
                r = _launch._ssh(host, a.user, f"cat {shlex.quote(stats_path(c, k, a.stats_dir))}"
                                 " 2>/dev/null || true", timeout=120)
                for ln in r.stdout.splitlines():
                    try:
                        d = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    if d.get("p2p_id"):
                        got[d["p2p_id"]].append(d)
        nj = join_stats(rows, got)
        log.info("A5 统计：%d/%d 条拼上了各级记录", nj, len(rows))
        dump = a.out.with_suffix(".front.jsonl")
        dump.parent.mkdir(parents=True, exist_ok=True)
        nd = write_front_dump(dump, rows, gwr, l0=int(cj["l0"]),
                              n_experts=int(cj.get("n_experts") or 128), deployed=tasks)
        if nd:
            log.info("  前 L₀ 层统计 → %s（%d 条）；离线复核：python3 -m p2pmoe.deploy.lr_tool "
                     "check --model <选择器> --dump %s", dump, nd, dump)
        if a.lr_model:
            from p2pmoe.runtime.lr_classifier import LRClassifier

            clf = LRClassifier.load(a.lr_model, l0=int(cj["l0"]),
                                    n_experts=int(cj.get("n_experts") or 128),
                                    priors={u: 1.0 for u in tasks}, class_map=a.lr_classes)
            a5 = a5_check(rows, gwr, clf)
            log.info("  A5 核对（钩子特征 → LR vs 识别服务）：%s", a5)
        for r in rows:
            r.pop("_front_layers", None)
            r.pop("_front_body", None)

    out = {
        "model": f"vLLM {mode} · {Path(cj.get('plan', '')).name}",
        "classifier": gw_info.get("identifier") and f"识别服务 {gw_info['identifier']}",
        "l0": int(cj.get("l0", 0)),
        "backend": "vllm",
        "mode": {"oracle": "static", "shadow": "static", "any": "static", "lr": "dynamic"}[mode],
        "gateway_mode": mode,
        "label": a.label,
        "miss_policy": "local_topk",
        "coverage": None,
        "tokens": a.max_tokens,
        "concurrency": inflight,
        "channels": {c: {"nodes": channels[c]["nodes"], "partition": channels[c].get("partition"),
                         "task": channels[c].get("task"),
                         "gpu_mem": channels[c].get("gpu_mem"),
                         "engine_metrics": mdelta.get(c)} for c in sorted(live)},
        # 多级：中段引擎（按 task 归到各自的列；路线见 routes）
        **({"mids": {m: {"nodes": channels[m]["nodes"], "task": channels[m].get("task"),
                         "gpu_mem": channels[m].get("gpu_mem")} for m in mids_live if m in channels},
            "routes": gw_info.get("routes")} if mids_live else {}),
        "requests": rows,
        "failed": failed,
        "summary": summarise(rows, mode),
        **({"a5_check": a5} if a5 else {}),
    }
    out["summary"]["errors"] = len(failed)
    out["run"] = run_metrics(rows, wall, usage=usage or None, inflight=inflight,
                             n_timeout=len(dead))
    out["run"]["order"] = a.order
    out["run"]["max_tokens"] = a.max_tokens
    out["run"]["marks"] = list(marks)
    out["run"]["offline"] = bool(a.offline)
    if sat is not None:
        ta, tb = sat["t_full_s"], sat["t_end_s"]
        for r in rows:
            r["phase"] = ("ramp" if ta is None or r["t_submit_s"] < ta else
                          "steady" if r["t_done_s"] <= tb else "tail")
        steady = [r for r in rows if r["phase"] == "steady"]
        win = (tb - ta) if ta is not None else 0.0
        wt = window_tokens(rows, ta, tb) if win > 0 else 0.0
        inw = [c[1] for c in sat["curve"] if ta is not None and ta <= c[0] <= tb]
        sat.update(by="task" if any(k != "None" for k in sat["caps"]) else "global",
                   per_back=a.per_back, ramp_step=a.ramp_step, ramp_every=a.ramp_every, mix=a.mix,
                   window_s=round(win, 3), window_output_tokens=round(wt, 1),
                   window_tok_per_s=round(wt / win, 3) if win > 0 else None,
                   window_inflight_mean=round(sum(inw) / len(inw), 2) if inw else None,
                   n_steady=len(steady),
                   n_phase={p: sum(r["phase"] == p for r in rows) for p in ("ramp", "steady", "tail")})
        out["run"]["saturate"] = sat
        if steady:
            out["steady"] = run_metrics(steady, win, inflight=inflight)
    if mode in ("oracle", "any"):
        # 路由按标注（any：不看 task），没有识别 —— run_metrics 会拿路由结果当「识别」算出个 100%，去掉免得误读
        out["run"]["accuracy"] = None
        for v in out["run"]["by_task"].values():
            v["accuracy"] = None
    out["run"]["per_request"] = request_stats(rows)
    out["run"]["marks_stats"] = mark_stats(rows)
    out["run"]["segment_stats"] = segment_stats(rows)
    out["run"]["cumulative_stats"] = cumulative_stats(rows)
    # 汇总表（全程；--saturate 再加满载段）：列 = 全部 / 各 task 的路径 / 前段（共享）
    out["run"]["summary_table"] = summary_table(out)
    if sat is not None:
        out["run"]["summary_table_steady"] = summary_table(out, steady=True)
    # 与 p2p-framework 连续性测试同一格式的汇总（均值 / 标准差 / 分位数、按 task、资源占用）
    log.info("")
    for ln in continuous_report(out):
        log.info("%s", ln)
    if sat is not None:
        log.info("")
        for ln in saturate_report(out):
            log.info("%s", ln)
    if failed:
        log.warning("  失败 %d 条，第一条：%s", len(failed), failed[0])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = a.out.with_suffix(".summary.csv")
    csv_path.write_text(summary_csv([out["run"].get("summary_table"),
                                     out["run"].get("summary_table_steady")]), encoding="utf-8-sig")
    log.info("汇总表（CSV）已写入 %s", csv_path)
    log.info("结果已写入 %s（%d 条）—— 看表：cd <p2p-framework> && ./deploy_15.sh report %s",
             a.out, len(rows), a.out.resolve())
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
