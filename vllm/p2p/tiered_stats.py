# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B1：分段结果的开销拆解，可选与 A 的结果按通道并排。**纯标准库，按文件路径运行。**

    python3 vllm/p2p/tiered_stats.py --b results/b1_gsm8k_inflight8.json [--a results/a3_gsm8k_inflight8.json]

按（前段节点, 后段引擎）分组：条数、TTFT / TPOT 中位数，以及分段网关逐请求报的
「前段 prefill」「绑定」「前段腿」（后段出 token → 前段 hidden 回到协调器）「后段腿」
（hidden 送出 → 下一个 token 回来）。给了 --a 时再列 A 同一条通道（后段 back-X ↔ 通道 X）的 TPOT。
每 token 的时间 ≈ 前段腿 + 后段腿；两条腿里都含两次跨机、一次调度、一次前向。

开头先打与 p2p-framework 连续性测试同一格式的汇总（`continuous_report`：各指标的均值 / 标准差 / p50 / p90 /
p99 / 最大，单请求 token/s 含排队 / 去排队 / 解码，吞吐、识别，按 task 的表，各节点资源占用）；
`bench.py` 跑完打的也是这张，逐请求统计另写进结果的 `run.per_request`。

最后是汇总表（`summary_table`）：E2E / TTFT（去排队）、TPOT、输出长度、单请求与解码 token/s、
输出 / 含 prompt 吞吐、请求/分钟、总输出 token、GPU 利用率、显存峰值与总量、功耗；列为全部、
各 task（该 task 的请求 + 它的后段流水线）、前段（共享）。`--csv` 写成 CSV。
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def _med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(st.median(xs), 1) if xs else None


def _tpot(r):
    n = int(r.get("n_generated") or 0)
    return (r["total_ms"] - r["prefill_ms"]) / (n - 1) if n > 1 else None


def _ms(xs):
    """→ {n, mean, std}（样本标准差；只有 1 个值时 std = 0）。"""
    xs = [float(x) for x in xs if isinstance(x, (int, float))]
    if not xs:
        return {"n": 0, "mean": None, "std": None}
    return {"n": len(xs), "mean": round(st.fmean(xs), 3),
            "std": round(st.stdev(xs), 3) if len(xs) > 1 else 0.0}


def _tok_s(r):
    """一条请求自己的速度：生成 token 数 / 这条从发出到收完的时间（含 TTFT）。"""
    n = int(r.get("n_generated") or 0)
    t = float(r.get("total_ms") or 0)
    return n / (t / 1e3) if n > 0 and t > 0 else None


def _decode_tok_s(r):
    """只看 decode 段：1000 / TPOT。"""
    tp = _tpot(r)
    return 1e3 / tp if tp and tp > 0 else None


def request_stats(rows) -> dict:
    """逐请求的 TTFT / TPOT 均值与标准差、每条请求的 token/s 取均值；整体 + 按 task。"""
    def one(rs):
        return {"n": len(rs),
                "ttft_ms": _ms([r.get("prefill_ms") for r in rs]),
                "tpot_ms": _ms([_tpot(r) for r in rs]),
                "tok_per_s": _ms([_tok_s(r) for r in rs]),
                "decode_tok_per_s": _ms([_decode_tok_s(r) for r in rs])}
    by = defaultdict(list)
    for r in rows:
        by[r.get("true_task") or "?"].append(r)
    return {"all": one(list(rows)), "by_task": {u: one(rs) for u, rs in sorted(by.items())},
            "note": "均值 ± 样本标准差；tok_per_s = 每条请求 生成 token / 总时延（含 TTFT），"
                    "decode_tok_per_s = 1000 / TPOT，都是先逐条算再对所有请求取均值"}


# --------------------------------------------------------------------------- #
# 与 p2p-framework 连续性测试同一格式的汇总
# --------------------------------------------------------------------------- #
def _pct(xs, q):
    """线性插值分位数（与 numpy.percentile 默认口径相同）。"""
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def full_stats(xs) -> dict:
    """→ {n, mean, std, p50, p90, p99, max}（std 为样本标准差）。"""
    xs = [float(x) for x in xs if isinstance(x, (int, float))]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": st.fmean(xs), "std": st.stdev(xs) if len(xs) > 1 else 0.0,
            "p50": _pct(xs, 50), "p90": _pct(xs, 90), "p99": _pct(xs, 99), "max": max(xs)}


def _queue(r):
    return float(r.get("queue_ms") or 0.0) + float(r.get("queue_back_ms") or 0.0)


def _tok_s_service(r):
    """去排队：拿到前后段之后到最后一个 token。"""
    n = int(r.get("n_generated") or 0)
    t = float(r.get("total_ms") or 0) - _queue(r)
    return n / (t / 1e3) if n > 0 and t > 0 else None


def _label_ok(r):
    raw = r.get("raw_scores")
    lab = str(max(raw, key=raw.get)).split("#")[0] if raw else r.get("task")
    return None if not r.get("true_task") else lab == r["true_task"]


def mark_stats(rows) -> dict:
    """bench 的逐请求打点（requests[i].marks）→ 每一档 k：到达的条数，以及各量的均值 / 标准差 / 分位数；
    另按 task 给平均 TPOT 与解码 token/s。只统计生成到了 k 的请求（短回答不进长的档）。"""
    ks = sorted({int(k) for r in rows for k in (r.get("marks") or {})})
    out = {"all": {}, "by_task": {}}
    for k in ks:
        ms = [r["marks"][str(k)] for r in rows if str(k) in (r.get("marks") or {})]
        out["all"][str(k)] = {"n": len(ms), **{f: full_stats([m.get(f) for m in ms])
                                               for f in ("t_ms", "since_first_ms", "tpot_ms",
                                                         "tok_per_s", "decode_tok_per_s")}}
        by = defaultdict(list)
        for r in rows:
            m = (r.get("marks") or {}).get(str(k))
            if m:
                by[r.get("true_task") or "?"].append(m)
        for u, ms_u in by.items():
            out["by_task"].setdefault(u, {})[str(k)] = {
                "n": len(ms_u), "tpot_ms": full_stats([m.get("tpot_ms") for m in ms_u]),
                "decode_tok_per_s": full_stats([m.get("decode_tok_per_s") for m in ms_u])}
    return out


def segment_stats(rows) -> dict:
    """相邻两个打点之间的平均 token 间隔（ms / token）：首 token→第 k₁ 个、k₁→k₂、…。
    每条请求：(t_b − t_a) / (n_b − n_a)，n 用打点时实际已生成的数；只统计两端都到了的请求。
    看「已经生成了多长」时每个 token 要多久（上下文越长、后段越忙，间隔怎么变）。"""
    ks = sorted({int(k) for r in rows for k in (r.get("marks") or {})})
    segs = [("1", str(ks[0]))] + [(str(a), str(b)) for a, b in zip(ks, ks[1:])] if ks else []
    out = {"segments": [f"{a}→{b}" for a, b in segs], "all": {}, "by_task": {}}
    for a, b in segs:
        name = f"{a}→{b}"
        vals, by = [], defaultdict(list)
        for r in rows:
            m = r.get("marks") or {}
            if b not in m:
                continue
            mb = m[b]
            if a == "1":
                sf, n = mb.get("since_first_ms"), mb.get("n") or 0
                v = sf / (n - 1) if sf is not None and n > 1 else None
            elif a in m and (mb["n"] - m[a]["n"]) > 0:
                v = (mb["t_ms"] - m[a]["t_ms"]) / (mb["n"] - m[a]["n"])
            else:
                v = None
            if v is not None:
                vals.append(v)
                by[r.get("true_task") or "?"].append(v)
        out["all"][name] = full_stats(vals)
        for u, vs in by.items():
            out["by_task"].setdefault(u, {})[name] = full_stats(vs)
    return out


CUM_KS = (64, 128, 256, 512, 1024, 2048)


def cumulative_stats(rows, ks=CUM_KS) -> dict:
    """首 token → 第 k 个 token 的平均间隔（ms / token）= (第 k 个 − 首 token) / (k − 1)。
    有第 k 个的打点就用打点；没有这一档打点、但这条正好生成了 k 个（如 max_tokens=2048 截断的）时，
    用整条的 (总时延 − 首 token) / (k − 1)。只统计生成到了 k 的请求。"""
    names = [f"1→{k}" for k in ks]
    out = {"segments": names, "all": {}, "by_task": {}}
    for k, name in zip(ks, names):
        vals, by = [], defaultdict(list)
        for r in rows:
            m = (r.get("marks") or {}).get(str(k))
            v = None
            if m and m.get("since_first_ms") is not None and (m.get("n") or 0) > 1:
                v = m["since_first_ms"] / (m["n"] - 1)
            elif not m and int(r.get("n_generated") or 0) == k and k > 1:
                v = (float(r["total_ms"]) - float(r["prefill_ms"])) / (k - 1)
            if v is not None:
                vals.append(v)
                by[r.get("true_task") or "?"].append(v)
        out["all"][name] = full_stats(vals)
        for u, vs in by.items():
            out["by_task"].setdefault(u, {})[name] = full_stats(vs)
    return out


def format_segment_stats(ss: dict, title: str | None = None, note: str | None = None) -> list[str]:
    if not ss.get("segments"):
        return []
    out = ["", "    " + (title or "分段生成间隔（相邻两个打点之间平均每个 token 的毫秒数；只算两端都生成到了的请求）"),
           f"    {'区间（第几个 token）':<18}{'条数':>6}" + "".join(
               f"{h:>10}" for h in ("均值", "标准差", "p50", "p90", "p99", "最大"))]
    for name in ss["segments"]:
        v = ss["all"][name]
        if not v.get("n"):
            out.append(f"    {name:<18}{0:>6}" + f"{'—':>10}" * 6)
            continue
        out.append(f"    {name:<18}{v['n']:>6}" + "".join(
            f"{v[k]:>10.1f}" for k in ("mean", "std", "p50", "p90", "p99", "max")))
    out.append(f"    {'按 task：均值 ms（条数）':<24}" + "".join(f"{s_:>14}" for s_ in ss["segments"]))
    for u, by in sorted(ss["by_task"].items()):
        cells = [f"{by[n]['mean']:.1f}（{by[n]['n']}）" if by.get(n, {}).get("n") else "—"
                 for n in ss["segments"]]
        out.append(f"    {u:<24}" + "".join(f"{c:>14}" for c in cells))
    out.append("    " + (note or "1→64 = (第 64 个 − 首 token) / 63；其余 = (第 k₂ 个 − 第 k₁ 个) / (k₂ − k₁)，"
                              "k 用打点时实际已生成的 token 数"))
    return out


def format_mark_stats(ms: dict) -> list[str]:
    if not ms.get("all"):
        return []

    def f(sv, d):
        return f"{sv['mean']:.{d}f} ± {sv['std']:.{d}f}" if sv.get("n") else "—"
    out = ["", "    生成长度打点（每条请求生成到第 k 个 token 时；只算生成到了 k 的请求；均值 ± 标准差）",
           f"    {'k':>6}{'条数':>6}{'提交→第k个 ms':>20}{'首token→第k个 ms':>22}{'平均TPOT ms':>16}"
           f"{'单请求tok/s':>16}{'解码tok/s':>16}"]
    for k, v in ms["all"].items():
        out.append(f"    {k:>6}{v['n']:>6}{f(v['t_ms'], 0):>20}{f(v['since_first_ms'], 0):>22}"
                   f"{f(v['tpot_ms'], 1):>16}{f(v['tok_per_s'], 2):>16}{f(v['decode_tok_per_s'], 2):>16}")
    ks = list(ms["all"])
    out.append(f"    {'按 task：平均TPOT ms（条数）':<26}" + "".join(f"{'k=' + k:>16}" for k in ks))
    for u, by_k in sorted(ms["by_task"].items()):
        cells = []
        for k in ks:
            v = by_k.get(k)
            cells.append(f"{v['tpot_ms']['mean']:.1f}（{v['n']}）" if v and v["tpot_ms"].get("n") else "—")
        out.append(f"    {u:<26}" + "".join(f"{c:>16}" for c in cells))
    out.append("    提交→第k个 含排队与首 token；平均TPOT = (第k个 − 首token) / (k − 1)；单请求tok/s = k ÷ 提交→第k个；"
               "解码tok/s = 1000 ÷ 平均TPOT")
    return out


def continuous_report(doc: dict, title: str = "连续性测试") -> list[str]:
    """bench 的结果 JSON → 与 p2p-framework 连续性测试同一格式的几张表（文本行）。"""
    rows = list(doc.get("requests") or [])
    run = doc.get("run") or {}
    if not rows:
        return [f"{title}：没有请求"]
    # 墙钟：bench 记的；老结果没有时用最后完成的时刻，再没有就用最长的那条（只影响吞吐那一行）
    wall = (run.get("wall_s") or max(float(r.get("t_done_s") or 0) for r in rows)
            or max(float(r.get("total_ms") or 0) for r in rows) / 1e3 or 1.0)
    if run.get("offline") and title == "连续性测试":
        title = "离线满载（全部请求一次提交）"
    if run.get("saturate") and title == "连续性测试":
        title = "满载连续测试（逐步加压 → 保持满载 → 发完收尾，全部请求）"
    out = [f"{title}：{len(rows)} 条（超时 {run.get('n_timeout', 0)}），在途 {run.get('inflight', doc.get('concurrency'))}，"
           f"最大生成 {run.get('max_tokens', doc.get('tokens'))}，墙钟 {wall:.0f}s"]
    metrics = [
        ("生成时延 ms", [r["total_ms"] for r in rows], 0),
        ("TTFT ms", [r["prefill_ms"] for r in rows], 0),
        ("TTFT(去排队) ms", [r["prefill_ms"] - _queue(r) for r in rows], 0),
        ("TPOT ms/token", [_tpot(r) for r in rows], 1),
        ("后段排队 ms", [r.get("queue_back_ms", 0.0) for r in rows], 0),
        ("生成长度 token", [r.get("n_generated") for r in rows], 0),
        ("token/s(含排队)", [_tok_s(r) for r in rows], 2),
        ("token/s(去排队)", [_tok_s_service(r) for r in rows], 2),
        ("token/s(解码)", [_decode_tok_s(r) for r in rows], 2),
    ]
    out.append(f"    {'指标':<20}" + "".join(f"{h:>10}" for h in ("均值", "标准差", "p50", "p90", "p99", "最大")))
    for name, xs, d in metrics:
        sv = full_stats(xs)
        if not sv["n"]:
            out.append(f"    {name:<20}" + f"{'—':>10}" * 6)
            continue
        out.append(f"    {name:<20}" + "".join(f"{sv[k]:>10.{d}f}" for k in ("mean", "std", "p50", "p90", "p99", "max")))
    out.append("    单请求 token/s：每条请求自己的生成速度（生成 token ÷ 用时），再对全部请求取均值 / 标准差；")
    out.append("      含排队 = 提交→最后一个 token，去排队 = 拿到前后段→最后一个 token，解码 = 1000 ÷ TPOT")
    out_tok = sum(int(r.get("n_generated") or 0) for r in rows)
    in_tok = sum(int(r.get("n_prompt") or 0) for r in rows)
    stops = run.get("stop_reasons") or {}
    if not stops:
        for r in rows:
            stops[r.get("stop_reason")] = stops.get(r.get("stop_reason"), 0) + 1
    if run.get("window_output_tokens") is not None:
        wt = float(run["window_output_tokens"])
        out.append(f"    吞吐（满载窗口内生成的 token ÷ 窗口长度，含跨窗口请求落在窗口里的部分）：输出 "
                   f"{wt / wall:.2f} token/s；窗口内平均在途 {run.get('window_inflight_mean')}；"
                   f"这 {len(rows)} 条的停止原因 {dict(stops)}")
    else:
        out.append(f"    吞吐：输出 {out_tok / wall:.2f} token/s，含 prompt {(out_tok + in_tok) / wall:.2f} token/s，"
                   f"{len(rows) / wall * 60:.2f} 请求/分钟（输出共 {out_tok} token；停止原因 {dict(stops)}）")
    oks = [x for x in (_label_ok(r) for r in rows) if x is not None]
    if oks and doc.get("gateway_mode", "lr") not in ("oracle", "any"):
        out.append(f"    识别：{sum(oks)}/{len(oks)} = {100 * sum(oks) / len(oks):.1f}%（选择器自己判的类）；"
                   f"通道二报警 {sum(int(r.get('alarms', 0)) for r in rows)} 次，"
                   f"换绑 {sum(int(r.get('rebinds', 0)) for r in rows)} 次")
    # ---- 按 task ----
    by = defaultdict(list)
    for r in rows:
        by[r.get("true_task") or "?"].append(r)
    hd = ("条数", "时延均值", "时延p99", "TTFT均值", "TTFT std", "去排队TTFT", "TPOT均值", "TPOT std",
          "tok/s均值", "tok/s std", "后段排队", "生成均长", "识别", "换绑")
    out.append("")
    out.append(f"    {'task':<12}" + "".join(f"{h:>10}" for h in hd))
    for u, rs in sorted(by.items()):
        L, T = full_stats([r["total_ms"] for r in rs]), full_stats([r["prefill_ms"] for r in rs])
        TS = full_stats([r["prefill_ms"] - _queue(r) for r in rs])
        P, K = full_stats([_tpot(r) for r in rs]), full_stats([_tok_s_service(r) for r in rs])
        Q, G = full_stats([r.get("queue_back_ms", 0.0) for r in rs]), full_stats([r.get("n_generated") for r in rs])
        ok = [x for x in (_label_ok(r) for r in rs) if x is not None]

        def v(sv, k, d):
            return f"{sv[k]:>10.{d}f}" if sv.get("n") else f"{'—':>10}"
        out.append(f"    {u:<12}{len(rs):>10}" + v(L, "mean", 0) + v(L, "p99", 0) + v(T, "mean", 0)
                   + v(T, "std", 0) + v(TS, "mean", 0) + v(P, "mean", 1) + v(P, "std", 1)
                   + v(K, "mean", 2) + v(K, "std", 2) + v(Q, "mean", 0) + v(G, "mean", 0)
                   + (f"{100 * sum(ok) / len(ok):>9.0f}%" if ok else f"{'—':>10}")
                   + f"{sum(int(r.get('rebinds', 0)) for r in rs):>10}")
    out.append("    （tok/s = 单请求 token/s，去排队口径）")
    out += format_mark_stats(mark_stats(rows))
    out += format_segment_stats(segment_stats(rows))
    out += format_segment_stats(
        cumulative_stats(rows),
        title="累计生成间隔（首 token → 第 k 个 token，平均每个 token 的毫秒数；只算生成到了 k 的请求）",
        note="1→k = (第 k 个 − 首 token) / (k − 1)；1→2048 没有打点时用正好生成 2048 个的请求整条的 TPOT")
    # ---- 资源 ----
    usage = run.get("nodes") or {}
    if usage:
        seg = {}
        for name, c in (doc.get("channels") or {}).items():
            for n in c.get("nodes") or []:
                seg.setdefault(n, name)
        for r in rows:
            if r.get("front_engine") and r.get("front"):
                for n in str(r["front"]).split("+"):
                    seg.setdefault(n, r["front_engine"])
        out.append("")
        out.append("  资源占用（连续测试窗口内）")
        out.append(f"    {'节点':<6}{'段':<18}{'计算占用':>8}{'GPU均值%':>10}{'GPU p95%':>9}"
                   f"{'显存峰值/总量 MB':>20}{'torch峰值MB':>12}{'功耗W':>8}")
        gu, mu, mt = [], [], []
        for n in sorted(usage):
            u_ = usage[n] or {}
            if u_.get("error"):
                out.append(f"    {n:<6}{seg.get(n, '—'):<18}  （{u_['error']}）")
                continue
            if u_.get("gpu_util_mean") is not None:
                gu.append(u_["gpu_util_mean"])
            if u_.get("mem_used_max_mb"):
                mu.append(u_["mem_used_max_mb"])
            if u_.get("mem_total_mb"):
                mt.append(u_["mem_total_mb"])
            mem = f"{u_.get('mem_used_max_mb', 0):.0f}/{u_.get('mem_total_mb', 0):.0f}"
            out.append(f"    {n:<6}{seg.get(n, '—'):<18}{'—':>8}{u_.get('gpu_util_mean', 0):>10.1f}"
                       f"{u_.get('gpu_util_p95', 0):>9.0f}{mem:>20}{'—':>12}{u_.get('power_mean_w') or 0:>8.0f}")
        out.append(f"    合计/均值：GPU 利用率 {st.fmean(gu) if gu else 0:.1f}%，显存峰值合计 "
                   f"{sum(mu):.0f} / {sum(mt):.0f} MB")
        out.append("    计算占用、torch 峰值：vLLM 不报逐节点的 forward 时长与 PyTorch 显存，记「—」；"
                   "GPU 利用率 = nvidia-smi 采样（SM 有活的时间比）")
    if not doc.get("_no_summary_table"):
        out += format_summary_table(summary_table(doc))
    return out


# --------------------------------------------------------------------------- #
# 汇总表：每个指标一行；列 = 全部 / 各 task（该 task 的请求 + 它的后段流水线）/ 前段（共享）
# --------------------------------------------------------------------------- #
SUMMARY_ROWS = (
    ("e2e", "E2E (excl. queue) ms", "req", 1),
    ("ttft", "TTFT (excl. queue) ms", "req", 1),
    ("tpot", "TPOT ms/token", "req", 2),
    ("out_len", "Output length (tokens)", "req", 1),
    ("tok_s", "Per-request tok/s (excl. queue)", "req", 2),
    ("decode", "Decode tok/s (1000/TPOT)", "req", 2),
    ("thr_out", "Throughput: output tok/s", "sys", 2),
    ("thr_all", "Throughput: incl. prompt tok/s", "sys", 2),
    ("thr_req", "Throughput: requests/min", "sys", 2),
    ("tot_out", "Total output tokens", "sys", 0),
    ("gpu_util", "Mean GPU utilization %", "res", 1),
    ("mem_peak", "Total peak GPU memory (MB)", "res", 0),
    ("mem_total", "Total GPU memory (MB)", "res", 0),
    ("power", "Total power (W)", "res", 1),
)


def _e2e_service(r):
    """端到端（去排队）：提交 → 最后一个 token，减去网关路由与引擎里的排队。"""
    t = float(r.get("total_ms") or 0) - _queue(r)
    return t if t > 0 else None


def _ttft_service(r):
    t = float(r.get("prefill_ms") or 0) - _queue(r)
    return t if t >= 0 else None


def _t_first(r):
    return float(r["t_submit_s"]) + float(r.get("prefill_ms") or 0) / 1e3


def _window_tokens(rows, t_a: float, t_b: float) -> float:
    """[t_a, t_b] 里生成的 token 数（与 bench.window_tokens 同一口径：首 token 计在首 token 时刻，
    其余按首 token → 最后一个 token 均匀摊开）。"""
    tot = 0.0
    for r in rows:
        n = int(r.get("n_generated") or 0)
        if n <= 0 or r.get("t_submit_s") is None or r.get("t_done_s") is None:
            continue
        tf, td = _t_first(r), float(r["t_done_s"])
        if t_a <= tf <= t_b:
            tot += 1
        if n > 1 and td > tf:
            tot += (n - 1) * max(0.0, min(td, t_b) - max(tf, t_a)) / (td - tf)
    return tot


def path_groups(doc: dict) -> tuple[dict, list]:
    """→ ({task: [它的后段流水线上的节点]}, [前段节点])。后段的 task 优先取结果里 channels 的
    task 字段，没有就看分到这个后段的请求的真实 task（oracle 下就是它服务的 task）。"""
    rows = list(doc.get("requests") or [])
    ch_task: dict = {}
    for r in rows:
        if r.get("channel") and r.get("true_task"):
            ch_task.setdefault(r["channel"], defaultdict(int))[r["true_task"]] += 1
    by_task: dict = defaultdict(list)
    for name, c in (doc.get("channels") or {}).items():
        t = c.get("task") or (max(ch_task[name], key=ch_task[name].get) if name in ch_task else None)
        if t is None and str(name).startswith("back-"):
            t = str(name)[5:].rsplit("-", 1)[0]
        for v in c.get("nodes") or []:
            if v not in by_task[t or "?"]:
                by_task[t or "?"].append(v)
    fronts: list = []
    for r in rows:
        for v in str(r.get("front") or "").split("+"):
            if v and v not in fronts:
                fronts.append(v)
    # 分段网关下 channels 只有后段；A 的静态通道里前段节点也在 nodes 里 —— 从后段里去掉前段
    for t in list(by_task):
        by_task[t] = [v for v in by_task[t] if v not in fronts]
    return dict(by_task), sorted(fronts)


def summary_table(doc: dict, *, steady: bool = False) -> dict | None:
    """→ {"title", "columns": [列名], "nodes": {列: [节点]}, "rows": [{key, label, kind, values: {列: 值}}]}。

    每列：「全部」= 全部请求、全部节点；task 列 = 真实 task 为它的请求 + 它的后段流水线上的节点；
    「前段（共享）」只列资源（三个 task 共用，吞吐不单算）。
    逐请求指标（req）：值是 {mean, std, n}；吞吐（sys）：全程 = 合计 ÷ 墙钟；`steady=True`（--saturate 的
    满载段）= 满载窗口 [坐满, 第一个 task 发完] 里生成的 token（落在窗口里的部分）÷ 窗口长度，
    prompt token 按首 token 落在窗口里的请求算、请求数按窗口里完成的算；
    资源（res）：nvidia-smi 在整个压测期间的采样（GPU 利用率取节点均值，显存与功耗按节点求和）。"""
    rows_all = list(doc.get("requests") or [])
    run = doc.get("run") or {}
    if not rows_all:
        return None
    win = None
    if steady:
        sat = run.get("saturate") or {}
        ta, tb = sat.get("t_full_s"), sat.get("t_end_s")
        if ta is None or tb is None or tb <= ta:
            return None
        win = (float(ta), float(tb))
        req_rows = [r for r in rows_all if r.get("phase") == "steady"]
        wall = win[1] - win[0]
        title = f"满载段（第 {win[0]:.0f}s → {win[1]:.0f}s，{wall:.0f}s；逐请求指标只算段内提交并完成的 {len(req_rows)} 条）"
    else:
        req_rows = rows_all
        wall = float(run.get("wall_s") or max(float(r.get("t_done_s") or 0) for r in rows_all) or 1.0)
        title = f"全程（{len(rows_all)} 条，墙钟 {wall:.0f}s）"
    tasks = sorted({r.get("true_task") or "?" for r in rows_all})
    by_task, fronts = path_groups(doc)
    usage = run.get("nodes") or {}
    all_nodes = sorted(set(fronts) | {v for vs in by_task.values() for v in vs} | set(usage))
    cols = ["全部"] + tasks + (["前段（共享）"] if fronts else [])
    nodes = {"全部": all_nodes, **{t: by_task.get(t, []) for t in tasks}}
    if fronts:
        nodes["前段（共享）"] = fronts

    def req_of(col):
        return req_rows if col == "全部" else [r for r in req_rows if (r.get("true_task") or "?") == col]

    def all_of(col):
        return rows_all if col == "全部" else [r for r in rows_all if (r.get("true_task") or "?") == col]

    def sys_vals(col):
        if col == "前段（共享）":
            return {}
        if win is None:
            rs = all_of(col)
            out_t = sum(int(r.get("n_generated") or 0) for r in rs)
            in_t = sum(int(r.get("n_prompt") or 0) for r in rs)
            n_req = len(rs)
        else:
            rs = all_of(col)
            out_t = _window_tokens(rs, *win)
            in_t = sum(int(r.get("n_prompt") or 0) for r in rs
                       if r.get("t_submit_s") is not None and win[0] <= _t_first(r) <= win[1])
            n_req = sum(1 for r in rs if r.get("t_done_s") is not None
                        and win[0] <= float(r["t_done_s"]) <= win[1])
        return {"thr_out": out_t / wall, "thr_all": (out_t + in_t) / wall,
                "thr_req": n_req / wall * 60.0, "tot_out": out_t}

    def res_vals(col):
        us = [usage[v] for v in nodes.get(col, []) if isinstance(usage.get(v), dict)
              and not usage[v].get("error")]
        if not us:
            return {}
        gu = [u["gpu_util_mean"] for u in us if u.get("gpu_util_mean") is not None]
        return {"gpu_util": st.fmean(gu) if gu else None,
                "mem_peak": sum(float(u.get("mem_used_max_mb") or 0) for u in us),
                "mem_total": sum(float(u.get("mem_total_mb") or 0) for u in us),
                "power": sum(float(u.get("power_mean_w") or 0) for u in us)}

    req_fn = {"e2e": _e2e_service, "ttft": _ttft_service, "tpot": _tpot,
              "out_len": lambda r: r.get("n_generated"), "tok_s": _tok_s_service, "decode": _decode_tok_s}
    table = {k: {} for k, *_ in SUMMARY_ROWS}
    for col in cols:
        sv, rv = sys_vals(col), res_vals(col)
        for key, _, kind, _ in SUMMARY_ROWS:
            if kind == "req":
                if col == "前段（共享）":
                    table[key][col] = None
                else:
                    fs = full_stats([req_fn[key](r) for r in req_of(col)])
                    table[key][col] = ({"mean": fs["mean"], "std": fs["std"], "n": fs["n"]}
                                       if fs["n"] else None)
            elif kind == "sys":
                table[key][col] = sv.get(key)
            else:
                table[key][col] = rv.get(key)
    return {"title": title, "steady": bool(steady), "columns": cols, "nodes": nodes,
            "rows": [{"key": k, "label": lab, "kind": kind, "values": table[k]}
                     for k, lab, kind, _ in SUMMARY_ROWS]}


def _fmt_cell(v, digits: int) -> str:
    if v is None:
        return "—"
    if isinstance(v, dict):
        return f"{v['mean']:.{digits}f} ± {v['std']:.{digits}f}"
    return f"{v:.{digits}f}"


def format_summary_table(t: dict | None) -> list[str]:
    if not t:
        return []
    digits = {k: d for k, _, _, d in SUMMARY_ROWS}
    cols = t["columns"]
    w = max(18, max(len(_fmt_cell(r["values"].get(c), digits[r["key"]])) for r in t["rows"] for c in cols) + 2)
    out = ["", f"  汇总表 · {t['title']}",
           f"    {'':<34}" + "".join(f"{c:>{w}}" for c in cols)]
    for r in t["rows"]:
        out.append(f"    {r['label']:<34}" + "".join(
            f"{_fmt_cell(r['values'].get(c), digits[r['key']]):>{w}}" for c in cols))
    out.append("    路径节点：" + "；".join(f"{c} {'+'.join(t['nodes'].get(c) or []) or '—'}"
                                       for c in cols if c != "全部"))
    out.append("    逐请求指标 = 均值 ± 标准差（去排队 = 减去网关路由与引擎排队）；task 列的资源只算它的后段流水线，"
               "前段三个 task 共用、单列；GPU 利用率为节点均值，显存与功耗为节点合计（nvidia-smi，整个压测期间）")
    return out


def summary_csv(tables) -> str:
    """一个或多个汇总表 → CSV 文本（贴进表格用）：每张表一段，行 = 指标，列 = 全部 / 各 task / 前段；
    逐请求指标给均值，下面另起一行给标准差。"""
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    digits = {k: d for k, _, _, d in SUMMARY_ROWS}
    for t in tables:
        if not t:
            continue
        w.writerow([t["title"]] + t["columns"])
        for r in t["rows"]:
            vals = [r["values"].get(c) for c in t["columns"]]
            w.writerow([r["label"]] + ["" if v is None else round(v["mean"] if isinstance(v, dict) else v,
                                                                   digits[r["key"]] + 1) for v in vals])
            if r["kind"] == "req":
                w.writerow([f"{r['label']} std"] + ["" if not isinstance(v, dict) else
                                                    round(v["std"], digits[r["key"]] + 1) for v in vals])
        w.writerow(["路径节点"] + ["+".join(t["nodes"].get(c) or []) for c in t["columns"]])
        w.writerow([])
    return buf.getvalue()


def saturate_report(doc: dict) -> list[str]:
    """--saturate 的结果：加压过程、满载段长度、满载段吞吐，再加一张只算满载段请求的表。"""
    run = doc.get("run") or {}
    sat = run.get("saturate")
    if not sat:
        return []
    rows = list(doc.get("requests") or [])
    caps = ", ".join(f"{'全部' if k == 'None' else k} {c}（{sat['backs'].get(k)} 个后段）"
                     for k, c in sat["caps"].items())
    ta, tb = sat.get("t_full_s"), sat.get("t_end_s")
    out = [f"满载过程：按 {'task' if sat.get('by') == 'task' else '全局'} 保持，满载在途 {sat['target']}"
           f"（{caps}）；从每个后段 1 条起、每 {sat.get('ramp_every'):g}s 每个后段 +{sat.get('ramp_step')}"]
    if ta is None:
        out.append("    一直没坐满（请求不够或加压太慢），没有满载段")
        return out
    out.append(f"    第 {ta:.0f}s 坐满 → 第 {tb:.0f}s 第一个 {'task ' if sat.get('by') == 'task' else ''}"
               f"发完最后一条：满载段 {sat['window_s']:.0f}s，段内平均在途 {sat.get('window_inflight_mean')}，"
               f"段内输出 {sat.get('window_tok_per_s') or 0:.2f} token/s")
    ph = sat.get("n_phase") or {}
    out.append(f"    请求分段：加压中提交 {ph.get('ramp', 0)} 条、满载段内提交并完成 {ph.get('steady', 0)} 条、"
               f"满载段内提交但收尾时才完成 {ph.get('tail', 0)} 条")
    cv = sat.get("curve") or []
    if cv:
        stride = max(1, len(cv) // 16)
        out.append("    在途曲线（秒:在途/上限）：" + "  ".join(
            f"{t:.0f}:{n}/{lim}" for t, n, lim in cv[::stride]))
    steady = [r for r in rows if r.get("phase") == "steady"]
    if steady:
        sub = dict(doc)
        sub["requests"] = steady
        sub["run"] = {**(doc.get("steady") or {}), "wall_s": sat["window_s"],
                      "window_output_tokens": sat.get("window_output_tokens"),
                      "window_inflight_mean": sat.get("window_inflight_mean"),
                      "inflight": sat["target"], "max_tokens": run.get("max_tokens")}
        sub["run"].pop("nodes", None)
        sub["_no_summary_table"] = True
        out.append("")
        out += continuous_report(sub, title=f"满载段（坐满之后提交、第一个发完之前完成的 {len(steady)} 条）")
    out += format_summary_table(summary_table(doc, steady=True))
    return out


def format_request_stats(rs: dict) -> list[str]:
    def f(d, digits=1):
        if not d or d.get("mean") is None:
            return "—"
        return f"{d['mean']:.{digits}f} ± {d['std']:.{digits}f}"
    head = f"  {'':<12}{'条数':>5}{'TTFT ms':>20}{'TPOT ms':>18}{'token/s':>16}{'decode token/s':>18}"
    out = ["逐请求统计（均值 ± 标准差）", head]
    rows = [("全部", rs["all"])] + [(u, v) for u, v in rs["by_task"].items()]
    for name, v in rows:
        out.append(f"  {name:<12}{v['n']:>5}{f(v['ttft_ms'], 0):>20}{f(v['tpot_ms']):>18}"
                   f"{f(v['tok_per_s']):>16}{f(v['decode_tok_per_s']):>18}")
    out.append("  （token/s = 每条请求 生成 token / 总时延，含首 token；decode token/s = 1000 / TPOT；"
               "都是逐条算再对所有请求取均值）")
    return out


def summarise_b(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r.get("front") or "?", r.get("channel") or "?")].append(r)
    out = []
    for (f, b), rs in sorted(g.items()):
        eng = [r.get("engine") or {} for r in rs]
        out.append({"front": f, "back": b, "n": len(rs),
                    "ttft": _med([r["prefill_ms"] for r in rs]),
                    "tpot": _med([_tpot(r) for r in rs]),
                    "front_prefill": _med([e.get("front_prefill_ms") for e in eng]),
                    "bind": _med([e.get("bind_ms") for e in eng]),
                    "front_leg": _med([e.get("front_leg_ms_p50") for e in eng]),
                    "back_leg": _med([e.get("back_leg_ms_p50") for e in eng]),
                    "front_eng": _med([e.get("front_engine_ms_p50") for e in eng]),
                    "back_eng": _med([e.get("back_engine_ms_p50") for e in eng])})
    return out


def summarise_a(rows):
    g = defaultdict(list)
    for r in rows:
        g[r.get("channel") or "?"].append(r)
    return {c: {"n": len(rs), "ttft": _med([r["prefill_ms"] for r in rs]),
                "tpot": _med([_tpot(r) for r in rs])} for c, rs in g.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--b", required=True, type=Path, help="分段网关的结果")
    ap.add_argument("--a", type=Path, default=None, help="A 的结果（同一批 case）")
    ap.add_argument("--csv", type=Path, default=None,
                    help="把汇总表（全程；--saturate 的结果再加满载段）写成 CSV，贴进表格用")
    a = ap.parse_args(argv)
    rb = json.loads(a.b.read_text(encoding="utf-8"))["requests"]
    ra = json.loads(a.a.read_text(encoding="utf-8"))["requests"] if a.a else []
    A = summarise_a(ra)
    db = json.loads(a.b.read_text(encoding="utf-8"))
    for ln in continuous_report(db, title=f"连续性测试（{a.b.name}）"):
        print(ln)
    if (db.get("run") or {}).get("saturate"):
        print()
        for ln in saturate_report(db):
            print(ln)
    if a.csv:
        a.csv.write_text(summary_csv([summary_table(db), summary_table(db, steady=True)]),
                         encoding="utf-8-sig")
        print(f"\n汇总表已写 {a.csv}")
    if a.a:
        print()
        for ln in continuous_report(json.loads(a.a.read_text(encoding="utf-8")),
                                    title=f"对照（{a.a.name}）"):
            print(ln)
    print()

    def f(x):
        return "—" if x is None else f"{x:g}"
    print(f"{'前段':<8}{'后段':<16}{'条数':>4}{'TTFT':>8}{'TPOT':>8}{'前段prefill':>12}{'绑定':>7}"
          f"{'前段腿':>8}{'(引擎内)':>9}{'后段腿':>8}{'(引擎内)':>9}   {'A 同通道 TPOT / TTFT':>22}")
    for s in summarise_b(rb):
        ac = A.get(s["back"].removeprefix("back-"))
        tail = f"{f(ac['tpot'])} / {f(ac['ttft'])}（{ac['n']} 条）" if ac else ""
        print(f"{s['front']:<8}{s['back']:<16}{s['n']:>4}{f(s['ttft']):>8}{f(s['tpot']):>8}"
              f"{f(s['front_prefill']):>12}{f(s['bind']):>7}{f(s['front_leg']):>8}"
              f"{f(s['front_eng']):>9}{f(s['back_leg']):>8}{f(s['back_eng']):>9}   {tail:>22}")
    print("（单位 ms；TPOT = (总时延 − 首 token) / (生成数 − 1)；腿是每条请求的中位数再取中位数；\n"
          " 「引擎内」= 代理收到这条请求的输入到送出它的输出（排队 + 调度 + 前向），腿 − 引擎内 ≈ 网络 + 协调器）")
    ident = summarise_ident(rb)
    if ident:
        print(f"\n前段出口识别（lr 模式）：绑对 task {ident['ok']}/{ident['n']}"
              f"（{100 * ident['ok'] / ident['n']:.1f}%）；LR 耗时中位数 {f(ident['lr_ms'])} ms；"
              f"区间 {ident['zones']}")
        tasks = sorted({t for pair in ident["matrix"] for t in pair})
        head = "真实 \\ 绑到"
        print(f"  {head:<14}" + "".join(f"{t:>12}" for t in tasks))
        for u in tasks:
            if any(k[0] == u for k in ident["matrix"]):
                print(f"  {u:<14}" + "".join(f"{ident['matrix'].get((u, t), 0) or '·':>12}"
                                             for t in tasks))
        if ident["wrong"]:
            print(f"  绑错的（前 10 条）：{ident['wrong'][:10]}")
    return 0


def summarise_ident(rows):
    """lr 模式的结果：真实 task × 绑到的 task、准确率、LR 耗时、置信区间分布。"""
    rs = [r for r in rows if r.get("correct") is not None and r.get("true_task")]
    if not rs:
        return None
    m = defaultdict(int)
    zones = defaultdict(int)
    for r in rs:
        m[(r["true_task"], r.get("task") or "?")] += 1
        zones[r.get("zone") or "?"] += 1
    return {"n": len(rs), "ok": sum(bool(r["correct"]) for r in rs), "matrix": dict(m),
            "zones": dict(zones),
            "lr_ms": _med([(r.get("identify") or {}).get("compute_ms") for r in rs]),
            "wrong": [f"{r['req']}:{r['true_task']}→{r.get('task')}" for r in rs
                      if not r["correct"]]}


if __name__ == "__main__":
    raise SystemExit(main())
