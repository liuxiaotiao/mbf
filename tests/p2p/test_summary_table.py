# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tiered_stats 的汇总表：逐请求指标（去排队）、吞吐、资源，按 task 的路径分列。纯标准库。"""

import csv
import importlib.util
import io
import sys
from pathlib import Path

import pytest

P2P = Path(__file__).resolve().parents[2] / "vllm" / "p2p"
spec = importlib.util.spec_from_file_location("_t_tiered_stats", P2P / "tiered_stats.py")
T = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = T
spec.loader.exec_module(T)


def _row(task, ch, front, *, sub, total, ttft, n, n_prompt=100, q=0.0, qb=0.0, phase=None):
    r = {"true_task": task, "channel": ch, "front": front, "n_generated": n, "n_prompt": n_prompt,
         "total_ms": total, "prefill_ms": ttft, "queue_ms": q, "queue_back_ms": qb,
         "t_submit_s": sub, "t_done_s": sub + total / 1e3}
    if phase:
        r["phase"] = phase
    return r


def _doc():
    rows = [
        # gsm8k：两条，走 front N03 / N04；第二条在引擎里排队 100 ms
        _row("gsm8k", "back-gsm8k-0", "N03", sub=0.0, total=1100, ttft=100, n=11),
        _row("gsm8k", "back-gsm8k-0", "N04", sub=1.0, total=2200, ttft=300, n=21, qb=100),
        # mbpp：一条
        _row("mbpp", "back-mbpp-0", "N03", sub=0.0, total=3000, ttft=200, n=15, n_prompt=50),
    ]
    usage = {"N03": {"gpu_util_mean": 80.0, "mem_used_max_mb": 20000, "mem_total_mb": 24000, "power_mean_w": 150},
             "N04": {"gpu_util_mean": 60.0, "mem_used_max_mb": 18000, "mem_total_mb": 24000, "power_mean_w": 120},
             "N11": {"gpu_util_mean": 50.0, "mem_used_max_mb": 15000, "mem_total_mb": 24000, "power_mean_w": 100},
             "N09": {"gpu_util_mean": 70.0, "mem_used_max_mb": 16000, "mem_total_mb": 24000, "power_mean_w": 110},
             "N05": {"gpu_util_mean": 90.0, "mem_used_max_mb": 21000, "mem_total_mb": 24000, "power_mean_w": 160}}
    return {"requests": rows,
            "channels": {"back-gsm8k-0": {"nodes": ["N11", "N09"]},          # 没写 task：从请求推
                         "back-mbpp-0": {"nodes": ["N05"], "task": "mbpp"}},
            "run": {"wall_s": 4.0, "nodes": usage}}


def _vals(t, key):
    return next(r["values"] for r in t["rows"] if r["key"] == key)


def test_columns_and_path_nodes():
    t = T.summary_table(_doc())
    assert t["columns"] == ["全部", "gsm8k", "mbpp", "前段（共享）"]
    assert t["nodes"]["gsm8k"] == ["N11", "N09"] and t["nodes"]["mbpp"] == ["N05"]
    assert t["nodes"]["前段（共享）"] == ["N03", "N04"]
    assert t["nodes"]["全部"] == ["N03", "N04", "N05", "N09", "N11"]
    assert [r["label"] for r in t["rows"]][:3] == ["E2E (excl. queue) ms", "TTFT (excl. queue) ms",
                                                  "TPOT ms/token"]


def test_per_request_metrics_exclude_queue():
    t = T.summary_table(_doc())
    e2e = _vals(t, "e2e")["gsm8k"]                 # 1100 与 2200 − 100
    assert e2e["mean"] == pytest.approx(1600) and e2e["n"] == 2
    ttft = _vals(t, "ttft")["gsm8k"]               # 100 与 300 − 100
    assert ttft["mean"] == pytest.approx(150)
    tpot = _vals(t, "tpot")["gsm8k"]               # (1100−100)/10 = 100；(2200−300)/20 = 95
    assert tpot["mean"] == pytest.approx(97.5)
    dec = _vals(t, "decode")["gsm8k"]
    assert dec["mean"] == pytest.approx((10 + 1000 / 95) / 2)
    tok = _vals(t, "tok_s")["gsm8k"]               # 11 / 1.1 s，21 / 2.1 s
    assert tok["mean"] == pytest.approx(10.0)
    assert _vals(t, "out_len")["mbpp"]["mean"] == 15
    assert _vals(t, "e2e")["前段（共享）"] is None


def test_throughput_and_resources():
    t = T.summary_table(_doc())
    assert _vals(t, "tot_out") == {"全部": 47, "gsm8k": 32, "mbpp": 15, "前段（共享）": None}
    assert _vals(t, "thr_out")["gsm8k"] == pytest.approx(32 / 4)
    assert _vals(t, "thr_all")["全部"] == pytest.approx((47 + 250) / 4)
    assert _vals(t, "thr_req")["mbpp"] == pytest.approx(1 / 4 * 60)
    # 资源：task 列只算自己的后段；前段单列；全部 = 所有节点
    assert _vals(t, "power") == {"全部": 640, "gsm8k": 210, "mbpp": 160, "前段（共享）": 270}
    assert _vals(t, "gpu_util")["gsm8k"] == pytest.approx(60.0)
    assert _vals(t, "mem_peak")["前段（共享）"] == 38000
    assert _vals(t, "mem_total")["全部"] == 5 * 24000


def test_steady_window():
    d = _doc()
    d["run"]["saturate"] = {"t_full_s": 1.0, "t_end_s": 3.0}
    for r, ph in zip(d["requests"], ("ramp", "steady", "tail")):
        r["phase"] = ph
    t = T.summary_table(d, steady=True)
    assert "满载段" in t["title"]
    # 逐请求只算 steady 的那条（gsm8k 第二条）
    assert _vals(t, "e2e")["gsm8k"]["n"] == 1 and _vals(t, "e2e")["mbpp"] is None
    # 窗口 [1, 3] 里的 token：第一条 gsm8k 首 token 0.1s、完 1.1s → 10 个摊在 1.0s 里，落 0.1s → 1 个
    # 第二条首 token 1.3s（计 1）、完 3.2s → 20 个摊 1.9s，落 1.7s；mbpp 首 0.2s、完 3.0s → 14 个摊 2.8s，落 2s
    g = 1 + 1 + 20 * 1.7 / 1.9
    m = 14 * 2.0 / 2.8
    assert _vals(t, "tot_out")["gsm8k"] == pytest.approx(g)
    assert _vals(t, "thr_out")["全部"] == pytest.approx((g + m) / 2.0)
    # prompt：首 token 落在窗口里的只有第二条 gsm8k；完成在窗口里的：第一条之后没有、第二条 3.2s 不在、mbpp 3.0s 在
    assert _vals(t, "thr_all")["gsm8k"] == pytest.approx((g + 100) / 2.0)
    assert _vals(t, "thr_req")["mbpp"] == pytest.approx(1 / 2.0 * 60)
    assert T.summary_table({"requests": d["requests"], "run": {}}, steady=True) is None


def test_report_and_csv():
    d = _doc()
    lines = T.continuous_report(d)
    i = next(k for k, ln in enumerate(lines) if "汇总表" in ln)
    assert "全部" in lines[i + 1] and "E2E (excl. queue) ms" in lines[i + 2]
    assert any("前段（共享） N03+N04" in ln for ln in lines)
    text = T.summary_csv([T.summary_table(d), None])
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0][1:] == ["全部", "gsm8k", "mbpp", "前段（共享）"]
    lab = {r[0]: r for r in rows if r}
    assert lab["Total power (W)"][1:] == ["640.0", "210.0", "160.0", "270.0"]
    assert lab["E2E (excl. queue) ms std"][2] != "" and lab["E2E (excl. queue) ms std"][4] == ""
