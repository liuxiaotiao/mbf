# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm/p2p/colorplan.py：moe_color_planner 的可部署方案 → stages.json。纯 Python，不要 GPU。"""

import importlib.util
import json
import shlex
import sys
from pathlib import Path

import pytest

P2P = Path(__file__).resolve().parents[2] / "vllm" / "p2p"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", P2P / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cp = _load("colorplan")
lay = _load("layout")

L, L0, E = 48, 8, 128


def _experts():
    # 每个 task 每层驻留 0..(63+层号%5)，第 9 层全装（检验“装全的层不列”）
    def sets(t):
        out = []
        for l1 in range(1, L + 1):
            out.append(list(range(E)) if l1 == 9 else list(range(64 + (l1 + t) % 5)))
        return out
    return {"tasks": {"gsm8k": {"union": sets(0)}, "mbpp": {"union": sets(1)},
                      "no_robots": {"union": sets(2)}}}


def _result():
    return {
        "summary": {"theta": 1000.0},
        "colors": {k: {"tpot_ms": 240.0} for k in ("gsm8k", "mbpp", "no_robots")},
        "deploy": {
            "via_coordinator": False, "coordinator": "N09", "l0": L0, "problems": [],
            "fronts": [{"node": "N09", "tokens_per_s": 1500.0, "avg_cycle_ms": 240.0},
                       {"node": "N10", "tokens_per_s": 1500.0, "avg_cycle_ms": 240.0}],
            "backs": [
                {"task": "gsm8k", "nodes": ["N01", "N02"], "layers": [[9, 28], [29, 48]],
                 "tokens_per_s": 1000.0, "share_of_task": 1.0, "avg_cycle_ms": 240.0},
                {"task": "mbpp", "nodes": ["N03", "N04", "N05"], "layers": [[9, 20], [21, 34], [35, 48]],
                 "tokens_per_s": 600.0, "share_of_task": 0.6, "avg_cycle_ms": 235.0},
                {"task": "mbpp", "nodes": ["N06", "N07"], "layers": [[9, 30], [31, 48]],
                 "tokens_per_s": 400.0, "share_of_task": 0.4, "avg_cycle_ms": 245.0},
                {"task": "no_robots", "nodes": ["N08", "N11"], "layers": [[9, 25], [26, 48]],
                 "tokens_per_s": 1000.0, "share_of_task": 1.0, "avg_cycle_ms": 240.0},
            ],
        },
    }


def _hosts(tmp_path):
    p = tmp_path / "hosts.txt"
    p.write_text("".join(f"N{i:02d}  10.0.0.{i}:9101   # A30 24GB\n" for i in range(1, 16)))
    return lay.parse_hosts(p)


def _build(tmp_path, res=None, **kw):
    inst = {"L": L, "block_layers": 1, "batch": 64,
            "gpus": [{"id": f"N{i:02d}", "type": "A30"} for i in range(1, 16)],
            "gpu_types": {"A30": {"d_layer_ms": 3.6, "d_ms": 3.6}}}
    return cp.stages_from_color_plan(res or _result(), inst, _experts(), _hosts(tmp_path),
                                     model_dir="/w", python="/venv/bin/python", **kw)


def _overrides(entry, rank=0):
    toks = shlex.split(entry["commands"][rank]["cmd"])
    return json.loads(toks[toks.index("--hf-overrides") + 1])


def test_fronts_and_backs_not_paired_one_to_one(tmp_path):
    sj, _ = _build(tmp_path)
    assert sj["kind"] == "tiered" and sj["l0"] == L0 and sj["n_layers"] == L
    assert sj["fronts"] == ["front-N09", "front-N10"]
    assert sj["backs"] == ["back-gsm8k-0", "back-mbpp-0", "back-mbpp-1", "back-no_robots-0"]
    names = [c["name"] for c in sj["channels"]]
    assert names == sj["fronts"] + sj["backs"]


def test_back_partition_layers_and_experts(tmp_path):
    sj, _ = _build(tmp_path)
    b = next(c for c in sj["channels"] if c["name"] == "back-mbpp-0")
    assert b["nodes"] == ["N03", "N04", "N05"] and b["partition"] == [12, 14, 14]
    assert b["layers"] == [L0, L]
    cmd0 = b["commands"][0]["cmd"]
    assert cmd0.startswith("VLLM_PP_LAYER_PARTITION=12,14,14 ")
    assert "--layers 8,48" in cmd0 and "--node-rank 0" in cmd0 and "--master-addr 10.0.0.3" in cmd0
    assert "--node-rank 2" in b["commands"][2]["cmd"]
    ov = _overrides(b)
    le = ov["p2p_layer_experts"]
    # 0-based；第 9 层（0-based 8）装全 → 不列；前段层不列
    assert "8" not in le and "7" not in le and set(le) == {str(l) for l in range(9, L)}
    assert le["20"] == list(range(64 + (21 + 1) % 5))
    # 三个节点的命令除 node_rank 外一致（stage_agent 要求逐字一致）
    strip = [c["cmd"].replace(f"--node-rank {r}", "") for r, c in enumerate(b["commands"])]
    assert len(set(strip)) == 1


def test_front_engine(tmp_path):
    sj, _ = _build(tmp_path)
    f = sj["channels"][0]
    assert f["role"] == "front" and f["nodes"] == ["N09"] and f["layers"] == [0, L0]
    assert f["zmq"] == "tcp://10.0.0.9:8301"
    ov = _overrides(f)
    assert ov["p2p_layer_experts"] == {} and ov["p2p_stats"]["lr_layers"] == list(range(L0))
    sj2, _ = _build(tmp_path, front_lr=False)
    assert "p2p_stats" not in _overrides(sj2["channels"][0])


def test_max_num_seqs_from_plan(tmp_path):
    sj, notes = _build(tmp_path)
    f = sj["channels"][0]            # 1500 tokens/s × 0.24 s = 360 在飞 → ×1.3 → 472
    assert f["max_num_seqs"] == 472
    # 每步批量 = 速率 × 本卡一步耗时：1500 × 8 层 × 3.6 ms ≈ 43（在 64 附近，不提示）
    assert f["plan"]["step_batch_est"]["N09"] == pytest.approx(43.2)
    assert not any("front-N09" in s for s in notes)
    sj2, _ = _build(tmp_path, front_max_num_seqs=64, back_max_num_seqs=32)
    assert {c["max_num_seqs"] for c in sj2["channels"]} == {64, 32}


def test_rejects_bad_plans(tmp_path):
    with pytest.raises(cp.ColorPlanError, match="deploy"):
        _build(tmp_path, res={"summary": {}, "colors": {}})
    res = _result()
    res["deploy"]["backs"][0]["layers"] = [[9, 27], [29, 48]]
    with pytest.raises(cp.ColorPlanError, match="上一级"):
        _build(tmp_path, res=res)
    res = _result()
    res["deploy"]["problems"] = ["N01 出现在两条后段流水线里"]
    with pytest.raises(cp.ColorPlanError, match="不能部署"):
        _build(tmp_path, res=res)
    res = _result()
    res["deploy"]["fronts"][0]["node"] = "N99"
    with pytest.raises(cp.ColorPlanError, match="hosts"):
        _build(tmp_path, res=res)


def test_front_lr_config_goes_to_every_front(tmp_path):
    sj, _ = _build(tmp_path, lr={"model": "/sel/selector_body_L8_params.npz"})
    for name in sj["fronts"]:
        f = next(c for c in sj["channels"] if c["name"] == name)
        lr = _overrides(f)["p2p_lr"]
        assert lr["model"].endswith("L8_params.npz") and lr["tau_lo"] == 0.4
        assert lr["tasks"] == {"gsm8k": 1.0, "mbpp": 1.0, "no_robots": 1.0}
        assert _overrides(f)["p2p_stats"]["emit_lr"] is True
    for name in sj["backs"]:                       # 后段不带
        b = next(c for c in sj["channels"] if c["name"] == name)
        assert "p2p_lr" not in _overrides(b)
    cp_ = sj["color_plan"]
    assert cp_["lr_at"] == "front" and "--lr-at front" in cp_["gateway_cmd"]
    assert "--via-coordinator" not in cp_["gateway_cmd"]


def test_front_lr_rejects_tasks_without_backs_and_no_stats(tmp_path):
    with pytest.raises(cp.ColorPlanError, match="没有后段"):
        _build(tmp_path, lr={"model": "/s.npz", "tasks": {"gsm8k": 1, "other": 1}})
    with pytest.raises(cp.ColorPlanError, match="no-front-lr"):
        _build(tmp_path, lr={"model": "/s.npz"}, front_lr=False)


def test_via_coordinator_plan_tells_gateway(tmp_path):
    res = _result()
    res["deploy"]["via_coordinator"] = True
    sj, notes = _build(tmp_path, res=res)
    assert "--via-coordinator" in sj["color_plan"]["gateway_cmd"]
    assert any("--via-coordinator" in s and "N09" in s for s in notes)
    sj2, notes2 = _build(tmp_path)                 # 默认直连：不提协调器
    assert sj2["color_plan"]["coordinator"] is None and not any("协调器" in s for s in notes2)


def test_step_batch_far_from_plan_batch_is_flagged(tmp_path):
    res = _result()
    res["deploy"]["backs"][0]["tokens_per_s"] = 5000.0     # 5000 × 20 层 × 3.6 ms = 360 条/步
    _, notes = _build(tmp_path, res=res)
    assert any("back-gsm8k-0" in s and "N01" in s and "偏大" in s for s in notes)
