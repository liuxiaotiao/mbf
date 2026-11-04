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


def test_code_dir_prefixes_pythonpath(tmp_path):
    sj, _ = _build(tmp_path, code_dir="/home/ubuntu/mbf/")
    for c in sj["channels"]:
        for stc in c["commands"]:
            assert stc["cmd"].startswith("PYTHONPATH=/home/ubuntu/mbf ")
            toks = shlex.split(stc["cmd"].split(" --hf-overrides ")[0])
            assert "/venv/bin/python" in toks                  # launch.py 仍能认出 venv
    b = next(c for c in sj["channels"] if c["role"] == "back")
    assert "VLLM_PP_LAYER_PARTITION=" in b["commands"][0]["cmd"]
    assert sj["color_plan"]["code_dir"] == "/home/ubuntu/mbf/"
    sj2, _ = _build(tmp_path)
    assert not sj2["channels"][0]["commands"][0]["cmd"].startswith("PYTHONPATH=")


# --------------------------------------------------------------------------- #
# 多级（moe_color_planner.py --deploy --interleave）：engines + routes → 中段 / 后段 / 路线
# --------------------------------------------------------------------------- #
def _multi_result():
    """gsm8k：N01 切成两段（N01[9-20] 中段、N01[21-48] 后段），N02[9-20] 从 N01 的第 21 层进来；
    mbpp：一条两级 PP 的后段 N03+N04；no_robots：单卡后段 N05。"""
    res = _result()
    dep = res["deploy"]
    dep.update(interleave=True, backs=[])
    dep["engines"] = [
        {"name": "back-gsm8k-0", "role": "back", "task": "gsm8k", "nodes": ["N01"],
         "layers": [[21, 48]], "weights_gb": [12.0], "kv_gb": [3.0], "tokens_per_s": 1000.0, "routes": 2},
        {"name": "mid-gsm8k-0", "role": "mid", "task": "gsm8k", "nodes": ["N01"],
         "layers": [[9, 20]], "weights_gb": [5.0], "kv_gb": [1.0], "tokens_per_s": 600.0, "routes": 1},
        {"name": "mid-gsm8k-1", "role": "mid", "task": "gsm8k", "nodes": ["N02"],
         "layers": [[9, 20]], "weights_gb": [5.0], "kv_gb": [1.0], "tokens_per_s": 400.0, "routes": 1},
        {"name": "back-mbpp-0", "role": "back", "task": "mbpp", "nodes": ["N03", "N04"],
         "layers": [[9, 28], [29, 48]], "weights_gb": [9.0, 9.0], "kv_gb": [2.0, 2.0],
         "tokens_per_s": 1000.0, "routes": 1},
        {"name": "back-no_robots-0", "role": "back", "task": "no_robots", "nodes": ["N05"],
         "layers": [[9, 48]], "weights_gb": [17.0], "kv_gb": [3.0], "tokens_per_s": 1000.0, "routes": 1},
    ]
    dep["routes"] = [
        {"task": "gsm8k", "engines": ["mid-gsm8k-0", "back-gsm8k-0"], "tokens_per_s": 600.0,
         "share_of_task": 0.6, "avg_cycle_ms": 240.5, "weight": 144.3, "local_hops": 1},
        {"task": "gsm8k", "engines": ["mid-gsm8k-1", "back-gsm8k-0"], "tokens_per_s": 400.0,
         "share_of_task": 0.4, "avg_cycle_ms": 250.0, "weight": 100.0, "local_hops": 0},
        {"task": "mbpp", "engines": ["back-mbpp-0"], "tokens_per_s": 1000.0, "share_of_task": 1.0,
         "avg_cycle_ms": 235.0, "weight": 235.0, "local_hops": 0},
        {"task": "no_robots", "engines": ["back-no_robots-0"], "tokens_per_s": 1000.0,
         "share_of_task": 1.0, "avg_cycle_ms": 240.0, "weight": 240.0, "local_hops": 0},
    ]
    return res


def _args(entry, rank=0):
    toks = shlex.split(entry["commands"][rank]["cmd"].split(" --hf-overrides ")[0])
    out = {}
    for i, t in enumerate(toks):
        if t.startswith("--") and i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            out[t] = toks[i + 1]
    out["_env"] = {t.split("=")[0]: t.split("=", 1)[1] for t in toks if "=" in t and not t.startswith("-")}
    return out


def test_multi_level_plan_to_stages(tmp_path):
    sj, notes = _build(tmp_path, res=_multi_result())
    ch = {c["name"]: c for c in sj["channels"]}
    assert sj["mids"] == ["mid-gsm8k-0", "mid-gsm8k-1"]
    assert set(sj["backs"]) == {"back-gsm8k-0", "back-mbpp-0", "back-no_robots-0"}
    # 路线：中段 + 后段，weight 取规划的（Little 定律的在飞数）
    rts = {rt["name"]: rt for rt in sj["routes"]}
    assert rts["mid-gsm8k-0 → back-gsm8k-0"]["hops"] == ["mid-gsm8k-0", "back-gsm8k-0"]
    assert rts["mid-gsm8k-0 → back-gsm8k-0"]["weight"] == pytest.approx(144.3)
    # 中段：角色 / 层（0-based 半开）/ 驻留专家只列本段的层
    m0 = ch["mid-gsm8k-0"]
    a = _args(m0)
    assert a["--role"] == "mid" and a["--layers"] == "8,20" and m0["layers"] == [8, 20]
    le = _overrides(m0)["p2p_layer_experts"]
    assert set(le) == {str(l) for l in range(9, 20)}            # 第 9 层（0-based 8）装全，不列
    b0 = ch["back-gsm8k-0"]
    assert _args(b0)["--layers"] == "20,48"
    # 同卡两个引擎：端口 / master_port 错开，显存按（权重 + KV + 1）比例分
    assert m0["shared_gpu"] == {"N01": ["back-gsm8k-0"]} and b0["shared_gpu"] == {"N01": ["mid-gsm8k-0"]}
    assert {_args(m0)["--port"], _args(b0)["--port"]} == {"8300", "8310"}
    assert m0["zmq"] != b0["zmq"]
    g_mid, g_back = m0["gpu_mem"]["N01"], b0["gpu_mem"]["N01"]
    assert g_mid + g_back == pytest.approx(0.9, abs=2e-3)
    assert g_mid / g_back == pytest.approx((5 + 1 + 1) / (12 + 3 + 1), rel=1e-2)
    assert float(_args(m0)["--gpu-memory-utilization"]) == pytest.approx(g_mid)
    assert any("N01 上有 2 个引擎" in s for s in notes)
    # 不共卡的 mid-gsm8k-1 用槽 0、整卡比例
    m1 = ch["mid-gsm8k-1"]
    assert _args(m1)["--port"] == "8300" and m1["gpu_mem"] == {"N02": 0.9} and "shared_gpu" not in m1
    # 两级 PP 的后段：同一个槽、切层
    bm = ch["back-mbpp-0"]
    assert bm["partition"] == [20, 20] and _args(bm, 1)["--node-rank"] == "1"
    assert _args(bm)["_env"]["VLLM_PP_LAYER_PARTITION"] == "20,20"
    # max_num_seqs：经过它的路线的 流量 × 循环时间 之和 × 1.3
    assert b0["max_num_seqs"] >= (600 * 240.5 + 400 * 250) / 1e3 * 1.3


def test_multi_level_weights_specs(tmp_path):
    """weights.py 按节点合并：N01 上两个引擎 → 一份 spec，层 [8,20) + [20,48)，有 head、没有 embed。"""
    w = _load("weights")
    sj, _ = _build(tmp_path, res=_multi_result())
    specs = w.node_specs(sj)
    s1 = specs["N01"]["spec"]
    assert sorted(map(tuple, s1["layers"])) == [(8, 20), (20, 48)]
    assert s1["head"] and not s1["embed"]
    assert not specs["N02"]["spec"]["head"] and not specs["N02"]["spec"]["embed"]
    assert specs["N09"]["spec"]["embed"]


def test_multi_level_rejects_bad_routes(tmp_path):
    res = _multi_result()
    res["deploy"]["routes"][0]["engines"] = ["back-gsm8k-0", "mid-gsm8k-0"]
    with pytest.raises(cp.ColorPlanError, match="中段 \\+ 一个后段"):
        _build(tmp_path, res=res)
    res = _multi_result()
    res["deploy"]["engines"][1]["layers"] = [[9, 48]]          # 中段不能到最后一层
    with pytest.raises(cp.ColorPlanError, match="mid"):
        _build(tmp_path, res=res)
    res = _multi_result()
    del res["deploy"]["routes"]
    with pytest.raises(cp.ColorPlanError, match="routes"):
        _build(tmp_path, res=res)


def test_split_pp_turns_pipelines_into_per_card_engines(tmp_path):
    """--split-pp：每条后段流水线拆成每卡一个引擎（中段 … → 后段），每条流水线一条路线。"""
    inst_extra = {"n": L0, "kv_gb_per_tps_block": 0.001,
                  "w_color_gb": {k: [0.5] * (L - L0) for k in ("gsm8k", "mbpp", "no_robots")}}
    inst = {"L": L, "block_layers": 1, "batch": 64,
            "gpus": [{"id": f"N{i:02d}", "type": "A30"} for i in range(1, 16)],
            "gpu_types": {"A30": {"d_layer_ms": 3.6, "d_ms": 3.6}}, **inst_extra}
    sj, notes = cp.stages_from_color_plan(_result(), inst, _experts(), _hosts(tmp_path), model_dir="/w",
                                          python="/venv/bin/python", split_pp=True)
    ch = {c["name"]: c for c in sj["channels"]}
    assert any("--split-pp" in s for s in notes)
    # mbpp 有两条流水线：N03+N04+N05 与 N06+N07 → 中段 3 个、后段 2 个
    mb = [r for r in sj["routes"] if r["task"] == "mbpp"]
    assert sorted(len(r["hops"]) for r in mb) == [2, 3]
    for r in sj["routes"]:
        hops = [ch[h] for h in r["hops"]]
        assert [h["role"] for h in hops] == ["mid"] * (len(hops) - 1) + ["back"]
        assert hops[0]["layers"][0] == L0 and hops[-1]["layers"][1] == L
        for a_, b_ in zip(hops, hops[1:]):
            assert a_["layers"][1] == b_["layers"][0]          # 首尾相接
        for h in hops:
            assert len(h["nodes"]) == 1 and h["partition"] == [h["layers"][1] - h["layers"][0]]
            assert "VLLM_PP_LAYER_PARTITION" not in h["commands"][0]["cmd"]
            assert "--pipeline-parallel-size 1" in h["commands"][0]["cmd"]
    g = [r for r in sj["routes"] if r["task"] == "gsm8k"][0]
    assert g["weight"] == pytest.approx(1000 * 240 / 1e3)
    assert ch[g["hops"][0]]["gpu_mem"] == {"N01": 0.9}
    assert not any(c.get("shared_gpu") for c in sj["channels"])
