# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm/p2p/layer_bench.py：每层耗时的拟合、汇总、写拓扑，以及量法本身（假引擎 + 假时钟，不要 GPU）。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vllm.p2p import layer_bench as LB


def test_gpu_key_matches_topology_roles() -> None:
    assert LB.gpu_key("NVIDIA A30") == "A30" == LB.gpu_key("A30")
    assert LB.gpu_key("Quadro RTX 6000") == "RTX6000" == LB.gpu_key("RTX6000")
    assert LB.gpu_key("Tesla T4") == "T4"


def test_fit_per_layer() -> None:
    assert LB.fit_per_layer({2: 10.0, 6: 18.0}) == pytest.approx((2.0, 6.0))
    s, c = LB.fit_per_layer({1: 5.1, 3: 9.0, 5: 12.9})
    assert s == pytest.approx(1.95) and c == pytest.approx(3.1, abs=0.05)
    with pytest.raises(ValueError):
        LB.fit_per_layer({2: 1.0})


class _Clock:
    def __init__(self):
        self.t = 0.0

    def perf_counter(self):
        return self.t


class _FakeLLM:
    """一步 decode = 开销 + 层数 × 每层(batch, 上下文)；prefill = 开销 + 层数 × 每层(长度)。推进假时钟。"""

    def __init__(self, clock, layers, per_layer_dec, per_layer_pre, overhead=3.0):
        self.c, self.L, self.dec, self.pre, self.oh = clock, layers, per_layer_dec, per_layer_pre, overhead
        self.calls = []

    def generate(self, prompts, sp, use_tqdm=False):
        b, p = len(prompts), len(prompts[0]["prompt_token_ids"])
        self.calls.append((b, p, sp.max_tokens))
        ms = self.oh + self.L * self.pre(p * b)                      # prefill + 第一个 token
        ms += (sp.max_tokens - 1) * (self.oh + self.L * self.dec(b, p))
        self.c.t += ms / 1e3


def test_parse_lengths() -> None:
    assert LB.parse_lengths("128:1024:128") == [128, 256, 384, 512, 640, 768, 896, 1024]
    assert LB.parse_lengths(LB.DEFAULT_LENGTHS)[0] == 128 and LB.parse_lengths(LB.DEFAULT_LENGTHS)[-1] == 1024
    assert LB.parse_lengths("128,512") == [128, 512] and LB.parse_lengths("64:64") == [64]
    for bad in ("1024:128:128", "0:10:5", "", "128:1024:0"):
        with pytest.raises(ValueError):
            LB.parse_lengths(bad)


def test_measure_and_summarise_recover_per_layer_time(monkeypatch) -> None:
    clock = _Clock()
    monkeypatch.setattr(LB, "time", SimpleNamespace(perf_counter=clock.perf_counter))

    def dec(b, c):
        return 0.4 + 0.05 * b + 0.0001 * c

    def pre(n):
        return 0.002 * n

    runs = []
    for e in (128, 64):
        for l in (2, 6):
            llm = _FakeLLM(clock, l, dec if e == 128 else (lambda b, c: dec(b, c) * 0.9), pre)
            r = LB.measure(llm, batches=[1, 4], contexts=[128, 1024], decode_steps=16,
                           prefill_lens=[128, 512], repeats=2, vocab=2000)
            r.update(layers=l, experts=e)
            runs.append(r)
            # 每个（上下文, batch）：预热 1 次 + 每次重复两条（1 个 token / 1+K 个）；每次都是新 prompt
            assert (4, 1024, 17) in llm.calls and (1, 128, 17) in llm.calls and (1, 512, 1) in llm.calls
    res = LB.summarise({"node": "N01", "gpu": "NVIDIA A30", "runs": runs})
    d = res["decode"]["128"]
    assert d["128"]["1"] == pytest.approx(0.4628, abs=1e-3) and d["1024"]["4"] == pytest.approx(0.7024, abs=1e-3)
    assert res["decode"]["64"]["1024"]["4"] == pytest.approx(0.7024 * 0.9, abs=1e-3)
    assert res["decode_overhead_ms"]["128"]["128"]["1"] == pytest.approx(3.0, abs=1e-2)   # 与层数无关的部分
    assert res["prefill"]["128"]["512"] == pytest.approx(1.024, abs=1e-3)
    # 给规划器的数：E=128、B=1、两种上下文的均值
    assert res["summary"]["ms_per_layer"] == pytest.approx((0.4628 + 0.5524) / 2, abs=1e-3)
    assert LB.summarise({"runs": runs}, ref_context=1024)["summary"]["ms_per_layer"] == pytest.approx(0.5524, abs=1e-3)
    assert res["node"] == "N01" and len(res["runs"]) == 4


def _res(node, gpu, ms, mem=None):
    """B=1 时两种上下文的值是 0.9ms / 1.1ms → 均值正好 ms。"""
    return {"node": node, "gpu": gpu,
            "decode": {"128": {"128": {"1": ms * 0.9, "4": ms * 1.4}, "1024": {"1": ms * 1.1, "4": ms * 1.6}}},
            "prefill": {"128": {"512": ms * 4}},
            "memory": mem or {"128": {"weight_mib_per_layer": 1189.0, "device_non_kv_mib_fixed": 2000 + ms}},
            "summary": {"ref_batch": 1, "ref_experts": 128, "ref_context": "mean"}}


TOPO = {"nodes": [{"id": "N01", "compute_capacity": 165000.0, "memory_capacity": 24, "role": "A30"},
                  {"id": "N02", "compute_capacity": 165000.0, "memory_capacity": 24, "role": "A30"},
                  {"id": "N03", "compute_capacity": 165000.0, "memory_capacity": 24, "role": "A30"},
                  {"id": "N09", "compute_capacity": 130500.0, "memory_capacity": 24, "role": "RTX6000"},
                  {"id": "N12", "compute_capacity": 65000.0, "memory_capacity": 16, "role": "T4"}],
        "edges": [{"u": "N01", "v": "N02", "bandwidth": 1000, "propagation_delay": 0.1}]}

RES = [_res("N01", "NVIDIA A30", 0.5), _res("N02", "NVIDIA A30", 0.6),
       _res("N09", "Quadro RTX 6000", 1.9), _res("N12", "Tesla T4", 3.25)]


def test_aggregate_takes_the_mean_per_gpu_type() -> None:
    agg = LB.aggregate(RES)
    a30 = agg["A30"]
    assert a30["n"] == 2 and a30["nodes"] == ["N01", "N02"]
    assert a30["ms_per_layer"] == pytest.approx(0.55) and a30["slowest_over_fastest"] == pytest.approx(1.2)
    assert a30["ms_per_layer_nodes"] == {"N01": pytest.approx(0.5), "N02": pytest.approx(0.6)}
    assert a30["decode"]["128"]["1024"]["4"] == pytest.approx(0.55 * 1.6)        # 每一格都取均值
    assert a30["prefill"]["128"]["512"] == pytest.approx(2.2)
    assert a30["memory"]["128"]["device_non_kv_mib_fixed"] == pytest.approx(2000.55)
    assert agg["T4"]["n"] == 1 and agg["T4"]["ms_per_layer"] == pytest.approx(3.25)
    assert LB.aggregate(RES, ref_batch=4)["A30"]["ms_per_layer"] == pytest.approx(0.55 * 1.5)


def test_topo_with_measurements() -> None:
    out, notes = LB.topo_with_measurements(TOPO, RES)
    by = {n["id"]: n for n in out["nodes"]}
    # 默认：每个节点都用它这种卡型的均值（量过的 N01 / N02 和没量的 N03 一样）
    assert by["N01"]["ms_per_layer"] == by["N02"]["ms_per_layer"] == by["N03"]["ms_per_layer"] == 0.55
    assert "A30 均值（2 台实测）" == by["N03"]["ms_per_layer_src"]
    assert by["N09"]["ms_per_layer"] == 1.9 and by["N12"]["ms_per_layer"] == 3.25
    # 旧规划器只看 compute_capacity：改成与实测同比例，原值留着
    assert by["N01"]["compute_capacity"] == 165000.0 and by["N12"]["compute_capacity_orig"] == 65000.0
    assert by["N12"]["compute_capacity"] == pytest.approx(165000 * 0.55 / 3.25, rel=1e-3)
    assert out["edges"] == TOPO["edges"] and TOPO["nodes"][0].get("ms_per_layer") is None   # 原对象不动
    meta = out["ms_per_layer_meta"]
    assert meta["gpus"] == {"A30": 0.55, "RTX6000": 1.9, "T4": 3.25} and meta["gpus_n"]["A30"] == 2
    assert meta["memory_mib"]["A30"]["128"]["weight_mib_per_layer"] == 1189.0
    # per_node：量过的用自己的，没量过的用均值
    pn = {n["id"]: n for n in LB.topo_with_measurements(TOPO, RES, per_node=True)[0]["nodes"]}
    assert pn["N01"]["ms_per_layer"] == 0.5 and pn["N02"]["ms_per_layer"] == 0.6
    assert pn["N03"]["ms_per_layer"] == 0.55 and "本机" in pn["N01"]["ms_per_layer_src"]
    # 换 batch / 上下文取值
    out4, _ = LB.topo_with_measurements(TOPO, RES, ref_batch=4)
    assert {n["id"]: n["ms_per_layer"] for n in out4["nodes"]}["N09"] == pytest.approx(2.85)
    out_c, _ = LB.topo_with_measurements(TOPO, RES, ref_context=1024)
    assert {n["id"]: n["ms_per_layer"] for n in out_c["nodes"]}["N12"] == pytest.approx(3.575)
    # 缺卡型：默认报错；allow_missing 时按原算力比例换算并标出来
    with pytest.raises(ValueError, match="T4"):
        LB.topo_with_measurements(TOPO, RES[:3])
    out3, notes3 = LB.topo_with_measurements(TOPO, RES[:3], allow_missing=True)
    n12 = {n["id"]: n for n in out3["nodes"]}["N12"]
    assert n12["ms_per_layer"] == pytest.approx(1.9 * 130500 / 65000, rel=1e-3) and notes3
    with pytest.raises(ValueError, match="B=2"):
        LB.topo_with_measurements(TOPO, RES[:1], ref_batch=2, allow_missing=True)


def test_topo_cli_and_planner_reads_it(tmp_path, capsys) -> None:
    t = tmp_path / "topo.json"
    t.write_text(json.dumps(TOPO))
    rs = []
    for r in RES:
        p = tmp_path / f"{r['node']}.json"
        p.write_text(json.dumps(r))
        rs.append(str(p))
    out = tmp_path / "topo_measured.json"
    assert LB.main(["topo", "--topo", str(t), "--results", *rs, "--out", str(out)]) == 0
    assert LB.main(["topo", "--topo", str(t), "--results", *rs, "--out", str(t)]) == 2   # 不覆盖原文件
    assert "N12" in capsys.readouterr().out

    fw = LB.Path(__file__).resolve().parents[3] / "p2p-framework"
    if not (fw / "p2pmoe" / "planner" / "from_data.py").is_file():
        pytest.skip("旁边没有 p2p-framework")
    import sys
    sys.path.insert(0, str(fw))
    from p2pmoe.planner.from_data import load_topology

    ms = {n.id: n.ms_per_layer for n in load_topology(out).nodes}
    assert ms == {"N01": 0.55, "N02": 0.55, "N03": 0.55, "N09": 1.9, "N12": 3.25}
    old = {n.id: n.ms_per_layer for n in load_topology(t).nodes}     # 没有实测：旧口径
    assert old["N01"] == 0.35 and old["N12"] == pytest.approx(0.35 * 165000 / 65000, abs=1e-4)


def test_cluster_defaults_to_all_nodes(tmp_path, capsys) -> None:
    h = tmp_path / "hosts.txt"
    h.write_text("# 注释\nN01  192.168.1.2:9101   # A30 24GB\nN02  192.168.1.3:9101 # A30\n"
                 "N09  192.168.3.4:9101   # RTX6000 24GB\nN12 192.168.3.5:9101 # T4 16GB\n"
                 "N15 192.168.4.4 # T4\n")
    hosts = LB.parse_hosts(h)
    assert hosts["N09"] == ("192.168.3.4", "RTX6000") and hosts["N15"] == ("192.168.4.4", "T4")
    assert LB.pick_nodes(hosts, None) == LB.pick_nodes(hosts, "all") == ["N01", "N02", "N09", "N12", "N15"]
    assert LB.pick_nodes(hosts, "one-per-gpu") == ["N01", "N09", "N12"]
    assert LB.pick_nodes(hosts, "N02,N15") == ["N02", "N15"]
    with pytest.raises(SystemExit):
        LB.pick_nodes(hosts, "N99")
    a = SimpleNamespace(model_dir="/w", layers="2,6", experts="128,64", batches="1,4",
                        lengths="128:1024:128", prefill=None, contexts="256,512", decode_steps=32,
                        repeats=3, gpu_mem=0.85, max_batched_tokens=16384,
                        ref_batch=1, ref_experts=128, ref_context="mean",
                        eager=False, force=False, repo="/home/ubuntu/vllm-p2p",
                        python="/home/ubuntu/venvs/vllm-p2p/bin/python")
    cmd = LB.remote_command("N09", a)
    assert cmd.startswith("cd /home/ubuntu/vllm-p2p && /home/ubuntu/venvs/vllm-p2p/bin/python -m "
                          "vllm.p2p.layer_bench run ")
    assert "--node N09" in cmd and "--out /tmp/layer_bench_N09.json" in cmd and "--eager" not in cmd
    assert "--lengths 128:1024:128" in cmd and "--contexts 256,512" in cmd and "--prefill" not in cmd
    assert "--max-batched-tokens 16384" in cmd
    # 汇总表能打出来（均值表：上下文 × batch 一格一格）
    LB.print_gpu_summary(LB.aggregate(RES))
    txt = capsys.readouterr().out
    assert "A30" in txt and "N01:0.5" in txt and "2 台" in txt and "1024" in txt


def test_overrides_only_list_experts_when_partial() -> None:
    assert "p2p_layer_experts" not in LB._overrides(2, 128, 128)
    ov = LB._overrides(3, 64, 128)
    assert ov["num_hidden_layers"] == 3 and ov["architectures"] == [LB.ARCH]
    assert sorted(ov["p2p_layer_experts"]) == ["0", "1", "2"] and len(ov["p2p_layer_experts"]["2"]) == 64
    # 与模型读驻留集的解析器同口径（越界、空集都会报错）
    from vllm.p2p.layout import parse_layer_experts
    assert len(parse_layer_experts(ov["p2p_layer_experts"], 3, 128)[1]) == 64


def _fake_worker(n_layers, n_exp, kv_blocks=10, block=16):
    import torch
    from torch import nn

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.w13_weight = nn.Parameter(torch.zeros(n_exp, 2 * 6, 8, dtype=torch.float16))
            self.w2_weight = nn.Parameter(torch.zeros(n_exp, 8, 6, dtype=torch.float16))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Linear(8, 8, bias=False).half()
            self.input_layernorm = nn.LayerNorm(8, elementwise_affine=True).half()
            self.mlp = nn.Module()
            self.mlp.gate = nn.Linear(8, 128, bias=False).half()
            self.mlp.experts = Experts()

    inner = nn.Module()
    inner.layers = nn.ModuleList([Layer() for _ in range(n_layers)])
    inner.start_layer, inner.end_layer = 0, n_layers
    inner.embed_tokens = nn.Embedding(100, 8).half()
    model = nn.Module()
    model.model = inner
    model.lm_head = nn.Linear(8, 100, bias=False).half()
    kv = [torch.zeros(2, kv_blocks, block, 1, 4, dtype=torch.float16) for _ in range(n_layers)]
    mr = SimpleNamespace(get_model=lambda: model, kv_caches=kv)
    return SimpleNamespace(model_runner=mr, cache_config=SimpleNamespace(num_gpu_blocks=kv_blocks,
                                                                         block_size=block))


def test_mem_probe_counts_each_layer_exactly() -> None:
    m = LB.mem_probe(_fake_worker(3, 5))
    lay = m["layers"][0]
    per_exp = (12 * 8 + 8 * 6) * 2
    assert len(m["layers"]) == 3 and lay["n_experts"] == 5
    assert lay["experts_bytes"] == 5 * per_exp and lay["router_bytes"] == 8 * 128 * 2
    assert lay["attn_norm_bytes"] == (8 * 8 + 8 + 8) * 2
    assert m["non_layer_param_bytes"] == (100 * 8 + 100 * 8) * 2          # embed + lm_head
    assert m["kv_bytes"] == 3 * 2 * 10 * 16 * 4 * 2 and m["kv_tokens"] == 160
    # KV 张量共享存储时只算一次
    w = _fake_worker(2, 5)
    w.model_runner.kv_caches = [w.model_runner.kv_caches[0]] * 2
    assert LB.mem_probe(w)["kv_bytes"] == 2 * 10 * 16 * 4 * 2


def test_summarise_memory_fits_per_layer_and_fixed() -> None:
    MiB = LB.MiB

    def probe(L, E):
        lay = {"bytes": (40 + E * 9) * MiB, "experts_bytes": E * 9 * MiB, "router_bytes": 1 * MiB,
               "attn_norm_bytes": 39 * MiB, "n_experts": E}
        kv = 2000 * MiB
        params = L * lay["bytes"] + 1200 * MiB
        used = params + kv + 900 * MiB + L * 5 * MiB          # 固定 900 + 每层 5（图池 / 激活）
        return {"layers": [lay] * L, "param_bytes": params, "non_layer_param_bytes": 1200 * MiB,
                "kv_bytes": kv, "kv_tokens": 2000 * MiB // (L * 2048), "kv_layers": L,
                "device_total": 24000 * MiB, "device_free": 24000 * MiB - used,
                "torch_peak_allocated": params + kv + 300 * MiB + L * 2 * MiB}

    runs = [{"layers": L, "experts": 64, "decode": {"512": {"1": 1.0 + L}}, "prefill": {},
             "mem": {"after_init": probe(L, 64), "after_run": probe(L, 64)}} for L in (2, 6)]
    m = LB.summarise({"runs": runs})["memory"]["64"]
    assert m["weight_mib_per_layer"] == pytest.approx(40 + 64 * 9)
    assert m["mib_per_expert"] == pytest.approx(9) and m["experts_per_layer"] == 64
    assert m["non_layer_weight_mib"] == pytest.approx(1200)
    assert m["kv_kib_per_token_per_layer"] == pytest.approx(2.0, rel=1e-3)
    assert m["device_non_kv_mib_per_layer"] == pytest.approx(40 + 64 * 9 + 5, abs=0.1)
    assert m["device_non_kv_mib_fixed"] == pytest.approx(1200 + 900, abs=1)
    assert m["activation_peak_mib_per_layer"] == pytest.approx(2, abs=0.01)
    assert m["activation_peak_mib_fixed"] == pytest.approx(300, abs=1)
