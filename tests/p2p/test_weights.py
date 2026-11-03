# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""weights_node.py / weights.py：按清单核对、修复、下载各节点本地需要的权重。不要 GPU、不要网络：
源用一个假的 checkpoint 目录（--source dir:），ssh 换成在本机执行的假 ssh。"""

import base64
import importlib.util
import io
import json
import os
import stat
import subprocess
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


WN = _load("weights_node")
W = _load("weights")

# 假 checkpoint：4 层、每层 4 个专家，4 个分片
WEIGHT_MAP = {
    "model.embed_tokens.weight": "s1.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "s1.safetensors",
    **{f"model.layers.0.mlp.experts.{e}.up_proj.weight": "s1.safetensors" for e in range(4)},
    "model.layers.1.self_attn.q_proj.weight": "s2.safetensors",
    "model.layers.1.mlp.gate.weight": "s2.safetensors",
    **{f"model.layers.1.mlp.experts.{e}.up_proj.weight": "s2.safetensors" for e in (0, 1)},
    **{f"model.layers.1.mlp.experts.{e}.up_proj.weight": "s3.safetensors" for e in (2, 3)},
    "model.layers.2.self_attn.q_proj.weight": "s3.safetensors",
    **{f"model.layers.2.mlp.experts.{e}.up_proj.weight": "s3.safetensors" for e in range(4)},
    "model.layers.3.self_attn.q_proj.weight": "s4.safetensors",
    **{f"model.layers.3.mlp.experts.{e}.up_proj.weight": "s4.safetensors" for e in range(4)},
    "model.norm.weight": "s4.safetensors",
    "lm_head.weight": "s4.safetensors",
}
CONFIG = {"model_type": "qwen3_moe", "num_hidden_layers": 4, "num_experts": 4, "hidden_size": 8,
          "num_experts_per_tok": 2}


def _source(tmp_path, config=CONFIG):
    src = tmp_path / "hub"
    src.mkdir()
    (src / "config.json").write_text(json.dumps(config))
    (src / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": WEIGHT_MAP}))
    (src / "tokenizer.json").write_text("{}")
    for k, s in enumerate(sorted(set(WEIGHT_MAP.values())), 1):
        (src / s).write_bytes(bytes([k]) * (1000 * k))
    return src


def _b64(spec):
    return base64.b64encode(json.dumps(spec).encode()).decode()


# 第 1–2 层（0-based），第 1 层只要专家 0 → 要 s2（第 1 层注意力 + 专家 0）、s3（第 2 层）；不要 s1、s4
SPEC = {"node": "N05", "layers": [[1, 3]], "embed": False, "head": False, "experts": {"1": [0]}}


def _run(tmp_path, dst, *extra, spec=SPEC):
    out = io.StringIO()
    argv = ["--dir", str(dst), "--spec", _b64(spec), "--source", f"dir:{tmp_path / 'hub'}", *extra]
    from contextlib import redirect_stdout
    with redirect_stdout(out):
        rc = WN.main(argv)
    text = out.getvalue()
    rep = next((json.loads(ln.split(" ", 1)[1]) for ln in text.splitlines() if ln.startswith(WN.REPORT_MARK)),
               None)
    return rc, text, rep


def test_tensor_level_need():
    shards, n = WN.needed_shards(WEIGHT_MAP, SPEC)
    assert shards == ["s2.safetensors", "s3.safetensors"] and n == 1 + 1 + 1 + 1 + 4
    front = {"layers": [[0, 2]], "embed": True, "head": False, "experts": {}}
    assert WN.needed_shards(WEIGHT_MAP, front)[0] == ["s1.safetensors", "s2.safetensors", "s3.safetensors"]
    last = {"layers": [[3, 4]], "embed": False, "head": True, "experts": {"3": [1]}}
    assert WN.needed_shards(WEIGHT_MAP, last)[0] == ["s4.safetensors"]
    # 第 1 层只要专家 2：s2（注意力、门）和 s3（专家 2）都要
    assert WN.needed_shards(WEIGHT_MAP, {"layers": [[1, 2]], "experts": {"1": [2]}})[0] == \
        ["s2.safetensors", "s3.safetensors"]


def test_check_fetch_repair_prune(tmp_path):
    _source(tmp_path)
    dst = tmp_path / "w"
    rc, _, rep = _run(tmp_path, dst, "--check")
    assert rc == 0 and rep["status"] == {"s2.safetensors": "missing", "s3.safetensors": "missing"}
    assert not dst.exists()                                  # 只核对：不建目录、不改动
    rc, text, _ = _run(tmp_path, dst)
    assert rc == 0 and WN.OK_MARK in text
    assert sorted(p.name for p in dst.glob("*.safetensors")) == ["s2.safetensors", "s3.safetensors"]
    assert json.loads((dst / WN.PROGRESS).read_text())["state"] == "done"
    assert (dst / "config.json").exists() and (dst / "tokenizer.json").exists()
    _, _, rep = _run(tmp_path, dst, "--check")
    assert rep["ok"] and rep["need_bytes"] == 0

    # 坏了一个（大小不对）+ 多了一个用不到的 + 残留的 .part
    (dst / "s3.safetensors").write_bytes(b"x" * 10)
    (dst / "s1.safetensors").write_bytes(b"\x01" * 1000)
    (dst / "s4.safetensors.part").write_bytes(b"y")
    _, _, rep = _run(tmp_path, dst, "--check")
    assert rep["status"]["s3.safetensors"] == "bad_size" and rep["extra"] == ["s1.safetensors"]
    assert rep["stale"] == ["s4.safetensors.part"] and not rep["ok"]
    rc, _, _ = _run(tmp_path, dst, "--keep-extra")            # 留着用不到的：重下坏的、清残留
    assert rc == 0 and (dst / "s3.safetensors").stat().st_size == 3000
    assert (dst / "s1.safetensors").exists() and not (dst / "s4.safetensors.part").exists()
    idx = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(idx.values()) == {"s1.safetensors", "s2.safetensors", "s3.safetensors"}
    rc, _, _ = _run(tmp_path, dst)                            # 默认：用不到的删掉，index 只列留下的
    assert rc == 0 and not (dst / "s1.safetensors").exists()
    idx = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(idx.values()) == {"s2.safetensors", "s3.safetensors"}
    assert idx == {n: s_ for n, s_ in WEIGHT_MAP.items() if s_ in ("s2.safetensors", "s3.safetensors")}
    _, _, rep = _run(tmp_path, dst, "--check")
    assert rep["ok"] and rep["extra"] == []


def test_wrong_model_is_wiped_and_redownloaded(tmp_path):
    _source(tmp_path)
    dst = tmp_path / "w"
    dst.mkdir()
    (dst / "config.json").write_text(json.dumps({**CONFIG, "num_experts": 8}))
    (dst / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    (dst / "s2.safetensors").write_bytes(b"\x02" * 2000)       # 大小碰巧对，但模型不对：也要重下
    (dst / "other.bin").write_bytes(b"z")
    _, _, rep = _run(tmp_path, dst, "--check")
    assert "num_experts 8≠4" in rep["wrong_model"] and set(rep["status"].values()) == {"missing"}
    rc, text, _ = _run(tmp_path, dst)
    assert rc == 0 and "模型不对" in text
    assert (dst / "other.bin").exists()                       # 不是本脚本管的文件：不动
    assert json.loads((dst / "config.json").read_text())["num_experts"] == 4
    assert (dst / "s2.safetensors").read_bytes() == b"\x02" * 2000 and (dst / "s3.safetensors").exists()


def test_partial_checkpoint_index_is_not_wrong_and_not_overwritten(tmp_path):
    _source(tmp_path)
    dst = tmp_path / "w"
    dst.mkdir()
    (dst / "config.json").write_text(json.dumps(CONFIG))
    sub = {n: s for n, s in WEIGHT_MAP.items() if s == "s2.safetensors"}     # 只列自己有的张量
    (dst / "model.safetensors.index.json").write_text(json.dumps({"weight_map": sub}))
    (dst / "s2.safetensors").write_bytes(b"\x02" * 2000)
    _, _, rep = _run(tmp_path, dst, "--check")
    assert rep["wrong_model"] is None and rep["status"] == {"s2.safetensors": "ok", "s3.safetensors": "missing"}
    rc, _, _ = _run(tmp_path, dst)
    assert rc == 0 and (dst / "s3.safetensors").exists()
    # index 改成列出目录里留下的分片（新下的 s3 也在里面，否则 vLLM 不会加载它）
    idx = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(idx.values()) == {"s2.safetensors", "s3.safetensors"}
    # 张量指向别的分片：不是同一种切分 → 判不对
    bad = dict(sub)
    bad[next(iter(bad))] = "s9.safetensors"
    (dst / "model.safetensors.index.json").write_text(json.dumps({"weight_map": bad}))
    _, _, rep = _run(tmp_path, dst, "--check")
    assert "所在分片不同" in rep["wrong_model"]


def test_disk_space_guard(tmp_path):
    _source(tmp_path)
    rc = None
    with pytest.raises(SystemExit, match="磁盘不够"):
        rc = _run(tmp_path, tmp_path / "w", "--reserve-gb", "1e9")
    assert rc is None


# --------------------------------------------------------------------------- #
# 控制机：清单 → 各节点要什么；check / fetch 走（假的）ssh
# --------------------------------------------------------------------------- #
def _stages(tmp_path):
    import shlex

    def cmd(name, role, layers, nodes, rank, part=None, ov=None):
        c = (f"/venv/bin/python -m vllm.p2p.stage_agent --served-model-name {name} --model /w "
             f"--role {role} --layers {layers[0]},{layers[1]} --pipeline-parallel-size {len(nodes)}"
             + (f" --nnodes {len(nodes)} --node-rank {rank}" if len(nodes) > 1 else " --node-rank 0")
             + " --port 8300")
        if part:
            c = f"VLLM_PP_LAYER_PARTITION={part} {c}"
        return c + " --hf-overrides " + shlex.quote(json.dumps(ov or {}))
    back_ov = {"p2p_layer_experts": {"1": [0], "2": [0, 1, 2, 3]}}
    return {"kind": "tiered", "l0": 1, "n_layers": 4, "model_dir": str(tmp_path / "w"),
            "channels": [
                {"name": "front-N01", "role": "front", "nodes": ["N01"],
                 "commands": [{"node": "N01", "host": "h1", "cmd": cmd("front-N01", "front", (0, 1), ["N01"], 0)}]},
                {"name": "back-gsm8k-0", "role": "back", "nodes": ["N02", "N03"],
                 "commands": [{"node": v, "host": f"h{k + 2}", "cmd": cmd("back-gsm8k-0", "back", (1, 4),
                                                                          ["N02", "N03"], k, "2,1", back_ov)}
                              for k, v in enumerate(["N02", "N03"])]}]}


def test_node_specs_from_stages(tmp_path):
    ns = W.node_specs(_stages(tmp_path))
    assert ns["N01"]["spec"] == {"node": "N01", "layers": [[0, 1]], "embed": True, "head": False, "experts": {}}
    assert ns["N02"]["spec"]["layers"] == [[1, 3]] and ns["N02"]["spec"]["experts"] == {"1": [0], "2": [0, 1, 2, 3]}
    assert ns["N03"]["spec"] == {"node": "N03", "layers": [[3, 4]], "embed": False, "head": True, "experts": {}}
    assert ns["N02"]["python"] == "/venv/bin/python" and ns["N03"]["host"] == "h3"


def test_node_specs_on_real_color_stages():
    p = Path(__file__).resolve().parents[2] / "p2p_channels" / "stages_color.json"
    if not p.exists():
        pytest.skip("没有 p2p_channels/stages_color.json")
    sj = json.loads(p.read_text())
    ns = W.node_specs(sj)
    fronts = [v for v, i in ns.items() if i["spec"]["embed"]]
    heads = [v for v, i in ns.items() if i["spec"]["head"]]
    assert all(ns[v]["spec"]["layers"] == [[0, sj["l0"]]] for v in fronts)
    assert len(heads) == len(sj["backs"])                        # 每条后段流水线一个末级
    cover = sorted((a, b) for v, i in ns.items() if not i["spec"]["embed"] for a, b in i["spec"]["layers"])
    assert cover[0][0] == sj["l0"]


def _fake_ssh(tmp_path):
    f = tmp_path / "fake_ssh"
    f.write_text('#!/bin/bash\nshift\nexec bash -c "$1"\n')       # 丢掉 user@host，在本机执行
    f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return str(f)


def test_check_and_fetch_end_to_end(tmp_path, monkeypatch, capsys):
    _source(tmp_path)
    sj = _stages(tmp_path)
    for c in sj["channels"]:                                     # 三台“节点”用三个目录模拟不了同一路径：
        for st in c["commands"]:                                 # 用 --nodes 一台一台地验
            st["cmd"] = st["cmd"].replace("/venv/bin/python", sys.executable)
    sp = tmp_path / "stages.json"
    sp.write_text(json.dumps(sj))
    monkeypatch.setattr(W, "SSH", [_fake_ssh(tmp_path)])
    base = ["--stages", str(sp), "--nodes", "N02", "--source", f"dir:{tmp_path / 'hub'}"]
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "s4.safetensors").write_bytes(b"\x04" * 4000)   # 用不到的：fetch 时删
    assert W.main(["check", *base]) == 1
    out = capsys.readouterr().out
    assert "N02" in out and "要下载" in out
    assert W.main(["fetch", *base, "--interval", "0.1", "--log-dir", str(tmp_path / "logs")]) == 0
    out = capsys.readouterr().out
    assert "各节点权重下载进度" in out and "完成" in out and "全部节点就绪" in out
    assert sorted(p.name for p in (tmp_path / "w").glob("*.safetensors")) == ["s2.safetensors", "s3.safetensors"]
    assert "删掉用不到的分片" in out
    assert W.main(["check", *base]) == 0 and "就绪" in capsys.readouterr().out
    assert W.main(["status", "--stages", str(sp), "--nodes", "N02"]) == 0
    assert "完成" in capsys.readouterr().out


def test_progress_table_speed_and_eta():
    nodes = {"N01": {"host": "10.0.0.1", "engine": "front-N01"},
             "N02": {"host": "10.0.0.2", "engine": "back-gsm8k-0"}}
    prog = {"state": "downloading", "need": 8e9, "base": 1e9, "shards_done": 1, "shards_total": 2,
            "current": "model-00003-of-00016.safetensors", "started": 0.0}
    prev = {"N01": (90.0, 2e9)}
    buf = io.StringIO()
    tot = W.print_progress_table(nodes, {"N01": (prog, 5e9), "N02": (None, None)}, prev,
                                 now=100.0, clear=False, file=buf)
    text = buf.getvalue()
    assert tot["done"] == pytest.approx(4e9) and tot["speed"] == pytest.approx(3e8)   # 3 GB / 10 s
    assert "下载中" in text and "50" in text and "model-00003-of-00016" in text and "连不上" in text
    assert prev["N01"] == (100.0, 5e9)
