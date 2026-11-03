# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P2P 移植：`vllm/p2p/launch.py` 生成的远端 shell（不用 ssh，直接在本机 bash 里跑）。

钉住的是两个实际踩过的坑：
* `cd /tmp && nohup … &` 会让 ssh 卡到 vllm 退出 —— start 必须立即返回；
* `pgrep -f` 用字面量会匹配到执行查询的 shell 自己 —— 必须只找到真正的进程。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

import pytest

from vllm.p2p import launch as L

pytestmark = pytest.mark.skipif(shutil.which("bash") is None or sys.platform == "win32",
                                reason="需要 bash / pgrep")

# 不在本文件里拼出完整字面量，免得 pgrep 匹配到 pytest 进程自己
NAME = "p2ptest" + "-0"
FLAG = "--served" + "-model-name"


def _sh(s: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", s], capture_output=True, text=True, timeout=30)


def test_start_returns_and_passes_env_and_overrides(tmp_path) -> None:
    (tmp_path / "ov.json").write_text('{"architectures": ["X"]}', encoding="utf-8")
    got = tmp_path / "got"
    fake = (f"VLLM_PP_LAYER_PARTITION=8,13,27 {sys.executable} -c 'import os,sys,time; "
            f"open(\"{got}\",\"w\").write(os.environ[\"VLLM_PP_LAYER_PARTITION\"]+\"|\""
            f"+os.environ.get(\"EXTRA\",\"\")+\"|\"+sys.argv[-1]); time.sleep(60)' "
            f"{FLAG} {NAME} --hf-overrides \"$(cat {tmp_path / 'ov.json'})\"")
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                              FLAG, "p2ptest" + "-1", "--x"])
    try:
        t = time.time()
        r = _sh(L.remote_start(fake, NAME, "N01", {"EXTRA": "e1"}))
        assert time.time() - t < 5, "start 没有立即返回 —— ssh 会卡到 vllm 退出"
        pid = r.stdout.strip()
        for _ in range(50):
            if got.exists() and got.read_text():
                break
            time.sleep(0.1)
        assert got.read_text() == '8,13,27|e1|{"architectures": ["X"]}'
        assert _sh(L.remote_running(NAME)).stdout.split() == [pid]
        assert _sh(L.remote_running("p2ptest")).stdout.split() == []      # 前缀不误匹配
        _sh(L.remote_stop(NAME))
        assert _sh(L.remote_running(NAME)).stdout.split() == []
        assert other.poll() is None, "停一条通道把别的通道也杀了"
    finally:
        other.kill()
        _sh(L.remote_stop(NAME))
        if os.path.exists(L.log_path(NAME, "N01")):
            os.remove(L.log_path(NAME, "N01"))


def test_match_pattern_does_not_match_its_own_query() -> None:
    import re
    pat = L.match_pattern(NAME)
    assert re.search(pat, f"vllm serve /w {FLAG} {NAME} --pipeline-parallel-size 3")
    assert re.search(pat, f"vllm serve /w {FLAG} {NAME}")
    assert not re.search(pat, f"vllm serve /w {FLAG} {NAME}0 --x")
    assert not re.search(pat, L.remote_running(NAME))     # 执行查询的 shell 自己的命令行


def test_venv_is_put_on_path() -> None:
    cmd = ("VLLM_PP_LAYER_PARTITION=8,13,27 /home/ubuntu/venvs/vllm-p2p/bin/vllm serve /w "
           f"{FLAG} {NAME} --hf-overrides \"$(cat /x/ov.json)\"")
    assert L.venv_of(cmd) == "/home/ubuntu/venvs/vllm-p2p"
    assert L.venv_of(f"vllm serve /w {FLAG} {NAME}") is None          # 没用绝对路径就不猜
    script = L.remote_start(cmd, NAME, "N01", {})
    assert "export PATH=/home/ubuntu/venvs/vllm-p2p/bin:$PATH" in script
    assert "VIRTUAL_ENV=/home/ubuntu/venvs/vllm-p2p" in script


def test_net_pins_cluster_nic() -> None:
    import shutil
    import subprocess

    import pytest
    if not shutil.which("ip"):
        pytest.skip("没有 ip 命令（iproute2）")
    sh = L.remote_net("127.0.0.1") + 'echo "$GLOO_SOCKET_IFNAME|$NCCL_SOCKET_IFNAME|$VLLM_HOST_IP"'
    r = subprocess.run(["bash", "-c", sh], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "lo|lo|127.0.0.1"
    r = subprocess.run(["bash", "-c", L.remote_net("10.255.255.254") + "echo no"],
                       capture_output=True, text=True)
    assert r.returncode == 3 and "10.255.255.254" in r.stderr and "no" not in r.stdout


def test_user_env_overrides_detected_nic() -> None:
    cmd = f"VLLM_PP_LAYER_PARTITION=8 /v/bin/vllm serve /w {FLAG} {NAME}"
    script = L.remote_start(cmd, NAME, "N01", {"GLOO_SOCKET_IFNAME": "eth9"}, "192.168.1.2")
    assert script.index("GLOO_SOCKET_IFNAME=$P2P_IF") < script.index("GLOO_SOCKET_IFNAME=eth9")


def test_ident_command_uses_the_channel_venv(tmp_path) -> None:
    """识别服务用通道命令里 vllm 所在的 venv，跑在前段节点上，带上清单的 L₀ 与 task。"""
    cmd = ("VLLM_PP_LAYER_PARTITION=8,13,27 /home/ubuntu/venvs/vllm-p2p/bin/vllm serve /w/q3 "
           f"{FLAG} gsm8k-0 --node-rank 0 --hf-overrides \"$(cat /x/ov.json)\"")
    back = cmd.replace("--node-rank 0", "--node-rank 1")
    chans = [{"name": "gsm8k-0", "task": "gsm8k",
              "commands": [{"node": "N01", "host": "192.168.1.2", "cmd": cmd},
                           {"node": "N03", "host": "192.168.1.6", "cmd": back}]},
             {"name": "mbpp-0", "task": "mbpp", "commands": []}]
    host, c = L.ident_command(chans, {"l0": 8, "model_dir": "/w/q3"}, "N01", port=8100,
                              src="/home/ubuntu/vllm-p2p", p2p_framework="/p2p",
                              lr_model="/p2p/sel.npz", tasks=None)
    assert host == "192.168.1.2"
    assert c.startswith("/home/ubuntu/venvs/vllm-p2p/bin/python "
                        "/home/ubuntu/vllm-p2p/vllm/p2p/identifier.py --p2p-identifier")
    assert "--tasks gsm8k=1,mbpp=1" in c and "--l0 8" in c and "--model-dir /w/q3" in c
    assert L.venv_of(c) == "/home/ubuntu/venvs/vllm-p2p"
    assert "[-]-p2p-identifier" in L.ident_running()


def test_stop_kills_the_whole_process_group(tmp_path) -> None:
    """vllm 起的子进程（EngineCore / worker）命令行里没有 --served-model-name，只按命令行
    pkill 会漏掉它们 —— 它们占着显存，下一次启动在预热时 OOM。stop 要按进程组杀。"""
    marker = f"p2p-orphan-{os.getpid()}"
    child = f"import time; time.sleep(120)  # {marker}"
    pat = "[" + marker[0] + "]" + marker[1:]           # 不让 pgrep 匹配到查询它的那个 shell
    fake = (f"VLLM_PP_LAYER_PARTITION=8 {sys.executable} -c "
            f"'import subprocess,sys,time; subprocess.Popen([sys.executable, \"-c\", \"{child}\"]); "
            f"time.sleep(120)' {FLAG} {NAME}")
    try:
        _sh(L.remote_start(fake, NAME, "N01", {}))
        for _ in range(50):
            if _sh(f"pgrep -f -- '{pat}' || true").stdout.strip():
                break
            time.sleep(0.1)
        assert _sh(f"pgrep -f -- '{pat}' || true").stdout.strip(), "子进程没起来"
        _sh(L.remote_stop(NAME))
        time.sleep(0.5)
        assert _sh(f"pgrep -f -- '{pat}' || true").stdout.strip() == "", "子进程成了孤儿"
        assert _sh(L.remote_running(NAME)).stdout.split() == []
    finally:
        _sh(f"pkill -f -- '{pat}' || true")
        _sh(L.remote_stop(NAME))
        if os.path.exists(L.log_path(NAME, "N01")):
            os.remove(L.log_path(NAME, "N01"))


def test_parse_gpu() -> None:
    out = ("23500, 24576\n---\n1234, 20000, /home/ubuntu/venvs/vllm-p2p/bin/python3\n---\n"
           " 1234    01:02:03 python3 -c from multiprocessing.spawn import spawn_main\n")
    g = L.parse_gpu(out)
    assert g["used_mb"] == 23500 and g["total_mb"] == 24576
    assert g["apps"] == [(1234, 20000, "/home/ubuntu/venvs/vllm-p2p/bin/python3")]
    assert "spawn_main" in g["ps"][0]
    assert L.parse_gpu("")["used_mb"] is None


def test_shared_gpu_engine_starts_when_its_share_is_free(monkeypatch, capsys) -> None:
    """多级：同卡第二个引擎起的时候卡上已经有第一个引擎 —— 剩下的够它的比例就放行，不够才拦。"""
    calls = []

    def fake_ssh(host, user, script, timeout=60):
        calls.append(script)
        if script == L.GPU_Q:
            return subprocess.CompletedProcess([], 0, stdout=f"{used[0]}, 24000\n---\n---\n", stderr="")
        return subprocess.CompletedProcess([], 0, stdout="" if "pgrep" in script else "123", stderr="")

    monkeypatch.setattr(L, "_ssh", fake_ssh)
    c = {"name": "back-t-0", "api": "http://x:1/v1", "gpu_mem": {"N01": 0.5},
         "shared_gpu": {"N01": ["mid-t-0"]},
         "commands": [{"node": "N01", "host": "10.0.0.1", "cmd": "/v/bin/python -m x --hf-overrides {}"}]}
    used = [9000]                       # 空闲 0.625 ≥ 0.5
    assert L.cmd_start(c, "u", {}, wait=0) == 0
    assert "同卡" in capsys.readouterr().out
    used[0] = 15000                     # 空闲 0.375 < 0.5
    assert L.cmd_start(c, "u", {}, wait=0) == 1
    c2 = dict(c, shared_gpu={})         # 不是同卡的引擎：照旧拦
    used[0] = 9000
    assert L.cmd_start(c2, "u", {}, wait=0) == 1
