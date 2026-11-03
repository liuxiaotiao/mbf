# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""拉起 / 查看 / 停止 vLLM 静态通道（P2P 移植阶段 A2）。**纯 Python，控制机上按路径直接跑。**

    python3 vllm/p2p/launch.py start  gsm8k-0          # 按 channels.json 在各级节点上后台启动
    python3 vllm/p2p/launch.py status gsm8k-0          # 进程在不在、API 健康检查通没通、KV 容量
    python3 vllm/p2p/launch.py logs   gsm8k-0 [-n 40]  # 各级日志的最后几行
    python3 vllm/p2p/launch.py errors gsm8k-0          # 各级日志里的错误行（找根因用）
    python3 vllm/p2p/launch.py stop   gsm8k-0          # 停掉（按进程组，连子进程一起）
    python3 vllm/p2p/launch.py gpu    gsm8k-0          # 各级卡上的显存与占用进程
    python3 vllm/p2p/launch.py status all              # 所有通道

    # A4 的识别服务（前 L₀ 层 + LR）：起在一台前段节点上（它本地有层 1..L₀ 的全专家权重）
    python3 vllm/p2p/launch.py ident-start  N01 [--ident-port 8100] [--lr-model …]
    python3 vllm/p2p/launch.py ident-status N01 | ident-logs N01 | ident-stop N01

`channels.json` 由 `vllm/p2p/layout.py --hosts … --out DIR` 生成（默认读
/home/ubuntu/vllm-p2p/p2p_channels/channels.json，`--channels` 可改）。各级的启动命令、
地址都在里面，这里只负责 ssh 过去执行 —— 命令本身一个字不改，保证与 layout 生成的一致。

日志在各节点的 /tmp/p2p_vllm_<通道>_<节点>.log。
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_CHANNELS = "/home/ubuntu/vllm-p2p/p2p_channels/channels.json"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "StrictHostKeyChecking=accept-new"]


def log_path(channel: str, node: str) -> str:
    return f"/tmp/p2p_vllm_{channel}_{node}.log"


def match_pattern(channel: str) -> str:
    """pgrep / pkill 用的**正则**：只认本通道的 vllm serve 进程。

    写成 `[-]-served-model-name …` 而不是字面量：ssh 执行 `pgrep -f '<模式>'` 的那个 shell，
    自己的命令行里就带着这串字符 —— 字面量会匹配到它，于是「没在跑」也报成在跑。
    方括号写法匹配真正的 `--served-model-name`，却匹配不到写着 `[-]-served…` 的那条命令行。
    """
    return f"[-]-served-model-name {channel}( |$)"


def venv_of(cmd: str) -> str | None:
    """从启动命令里认出 venv：命令里的程序是绝对路径 `<venv>/bin/vllm`（或 `/bin/python`）时
    返回 `<venv>`。"""
    for tok in shlex.split(cmd.split(" --hf-overrides ", 1)[0]):
        if "=" in tok and not tok.startswith("/"):
            continue                      # 开头的 VAR=值
        for exe in ("/bin/vllm", "/bin/python", "/bin/python3"):
            if tok.endswith(exe):
                return tok[: -len(exe)]
        return None
    return None


NET_VARS = ("GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME", "VLLM_HOST_IP")


def remote_net(host: str) -> str:
    """在远端找出「承载 host 这个 IP 的网卡」，导出 Gloo / NCCL / vLLM 用的网卡与地址。

    不设的话 Gloo 用 hostname 解析出本机地址，Ubuntu 的 /etc/hosts 默认把 hostname 指到
    127.0.1.1 —— 每一级都对外宣告 127.0.1.1，别的节点连过去就是 Connection refused
    （第一次跑 gsm8k-0 就栽在这：`Gloo connectFullMesh failed … remote=[127.0.1.1]`）。
    各节点网卡名可能不同（不同子网），所以按 IP 在每台上现查，而不是写死一个名字。
    """
    ip = shlex.quote(host)
    return (f"P2P_IF=$(ip -o -4 addr show 2>/dev/null | awk -v ip={ip} "
            "'{split($4,a,\"/\"); if (a[1]==ip) {print $2; exit}}'); "
            f"[ -n \"$P2P_IF\" ] || {{ echo \"本机没有地址为 {host} 的网卡（ip -o -4 addr 查不到）\" >&2; exit 3; }}; "
            "export GLOO_SOCKET_IFNAME=$P2P_IF NCCL_SOCKET_IFNAME=$P2P_IF; "
            f"export VLLM_HOST_IP={ip}; ")


def remote_start(cmd: str, channel: str, node: str, env: dict[str, str],
                 host: str | None = None) -> str:
    """远端执行的 shell：后台起 vllm serve，输出写日志，ssh 断开后进程不退。

    `cmd` 原样放进 `bash -c`（里面有 `$(cat overrides.json)`，要由远端 bash 展开）。
    `cmd` 以 `VLLM_PP_LAYER_PARTITION=… vllm serve …` 开头 —— `exec VAR=值 命令` 在 bash 里
    不成立（VAR=值 会被当成命令名），所以用 `exec env …`：进程直接替换成 vllm，pgrep 只见一个。
    """
    # 等同于在远端 activate 了 venv：vllm 本身用绝对路径已经足够，但它运行中按名字调用的
    # 工具（比如个别算子现场编译时找 ninja）要从 PATH 里找，venv 的 bin/ 得在 PATH 上
    venv = venv_of(cmd)
    act = (f"export VIRTUAL_ENV={shlex.quote(venv)}; export PATH={shlex.quote(venv + '/bin')}:$PATH; "
           if venv else "")
    # 网卡探测在外层做（失败时 ssh 直接返回非 0、报错可见）；--env 给的同名变量排在后面，覆盖探测结果
    net = remote_net(host) if host else ""
    exports = act + "".join(f"export {k}={shlex.quote(v)}; " for k, v in sorted(env.items()))
    log = log_path(channel, node)
    inner = f"{exports}exec env {cmd}"
    # 注意是 `cd /tmp;` 不是 `cd /tmp &&`：`A && B &` 会把整个 `A && B` 放进一个后台子 shell，
    # 它一直占着 ssh 的输出管道，ssh 就要等到 vllm 退出才返回（实测卡死过）。
    # setsid：vllm 自成一个进程组（pgid = 它的 pid）。它起的 EngineCore / worker 子进程都在这个组里，
    # stop 时按组杀 —— 只按命令行 pkill 的话，子进程（命令行里没有 --served-model-name）可能
    # 变成孤儿、继续占着显存，下一次启动就在预热时 OOM（10-13 实测踩到）。
    return (f"cd /tmp || exit 1; {net}setsid nohup bash -c {shlex.quote(inner)} > {shlex.quote(log)} "
            f"2>&1 < /dev/null & echo $!")


def remote_running(channel: str) -> str:
    return f"pgrep -f -- {shlex.quote(match_pattern(channel))} || true"


ERROR_RE = (r"(Error|ERROR|Exception|error:|Traceback|Killed|out of memory|OutOfMemory|"
            r"[Tt]imed? ?out|NCCL WARN|refused|Unable|cannot|Cannot|not found|No such)")


def remote_errors(channel: str, node: str, n: int = 40) -> str:
    """日志里的错误行（去掉调用栈里的 `File "…"` 帧，只留真正说明原因的那些行）。

    vLLM 多进程启动失败时，API 服务器最后打印的永远是同一句
    `Engine core initialization failed. See root cause above` —— 根因在更前面，
    而且常常在**别的**级（headless worker）的日志里。所以要每一级都筛一遍。
    """
    log = shlex.quote(log_path(channel, node))
    return (f"[ -f {log} ] || {{ echo '  （没有日志）'; exit 0; }}; "
            f"grep -n -E {shlex.quote(ERROR_RE)} {log} | grep -v -E 'File \"' | head -n {int(n)}")


def remote_stop(channel: str) -> str:
    """按**进程组**停：vllm 主进程连同它的子进程（EngineCore、worker）一起。"""
    pat = shlex.quote(match_pattern(channel))
    return (f"pids=$(pgrep -f -- {pat}); [ -n \"$pids\" ] || exit 0; "
            f"pg=$(ps -o pgid= -p $(echo $pids | tr ' ' ,) | tr -d ' ' | sort -u); "
            f"for g in $pg; do kill -TERM -- -$g 2>/dev/null; done; pkill -TERM -f -- {pat}; "
            f"for i in $(seq 25); do a=0; for g in $pg; do pgrep -g $g >/dev/null && a=1; done; "
            f"pgrep -f -- {pat} >/dev/null && a=1; [ $a = 0 ] && exit 0; sleep 1; done; "
            f"for g in $pg; do kill -KILL -- -$g 2>/dev/null; done; pkill -KILL -f -- {pat}; true")


GPU_Q = ("nvidia-smi -i 0 --query-gpu=memory.used,memory.total --format=csv,noheader,nounits; "
         "echo ---; nvidia-smi -i 0 --query-compute-apps=pid,used_memory,process_name "
         "--format=csv,noheader,nounits; echo ---; "
         "for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader); do "
         "ps -o pid=,etime=,args= -p $p 2>/dev/null | cut -c1-160; done")


def parse_gpu(out: str) -> dict:
    """GPU_Q 的输出 → {used_mb, total_mb, apps: [(pid, mb, name)], ps: [行]}。"""
    parts = (out + "\n---\n---").split("---")
    used = total = None
    for ln in parts[0].splitlines():
        xs = [x.strip() for x in ln.split(",")]
        if len(xs) >= 2 and xs[0].isdigit():
            used, total = int(xs[0]), int(xs[1])
    apps = []
    for ln in parts[1].splitlines():
        xs = [x.strip() for x in ln.split(",")]
        if len(xs) >= 3 and xs[0].isdigit():
            apps.append((int(xs[0]), int(xs[1]) if xs[1].isdigit() else 0, xs[2]))
    return {"used_mb": used, "total_mb": total, "apps": apps,
            "ps": [ln.strip() for ln in parts[2].splitlines() if ln.strip()]}


def gpu_line(node: str, host: str, g: dict) -> str:
    if g["used_mb"] is None:
        return f"    {node:4s} {host:15s} 读不到 nvidia-smi"
    s = f"    {node:4s} {host:15s} 已用 {g['used_mb']:>6} / {g['total_mb']} MiB"
    if g["apps"]:
        s += "  进程 " + ", ".join(f"{p}({mb}MiB)" for p, mb, _ in g["apps"])
    return s


# --------------------------------------------------------------------------- #
def _ssh(host: str, user: str, script: str, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run([*SSH, f"{user}@{host}", script], capture_output=True, text=True,
                          timeout=timeout)


def _health(api: str, timeout: float = 3.0) -> bool:
    base = api.rsplit("/v1", 1)[0]
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _load(path: str) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"找不到 {p} —— 先用 vllm/p2p/layout.py --hosts … --out {p.parent} 生成")
    chans = json.loads(p.read_text(encoding="utf-8"))["channels"]
    if not all("commands" in c for c in chans):
        raise SystemExit(f"{p} 里没有启动命令 —— 生成时要给 layout.py 传 --hosts")
    return chans


def _pick(chans: list[dict], name: str) -> list[dict]:
    if name == "all":
        return chans
    got = [c for c in chans if c["name"] == name]
    if not got:
        raise SystemExit(f"没有通道 {name}；有的是 {[c['name'] for c in chans]}")
    return got


def _kv_lines(host: str, user: str, channel: str, node: str) -> list[str]:
    r = _ssh(host, user, f"grep -E 'KV cache memory|Maximum concurrency|GPU KV cache size' "
                         f"{shlex.quote(log_path(channel, node))} 2>/dev/null | tail -3")
    return [l.strip() for l in r.stdout.splitlines() if l.strip()]


def cmd_start(c: dict, user: str, env: dict[str, str], wait: float,
              force: bool = False) -> int:
    name = c["name"]
    for st in c["commands"]:
        r = _ssh(st["host"], user, remote_running(name))
        if r.stdout.strip():
            print(f"  ✗ {st['node']} 上 {name} 已经在跑（pid {r.stdout.split()}）—— 先 stop")
            return 1
    # 显存预检：卡上已经有别的进程（上一次没停干净的 worker、识别服务、对拍……）时，vLLM 照样按
    # 「比例 × 整卡」算 KV，然后在预热时 OOM —— 报错只说 OOM，不说是谁占的。这里先查出来。
    busy = []
    shared = c.get("shared_gpu") or {}
    for st in c["commands"]:
        g = parse_gpu(_ssh(st["host"], user, GPU_Q).stdout)
        if g["used_mb"] is not None and g["used_mb"] > 1024:
            if st["node"] in shared and g["total_mb"]:
                # 多级：这张卡本来就要放几个引擎（colorplan 已按比例分了显存）。只要剩下的够本引擎
                # 的比例就放行；不够多半是别的东西占着
                mine = float((c.get("gpu_mem") or {}).get(st["node"], 0.9))
                free = 1.0 - g["used_mb"] / g["total_mb"]
                ok = free >= mine
                print(f"  {'·' if ok else '⚠'} {st['node']} 与 {shared[st['node']]} 同卡：已用 "
                      f"{g['used_mb']}/{g['total_mb']} MiB，空闲 {free:.2f}，本引擎要 {mine:g}")
                if ok:
                    continue
            busy.append((st, g))
    if busy:
        print("  ⚠ 这些卡上已经有显存被占着（vLLM 按「比例 × 整卡」算 KV，不会扣掉别人的，"
              "多半在预热时 OOM）：")
        for st, g in busy:
            print(gpu_line(st["node"], st["host"], g))
            for ln in g["ps"]:
                print(f"         {ln}")
        if not force:
            print("    是没停干净的旧进程就 kill 掉；确实要和别的进程同卡（如 N01 上的识别服务），"
                  "确认显存比例让出了位置后加 --force")
            return 1
    # 先起 headless 的各级，最后起 node_rank 0（它起 API 服务、等各级连上）
    for st in sorted(c["commands"], key=lambda s: "--node-rank 0 " in s["cmd"]):
        r = _ssh(st["host"], user, remote_start(st["cmd"], name, st["node"], env, st["host"]))
        if r.returncode != 0:
            print(f"  ✗ {st['node']}@{st['host']} 启动失败：{r.stderr.strip()[:300]}")
            return 1
        print(f"  → {st['node']}@{st['host']} 已启动 pid {r.stdout.strip()}，日志 {log_path(name, st['node'])}")
    if wait <= 0:
        return 0
    print(f"  等 API 就绪（最多 {wait:.0f}s）：{c['api']}")
    t0 = time.time()
    while time.time() - t0 < wait:
        if _health(c["api"]):
            print(f"  ✓ {name} 就绪（{time.time() - t0:.0f}s）")
            head = c["commands"][0]
            for l in _kv_lines(head["host"], user, name, head["node"]):
                print(f"      {l[-160:]}")
            return 0
        for st in c["commands"]:
            if not _ssh(st["host"], user, remote_running(name)).stdout.strip():
                print(f"  ✗ {st['node']} 上的进程退出了。各级日志里的错误行（根因往往在最早的几行、"
                      f"也可能在别的级）：")
                cmd_errors(c, user, 25)
                print(f"  完整日志：python3 {sys.argv[0]} logs {name} -n 300")
                return 1
        time.sleep(10)
    print(f"  ✗ {wait:.0f}s 内没就绪 —— python3 {sys.argv[0]} logs {name}")
    return 1


def cmd_status(c: dict, user: str) -> int:
    name = c["name"]
    up = _health(c["api"])
    print(f"{name:12s} API {'✓ 就绪' if up else '✗ 不通'}  {c['api']}")
    bad = 0
    for st in c["commands"]:
        r = _ssh(st["host"], user, remote_running(name))
        pids = r.stdout.split()
        print(f"    {st['node']:4s} {st['host']:15s} {'运行中 pid ' + ','.join(pids) if pids else '没有进程'}")
        bad += not pids
    if up:
        head = c["commands"][0]
        for l in _kv_lines(head["host"], user, name, head["node"]):
            print(f"      {l[-160:]}")
    return 0 if up and not bad else 1


def cmd_logs(c: dict, user: str, n: int) -> int:
    for st in c["commands"]:
        print(f"===== {c['name']} / {st['node']} @ {st['host']}  {log_path(c['name'], st['node'])}")
        r = _ssh(st["host"], user, f"tail -n {int(n)} {shlex.quote(log_path(c['name'], st['node']))}")
        print(r.stdout or r.stderr)
    return 0


def cmd_errors(c: dict, user: str, n: int) -> int:
    for st in c["commands"]:
        print(f"----- {c['name']} / {st['node']} @ {st['host']}")
        r = _ssh(st["host"], user, remote_errors(c["name"], st["node"], n))
        print(r.stdout.rstrip() or "  （没有错误行）")
    return 0


# --------------------------------------------------------------------------- #
# A4 识别服务
# --------------------------------------------------------------------------- #
IDENT_PATTERN = "[-]-p2p-identifier"


def ident_log(node: str) -> str:
    return f"/tmp/p2p_identifier_{node}.log"


def find_node(chans: list[dict], node: str) -> tuple[dict, dict]:
    """节点 id → (它所在的通道, 那一级的命令条目)。"""
    for c in chans:
        for st in c.get("commands", []):
            if st["node"] == node:
                return c, st
    raise SystemExit(f"channels.json 里没有节点 {node}")


def ident_command(chans: list[dict], cj: dict, node: str, *, port: int, src: str,
                  p2p_framework: str, lr_model: str, tasks: str | None,
                  extra: str = "") -> tuple[str, str]:
    """→ (host, 命令)。解释器取通道命令里 vllm 所在的 venv（同一个 venv 里有 torch/aiohttp）。"""
    c, st = find_node(chans, node)
    if not st["cmd"].split("--node-rank ", 1)[1].startswith("0 "):
        print(f"  ⚠ {node} 不是前段节点（不是 node_rank 0）—— 它本地多半没有层 1..L₀ 的全专家权重")
    venv = venv_of(st["cmd"])
    if venv is None:
        raise SystemExit("通道命令里的 vllm 不是 venv 里的绝对路径，认不出解释器 —— "
                         "layout.py 生成时给 --vllm-bin /home/ubuntu/venvs/vllm-p2p/bin/vllm")
    model_dir = cj.get("model_dir") or shlex.split(st["cmd"].split(" serve ", 1)[1])[0]
    tasks = tasks or ",".join(f"{u}=1" for u in sorted({x["task"] for x in chans}))
    args = [f"{venv}/bin/python", f"{src.rstrip('/')}/vllm/p2p/identifier.py", "--p2p-identifier",
            "--p2p-framework", p2p_framework, "--model-dir", model_dir, "--lr-model", lr_model,
            "--tasks", tasks, "--l0", str(cj["l0"]), "--device", "cuda:0",
            "--dtype", "float16", "--port", str(port)]
    return st["host"], " ".join(shlex.quote(x) for x in args) + (f" {extra}" if extra else "")


def ident_running() -> str:
    return f"pgrep -f -- {shlex.quote(IDENT_PATTERN)} || true"


def ident_stop() -> str:
    pat = shlex.quote(IDENT_PATTERN)
    return (f"pkill -TERM -f -- {pat}; for i in $(seq 15); do pgrep -f -- {pat} >/dev/null || exit 0; "
            f"sleep 1; done; pkill -KILL -f -- {pat}; true")


def _pretty(raw: bytes) -> str:
    """识别服务 /info 的 JSON → 人能读的一行（中文不转义）。"""
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)[:600]
    except Exception:                                         # noqa: BLE001
        return raw.decode("utf-8", "replace")[:600]


def cmd_ident(action: str, node: str, a, chans: list[dict], cj: dict, env: dict) -> int:
    c, st = find_node(chans, node)
    host = st["host"]
    url = f"http://{host}:{a.ident_port}"
    if action == "ident-start":
        if _ssh(host, a.user, ident_running()).stdout.strip():
            print(f"  ✗ {node} 上识别服务已经在跑 —— 先 ident-stop")
            return 1
        host, cmd = ident_command(chans, cj, node, port=a.ident_port, src=a.src,
                                  p2p_framework=a.p2p_framework, lr_model=a.lr_model,
                                  tasks=a.tasks, extra=a.ident_args)
        venv = venv_of(cmd)
        act = (f"export VIRTUAL_ENV={shlex.quote(venv)}; "
               f"export PATH={shlex.quote(venv + '/bin')}:$PATH; ") if venv else ""
        exports = act + "".join(f"export {k}={shlex.quote(v)}; " for k, v in sorted(env.items()))
        inner = f"{exports}exec {cmd}"
        r = _ssh(host, a.user, f"cd /tmp || exit 1; nohup bash -c {shlex.quote(inner)} > "
                               f"{shlex.quote(ident_log(node))} 2>&1 < /dev/null & echo $!")
        if r.returncode != 0:
            print(f"  ✗ 启动失败：{r.stderr.strip()[:300]}")
            return 1
        print(f"  → {node}@{host} 识别服务 pid {r.stdout.strip()}，日志 {ident_log(node)}")
        t0 = time.time()
        while time.time() - t0 < a.wait:
            try:
                with urllib.request.urlopen(f"{url}/info", timeout=3) as resp:
                    print(f"  ✓ 就绪（{time.time() - t0:.0f}s）：{_pretty(resp.read())}")
                    print(f"    网关：python vllm/p2p/gateway.py --mode lr --identifier {url} …")
                    return 0
            except Exception:                                 # noqa: BLE001
                pass
            if not _ssh(host, a.user, ident_running()).stdout.strip():
                print("  ✗ 进程退出了，日志最后几行：")
                print(_ssh(host, a.user, f"tail -n 30 {shlex.quote(ident_log(node))}").stdout)
                return 1
            time.sleep(5)
        print(f"  ✗ {a.wait:.0f}s 内没就绪 —— python3 {sys.argv[0]} ident-logs {node}")
        return 1
    if action == "ident-status":
        pids = _ssh(host, a.user, ident_running()).stdout.split()
        up = False
        try:
            with urllib.request.urlopen(f"{url}/info", timeout=3) as resp:
                info = _pretty(resp.read())
                up = True
        except Exception:                                     # noqa: BLE001
            info = ""
        print(f"识别服务 {node}@{host}  {'✓ ' + url if up else '✗ 不通'}  "
              f"{'pid ' + ','.join(pids) if pids else '没有进程'}")
        if info:
            print(f"    {info[:400]}")
        return 0 if up else 1
    if action == "ident-logs":
        print(_ssh(host, a.user, f"tail -n {int(a.n)} {shlex.quote(ident_log(node))}").stdout)
        return 0
    _ssh(host, a.user, ident_stop(), timeout=60)
    print(f"  ■ {node}@{host} 识别服务已停")
    return 0


def cmd_stop(c: dict, user: str) -> int:
    for st in c["commands"]:
        _ssh(st["host"], user, remote_stop(c["name"]), timeout=90)
        g = parse_gpu(_ssh(st["host"], user, GPU_Q).stdout)
        left = (f"  ⚠ 卡上还占着 {g['used_mb']} MiB：{g['ps'] or g['apps']}"
                if g["used_mb"] and g["used_mb"] > 1024 else "")
        print(f"  ■ {st['node']}@{st['host']} 已停{left}")
    return 0


def cmd_gpu(c: dict, user: str) -> int:
    print(f"{c['name']}")
    for st in c["commands"]:
        g = parse_gpu(_ssh(st["host"], user, GPU_Q).stdout)
        print(gpu_line(st["node"], st["host"], g))
        for ln in g["ps"]:
            print(f"         {ln}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("start", "status", "logs", "errors", "stop", "gpu",
                                       "ident-start", "ident-status", "ident-logs", "ident-stop"))
    ap.add_argument("channel", help="通道名（如 gsm8k-0），或 all；ident-* 时是节点 id（如 N01）")
    ap.add_argument("--channels", default=DEFAULT_CHANNELS, help="layout.py 生成的 channels.json")
    ap.add_argument("--user", default="ubuntu", help="ssh 用户")
    ap.add_argument("--env", action="append", default=[], metavar="K=V",
                    help="额外环境变量，如 NCCL_SOCKET_IFNAME=eth0（多网卡时 NCCL 选错网卡会卡在初始化）")
    ap.add_argument("--wait", type=float, default=900, help="start 后等 API 就绪的秒数（0 = 不等）")
    ap.add_argument("-n", type=int, default=40, help="logs 显示的行数")
    ap.add_argument("--force", action="store_true",
                    help="start 时卡上已有别的进程占显存也照样起（如 N01 与识别服务同卡）")
    g = ap.add_argument_group("识别服务（ident-*）")
    g.add_argument("--ident-port", type=int, default=8100)
    g.add_argument("--src", default="/home/ubuntu/vllm-p2p", help="节点上 vllm-p2p 源码目录")
    g.add_argument("--p2p-framework", default="/home/ubuntu/p2p-framework")
    g.add_argument("--lr-model",
                   default="/home/ubuntu/p2p-framework/selector/selector_body_L8_params.npz")
    g.add_argument("--tasks", default=None, help="默认 = 通道里出现的 task，先验各 1")
    g.add_argument("--ident-args", default="", help="原样追加给 identifier.py 的参数")
    a = ap.parse_args(argv)
    env = dict(kv.split("=", 1) for kv in a.env)
    if a.action.startswith("ident-"):
        p = Path(a.channels)
        if not p.is_file():
            raise SystemExit(f"找不到 {p}")
        cj = json.loads(p.read_text(encoding="utf-8"))
        return cmd_ident(a.action, a.channel, a, _load(a.channels), cj, env)
    rc = 0
    for c in _pick(_load(a.channels), a.channel):
        if a.action == "start":
            print(f"== 启动 {c['name']}（{' → '.join(c['nodes'])}，PP={','.join(map(str, c['partition']))}，"
                  f"dtype={c['dtype']}）")
            rc |= cmd_start(c, a.user, env, a.wait, a.force)
        elif a.action == "status":
            rc |= cmd_status(c, a.user)
        elif a.action == "logs":
            rc |= cmd_logs(c, a.user, a.n)
        elif a.action == "errors":
            rc |= cmd_errors(c, a.user, a.n)
        elif a.action == "gpu":
            rc |= cmd_gpu(c, a.user)
        else:
            print(f"== 停止 {c['name']}")
            rc |= cmd_stop(c, a.user)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
