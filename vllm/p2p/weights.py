# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""控制机：按分段引擎清单（stages.json）核对 / 下载**每台节点本地需要**的权重，显示各节点进度。**纯标准库。**

    python3 vllm/p2p/weights.py check  --stages p2p_channels/stages_color.json      # 只看，不改
    python3 vllm/p2p/weights.py fetch  --stages p2p_channels/stages_color.json      # 修复 + 下载，进度表
    python3 vllm/p2p/weights.py status --stages p2p_channels/stages_color.json --watch 10   # 另开终端看进度

每台节点「本地需要」什么由清单推出（与 `qwen3_moe_partial.py` 各级实际加载的一致）：

* 前段卡：第 [0, L₀) 层（全部专家）+ embed_tokens；
* 后段流水线上的卡：自己那段层（按 `VLLM_PP_LAYER_PARTITION` 切），只要该 task 在这些层的
  驻留专家（命令里 hf_overrides 的 `p2p_layer_experts`）；流水线末级再加 final norm + lm_head。

`fetch` 在所有节点上同时跑 `weights_node.py`（脚本经 ssh 的标准输入送过去，节点上不用同步代码）：
模型不对 → 删掉权重分片与模型文件、重下；要的分片缺了 / 大小不对 / sha256 不对 → 删掉重下；
用不到的分片删掉（各节点只留 mbf 要的；`--keep-extra` 留着）；最后 index 改成只列留下的分片。下载中每 `--interval` 秒刷新一张进度表
（已下 / 要下、百分比、速度、剩余时间、分片数、正在下的文件），与 Helix 的 `cluster_ctl.py weights` 同一种做法；
Ctrl-C 只是不再等，重跑 `fetch` 会接着下（已下完、校验过的分片不重下）。各节点日志在 `--log-dir`。

节点间互拷（默认开，`--no-peers` 关）：fetch 先核对全部节点，某台缺的分片如果别的节点上有（它要的、
或它用不到但还没删的），先经控制机中转拷过去（`ssh 来源 cat | ssh 目标 cat`，不用节点之间的 ssh 信任），
拷来的照常算 sha256；集群里没有的才从源下。用不到的分片要等本节点要的都齐了才删，下载失败时它们还在。
HF 的 Xet 下载连不上时自动改走普通 HTTPS（也可以一开始就加 `--no-xet`）。

下载源：默认 HuggingFace（控制机上的 HF_ENDPOINT / HF_TOKEN 会带到节点上，可以用镜像）；
节点上没有外网时用 `--source dir:<目录>`（每台节点都能看到的完整 checkpoint，如 NFS）。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

HERE = Path(__file__).resolve().parent
NODE_SCRIPT = HERE / "weights_node.py"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new"]
PROGRESS = ".p2p_weights_progress.json"
OK_MARK = "P2P_WEIGHTS_OK"
REPORT_MARK = "P2P_WEIGHTS_REPORT"

__all__ = ["node_specs", "print_progress_table", "print_check_table", "main"]


# --------------------------------------------------------------------------- #
# 清单 → 每台节点要什么
# --------------------------------------------------------------------------- #
def _cmd_parts(cmd: str) -> tuple[dict, dict]:
    """agent 命令 → (参数, hf_overrides)。"""
    head, _, ov = cmd.partition(" --hf-overrides ")
    toks = shlex.split(head)
    args, env = {}, {}
    for t in toks:
        if "=" in t and t.split("=", 1)[0].isupper():
            k, v = t.split("=", 1)
            env[k] = v
    for i, t in enumerate(toks):
        if t.startswith("--") and i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            args[t] = toks[i + 1]
    args["_env"] = env
    args["_argv0"] = next((t for t in toks if "=" not in t or not t.split("=", 1)[0].isupper()), "")
    overrides = json.loads(shlex.split(ov)[0]) if ov.strip() else {}
    return args, overrides


def node_specs(sj: Mapping) -> dict[str, dict]:
    """stages.json → {节点: {"host", "engine", "python", "spec"}}；spec 见 weights_node.py。"""
    n_layers = int(sj["n_layers"])
    out: dict[str, dict] = {}
    for c in sj["channels"]:
        nodes = list(c["nodes"])
        for st in c["commands"]:
            v = st["node"]
            args, ov = _cmd_parts(st["cmd"])
            lo, hi = (int(x) for x in args["--layers"].split(","))
            rank = int(args.get("--node-rank", 0))
            part = args["_env"].get("VLLM_PP_LAYER_PARTITION")
            if part and len(nodes) > 1:
                sizes = [int(x) for x in part.split(",")]
                if len(sizes) != len(nodes) or sum(sizes) != hi - lo:
                    raise ValueError(f"{c['name']}：切层 {sizes} 与节点 {nodes}、层 [{lo},{hi}) 对不上")
                a = lo + sum(sizes[:rank])
                b = a + sizes[rank]
            else:
                a, b = lo, hi
            role = args.get("--role", c.get("role"))
            le = ov.get("p2p_layer_experts") or {}
            experts = {k: list(v_) for k, v_ in le.items() if a <= int(k) < b}
            spec = {"node": v, "layers": [[a, b]],
                    "embed": role == "front" and rank == 0 and a == 0,
                    "head": role != "front" and rank == len(nodes) - 1 and b == n_layers,
                    "experts": experts}
            prev = out.get(v)
            if prev:                              # 同一台卡在两个引擎里（一般不会）：合并
                ps = prev["spec"]
                ps["layers"] += spec["layers"]
                ps["embed"] |= spec["embed"]
                ps["head"] |= spec["head"]
                ps["experts"].update(experts)
                prev["engine"] += f"+{c['name']}"
                continue
            argv0 = args["_argv0"]
            out[v] = {"host": st["host"], "engine": c["name"], "spec": spec,
                      "python": argv0 if argv0.endswith(("/python", "/python3")) else None}
    return out


def _encode(spec: dict) -> str:
    return base64.b64encode(json.dumps(spec, separators=(",", ":")).encode("utf-8")).decode("ascii")


def _env_prefix() -> str:
    return " ".join(f"{k}={shlex.quote(os.environ[k])}" for k in ("HF_ENDPOINT", "HF_TOKEN", "HF_HUB_DISABLE_XET")
                    if os.environ.get(k))


def node_command(info: dict, a) -> str:
    py = a.python or info.get("python") or "python3"
    args = ["--dir", a.dir, "--spec", _encode(info["spec"]), "--source", a.source,
            "--repo", a.repo, "--revision", a.revision]
    for flag, on in (("--keep-extra", getattr(a, "keep_extra", False)), ("--clean", getattr(a, "clean", False)),
                     ("--no-verify", a.no_verify), ("--curl", a.curl), ("--no-xet", a.no_xet)):
        if on:
            args.append(flag)
    if getattr(a, "check_only", False):
        args.append("--check")
    env = _env_prefix()
    return f"{env + ' ' if env else ''}{shlex.quote(py)} - {' '.join(shlex.quote(x) for x in args)}"


def _ssh_argv(user: str, host: str, cmd: str) -> list[str]:
    return [*SSH, f"{user}@{host}", cmd]


# --------------------------------------------------------------------------- #
# 进度表
# --------------------------------------------------------------------------- #
def poll_progress(nodes: Mapping[str, dict], user: str, wdir: str, timeout: float = 30) -> dict:
    """节点 → (进度 dict 或 None, 目录字节数或 None)，并行 ssh。"""
    q = shlex.quote(wdir)
    cmd = f"cat {q}/{PROGRESS} 2>/dev/null || echo '{{}}'; echo; du -sb {q} 2>/dev/null | cut -f1"
    procs = {v: subprocess.Popen(_ssh_argv(user, i["host"], cmd), stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, text=True)
             for v, i in nodes.items()}
    out = {}
    for v, p in procs.items():
        try:
            text, _ = p.communicate(timeout=timeout)
            first, _, rest = text.strip().partition("\n")
            out[v] = (json.loads(first) or None, int(rest.strip()) if rest.strip().isdigit() else None)
        except Exception:  # noqa: BLE001
            p.kill()
            out[v] = (None, None)
    return out


def _dur(s: float) -> str:
    s = int(max(0, s))
    return f"{s // 3600}h{s // 60 % 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def print_progress_table(nodes: Mapping[str, dict], polled: Mapping, prev: dict, *, interval: float = 0,
                         status: Mapping[str, str] | None = None, now: float | None = None,
                         clear: bool | None = None, file=None) -> dict:
    """刷新一次进度表。prev：节点 → (时刻, 目录字节数)，用来算速度（这里更新）。返回合计。"""
    file = file or sys.stdout
    now = time.time() if now is None else now
    if clear if clear is not None else file.isatty():
        print("\033[H\033[J", end="", file=file)
    print(f"[{time.strftime('%H:%M:%S', time.localtime(now))}] 各节点权重下载进度"
          + (f"（每 {interval:g}s 刷新；Ctrl-C 只是不再等，重跑 fetch 会接着下）" if interval else ""), file=file)
    print(f"  {'节点':<5} {'主机':<15} {'引擎':<20} {'状态':<10} {'已下/要下':>15} {'%':>5} {'速度':>10} "
          f"{'剩余':>7} {'分片':>6}  正在", file=file)
    tot_done = tot_need = tot_speed = 0.0
    for v, info in nodes.items():
        prog, size = polled.get(v, (None, None))
        st = (status or {}).get(v)
        head = f"  {v:<5} {info['host']:<15} {info['engine'][:20]:<20}"
        if not prog:
            print(f"{head} {st or ('连不上' if size is None else '未开始'):<10}", file=file)
            continue
        need = float(prog.get("need") or 0)
        state = prog.get("state", "?")
        done = need if state == "done" else max(0.0, min(need, float(size or 0) - float(prog.get("base") or 0)))
        speed = 0.0
        if v in prev and size is not None and now > prev[v][0]:
            speed = max(0.0, (size - prev[v][1]) / (now - prev[v][0]))
        elif state == "downloading" and now > float(prog.get("started") or now):
            speed = done / (now - float(prog["started"]))
        if size is not None:
            prev[v] = (now, size)
        if st == "FAIL" or state == "failed":
            state = "失败"
        label = {"done": "完成", "downloading": "下载中", "checking": "核对中"}.get(state, state)
        pct = 100.0 * done / need if need else (100.0 if state == "done" else 0.0)
        eta = _dur((need - done) / speed) if state == "downloading" and speed > 1e5 else ""
        cur = str(prog.get("current") or "").replace(".safetensors", "")
        print(f"{head} {label:<10} {done / 1e9:6.1f}/{need / 1e9:5.1f} GB {pct:5.0f} "
              f"{(speed if state == 'downloading' else 0) / 1e6:6.0f} MB/s {eta:>7} "
              f"{prog.get('shards_done', 0):>2}/{prog.get('shards_total', 0):<3}  {cur}"
              + (f"  {str(prog.get('error', ''))[:60]}" if label == "失败" else ""), file=file)
        tot_done += done
        tot_need += need
        tot_speed += speed if state == "downloading" else 0.0
    pct = 100.0 * tot_done / tot_need if tot_need else 0.0
    eta = _dur((tot_need - tot_done) / tot_speed) if tot_speed > 1e5 else ""
    print(f"  {'合计':<43} {'':<10} {tot_done / 1e9:6.1f}/{tot_need / 1e9:5.1f} GB {pct:5.0f} "
          f"{tot_speed / 1e6:6.0f} MB/s {eta:>7}", file=file, flush=True)
    return {"done": tot_done, "need": tot_need, "speed": tot_speed}


def print_check_table(nodes: Mapping[str, dict], reports: Mapping[str, dict | str], file=None) -> list[str]:
    """check 的结果表；返回需要处理的节点。"""
    file = file or sys.stdout
    print(f"  {'节点':<5} {'引擎':<20} {'层(0-based)':<12} {'要的分片':>8} {'已有':>5} {'待校验':>6} {'缺/坏':>6}"
          f" {'要下':>9} {'用不到':>12} {'空闲':>9}  结论", file=file)
    bad = []
    for v, info in nodes.items():
        r = reports.get(v)
        lay = "+".join(f"{a}-{b}" for a, b in info["spec"]["layers"]) + ("·E" if info["spec"]["embed"] else "") \
            + ("·H" if info["spec"]["head"] else "")
        if not isinstance(r, dict):
            print(f"  {v:<5} {info['engine'][:20]:<20} {lay:<12} {'':>8}  （{r or '没有结果'}）", file=file)
            bad.append(v)
            continue
        stv = list(r["status"].values())
        ok, unv = stv.count("ok"), stv.count("unverified")
        miss = stv.count("missing") + stv.count("bad_size")
        if r.get("wrong_model"):
            verdict = f"模型不对：{r['wrong_model'][:40]} → 清空重下"
        elif miss:
            verdict = "要下载"
        elif unv:
            verdict = "要校验 sha256"
        elif r["extra"] and r.get("prune", True):
            verdict = "要删用不到的分片"
        else:
            verdict = "就绪"
        if verdict != "就绪":
            bad.append(v)
        extra = f"{len(r['extra'])}（{r['extra_bytes'] / 1e9:.1f}G）" if r["extra"] else "0"
        print(f"  {v:<5} {info['engine'][:20]:<20} {lay:<12} {r['shards_needed']:>4}/{r['shards_total']:<3}"
              f" {ok:>5} {unv:>6} {miss:>6} {r['need_bytes'] / 1e9:>7.1f}GB {extra:>12}"
              f" {r['free_bytes'] / 1e9:>7.0f}GB  {verdict}", file=file)
    print("  层后缀：·E = 还要 embed_tokens，·H = 还要 final norm + lm_head；「用不到」的分片 fetch 时删掉（--keep-extra 留着）",
          file=file, flush=True)
    return bad


# --------------------------------------------------------------------------- #
def _select(nodes: dict, only: str | None) -> dict:
    if not only:
        return nodes
    want = {x.strip() for x in only.split(",") if x.strip()}
    miss = want - set(nodes)
    if miss:
        raise SystemExit(f"清单里没有节点 {sorted(miss)}（有 {sorted(nodes)}）")
    return {v: i for v, i in nodes.items() if v in want}


def collect_reports(nodes: dict, a) -> dict:
    """各节点同时跑一遍只读核对 → {节点: 报告 dict 或错误文本}。"""
    saved = getattr(a, "check_only", False)
    a.check_only = True
    script = NODE_SCRIPT.read_text(encoding="utf-8")
    procs = {}
    for v, i in nodes.items():
        p = subprocess.Popen(_ssh_argv(a.user, i["host"], node_command(i, a)), stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            p.stdin.write(script)
            p.stdin.close()
        except BrokenPipeError:
            pass
        procs[v] = p
    reports: dict = {}
    t_end = time.time() + getattr(a, "timeout", 1800.0)
    for v, p in procs.items():
        try:
            out, _ = p.communicate(timeout=max(1.0, t_end - time.time()))
        except subprocess.TimeoutExpired:
            p.kill()
            reports[v] = "超时"
            continue
        line = next((ln for ln in out.splitlines() if ln.startswith(REPORT_MARK)), None)
        reports[v] = (json.loads(line[len(REPORT_MARK):]) if line else
                      (out.strip().splitlines() or ["没有输出"])[-1][:120])
    a.check_only = saved
    return reports


# --------------------------------------------------------------------------- #
# 节点间互拷：缺的分片先从集群里已经有它的节点拷（经控制机中转，不用节点之间的 ssh 信任），
# 拷不到的才从 HF 下。拷来的分片在节点上是「待校验」，节点脚本照常算 sha256，不对就删掉重下。
# --------------------------------------------------------------------------- #
def plan_peer_copies(targets, reports: Mapping[str, object], *, reserve_gb: float = 2.0) -> tuple[list, list]:
    """→ ([(目标, 分片, 来源, 字节数或 None)], [拷不到的 (目标, 分片)])。来源优先选 sha 已校验过的，
    再选大小对的，再选「用不到」的那些；同一个来源尽量少分。目标磁盘放不下的不拷（交给 HF 下载）。"""
    rank = {"ok": 0, "unverified": 1}
    have: dict = {}
    for v, r in reports.items():
        if not isinstance(r, dict) or r.get("wrong_model"):
            continue
        for s, st in (r.get("status") or {}).items():
            if st in rank:
                have.setdefault(s, []).append((rank[st], v))
        for s in r.get("extra") or ():
            have.setdefault(s, []).append((2, v))
    load: dict = {}
    plan, none = [], []
    for t in targets:
        r = reports.get(t)
        if not isinstance(r, dict) or r.get("wrong_model"):
            continue
        room = r.get("free_bytes", 0) - reserve_gb * 1e9
        for s, st in sorted((r.get("status") or {}).items()):
            if st not in ("missing", "bad_size"):
                continue
            cands = [(k, v) for k, v in have.get(s, []) if v != t]
            size = (r.get("sizes") or {}).get(s)
            if not cands:
                none.append((t, s))
                continue
            if size and size > room:
                none.append((t, s))
                continue
            k, v = min(cands, key=lambda kv: (kv[0], load.get(kv[1], 0), kv[1]))
            load[v] = load.get(v, 0) + 1
            room -= size or 4e9
            plan.append((t, s, v, size))
    return plan, none


def peer_copy_cmds(wdir: str, shard: str) -> tuple[str, str]:
    """（来源上执行的读命令, 目标上执行的写命令）；写到 .peer 再改名，失败不留半截。"""
    src = shlex.quote(os.path.join(wdir, shard))
    dst = shlex.quote(os.path.join(wdir, shard))
    tmp = shlex.quote(os.path.join(wdir, shard + ".peer"))
    return (f"cat -- {src}",
            f"mkdir -p {shlex.quote(wdir)} && cat > {tmp} && mv -f {tmp} {dst} || {{ rm -f {tmp}; exit 1; }}")


def run_peer_copies(plan: list, all_nodes: Mapping[str, dict], a) -> list:
    """按计划拷（并发 a.peer_parallel 个，每个来源同时最多 2 个）→ 失败的 (目标, 分片)。"""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    sem = {v: threading.Semaphore(2) for v in {p[2] for p in plan}}
    lock = threading.Lock()
    done = {"n": 0, "bytes": 0}
    failed = []
    t0 = time.time()

    def one(item):
        t, s, v, size = item
        rd, wr = peer_copy_cmds(a.dir, s)
        with sem[v]:
            t1 = time.time()
            src = subprocess.Popen(_ssh_argv(a.user, all_nodes[v]["host"], rd), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
            dst = subprocess.run(_ssh_argv(a.user, all_nodes[t]["host"], wr), stdin=src.stdout,
                                 capture_output=True, text=True)
            src.stdout.close()
            src.wait()
            ok = src.returncode == 0 and dst.returncode == 0
            if ok and size:
                chk = subprocess.run(_ssh_argv(a.user, all_nodes[t]["host"],
                                               f"stat -c %s {shlex.quote(os.path.join(a.dir, s))}"),
                                     capture_output=True, text=True)
                ok = chk.stdout.strip() == str(size)
            dt = time.time() - t1
        with lock:
            if ok:
                done["n"] += 1
                done["bytes"] += size or 0
                print(f"  ✓ {s}  {v} → {t}  {(size or 0) / 1e9:.1f} GB，{dt:.0f}s"
                      f"（{(size or 0) / max(dt, 1e-3) / 1e6:.0f} MB/s）  [{done['n']}/{len(plan)}]", flush=True)
            else:
                err = (src.stderr.read().decode("utf-8", "replace") if src.stderr else "") + dst.stderr
                print(f"  ✗ {s}  {v} → {t}：{err.strip()[-200:] or '大小不对'}", flush=True)
                failed.append((t, s))

    print(f"节点间互拷 {len(plan)} 个分片（经控制机中转，同时 {a.peer_parallel} 个）…", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, a.peer_parallel)) as ex:
        list(ex.map(one, plan))
    dt = time.time() - t0
    print(f"互拷完成 {done['n']}/{len(plan)}，{done['bytes'] / 1e9:.1f} GB，{dt:.0f}s"
          f"（{done['bytes'] / max(dt, 1e-3) / 1e6:.0f} MB/s）", flush=True)
    return failed


def cmd_check(nodes: dict, a) -> int:
    a.check_only = True
    script = NODE_SCRIPT.read_text(encoding="utf-8")
    procs = {}
    for v, i in nodes.items():                   # 先全部拉起、送完脚本，各节点同时核对
        p = subprocess.Popen(_ssh_argv(a.user, i["host"], node_command(i, a)), stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            p.stdin.write(script)
            p.stdin.close()
        except BrokenPipeError:
            pass
        procs[v] = p
    reports: dict = {}
    print(f"核对 {len(nodes)} 台节点的 {a.dir}（源 {a.source}{'' if a.source != 'hf' else ' ' + a.repo}）…", flush=True)
    t_end = time.time() + a.timeout
    for v, p in procs.items():
        try:
            out, _ = p.communicate(timeout=max(1.0, t_end - time.time()))
        except subprocess.TimeoutExpired:
            p.kill()
            reports[v] = f"超时（{a.timeout:g}s）"
            continue
        line = next((ln for ln in out.splitlines() if ln.startswith(REPORT_MARK)), None)
        reports[v] = (json.loads(line[len(REPORT_MARK):]) if line else
                      (out.strip().splitlines() or ["没有输出"])[-1][:120])
    bad = print_check_table(nodes, reports)
    if a.json:
        Path(a.json).write_text(json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")
    print("全部就绪" if not bad else f"要处理的节点：{bad} —— python3 vllm/p2p/weights.py fetch --stages {a.stages}"
          + (f" --nodes {','.join(bad)}" if len(bad) < len(nodes) else ""))
    return 0 if not bad else 1


def cmd_fetch(nodes: dict, a, all_nodes: Mapping[str, dict] | None = None) -> int:
    a.check_only = False
    script_path = str(NODE_SCRIPT)
    os.makedirs(a.log_dir, exist_ok=True)
    if a.peers and not a.clean:
        pool = dict(all_nodes or nodes)
        print(f"先核对 {len(pool)} 台节点上已有的分片（找能互拷的）…", flush=True)
        reports = collect_reports(pool, a)
        plan, none = plan_peer_copies(list(nodes), reports)
        if plan:
            failed = run_peer_copies(plan, pool, a)
            if failed:
                print(f"  {len(failed)} 个没拷成，交给下面的下载", flush=True)
        else:
            print("  集群里没有可以互拷的分片", flush=True)
        if none:
            by: dict = {}
            for t, s in none:
                by.setdefault(t, []).append(s)
            print("  要从源下载（集群里没有 / 磁盘放不下）：" + "；".join(
                f"{t} {len(ss)} 个" for t, ss in sorted(by.items())), flush=True)
        a.check_only = False
    print(f"{len(nodes)} 台节点 → {a.dir}（源 {a.source}{'' if a.source != 'hf' else ' ' + a.repo}）"
          f"{'，先清空目录' if a.clean else ''}{'，留着用不到的分片' if a.keep_extra else '，删掉用不到的分片'}；"
          f"最多同时 {a.max_parallel} 台",
          flush=True)
    queue = list(nodes)
    running: dict = {}
    status = {v: "排队" for v in nodes}
    failed, prev, last = [], {}, 0.0
    try:
        while queue or running:
            while queue and len(running) < a.max_parallel:
                v = queue.pop(0)
                log = os.path.join(a.log_dir, f"weights_{v}.log")
                fo = open(log, "w", encoding="utf-8")
                fi = open(script_path, encoding="utf-8")
                running[v] = (subprocess.Popen(_ssh_argv(a.user, nodes[v]["host"], node_command(nodes[v], a)),
                                               stdin=fi, stdout=fo, stderr=subprocess.STDOUT, text=True),
                              log, fo, fi)
                status[v] = "running"
            for v in list(running):
                p, log, fo, fi = running[v]
                if p.poll() is None:
                    continue
                fo.close()
                fi.close()
                del running[v]
                ok = p.returncode == 0 and OK_MARK in Path(log).read_text(encoding="utf-8", errors="replace")
                status[v] = "done" if ok else "FAIL"
                if not ok:
                    failed.append(v)
            if time.time() - last >= a.interval or not (queue or running):
                print_progress_table(nodes, poll_progress(nodes, a.user, a.dir), prev, interval=a.interval,
                                     status={v: ("排队" if s == "排队" else s) for v, s in status.items()})
                last = time.time()
            if queue or running:
                time.sleep(2)
    except KeyboardInterrupt:
        print("\n不再等（各节点上的下载可能还在跑；重跑 fetch 会接着下）")
        return 130
    for v in failed:
        log = os.path.join(a.log_dir, f"weights_{v}.log")
        print(f"[失败] {v}（日志 {log}）")
        for ln in Path(log).read_text(encoding="utf-8", errors="replace").strip().splitlines()[-6:]:
            print(f"        | {ln}")
    for v in nodes:
        log = os.path.join(a.log_dir, f"weights_{v}.log")
        if os.path.exists(log):
            got = [ln for ln in Path(log).read_text(encoding="utf-8", errors="replace").splitlines()
                   if ln.startswith("[weights] 要 ")]
            if got:
                print(f"  {v:<5} {got[-1].removeprefix('[weights] ')}")
    print("权重：全部节点就绪" if not failed else f"权重：{failed} 失败")
    return 0 if not failed else 1


def cmd_status(nodes: dict, a) -> int:
    prev: dict = {}
    try:
        while True:
            print_progress_table(nodes, poll_progress(nodes, a.user, a.dir), prev, interval=a.watch or 0)
            if not a.watch:
                return 0
            time.sleep(a.watch)
    except KeyboardInterrupt:
        print()
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, h in (("check", "只核对每台节点的权重目录（不改动）"), ("fetch", "修复 + 下载，显示进度"),
                    ("status", "看各节点的下载进度")):
        p = sub.add_parser(name, help=h)
        p.add_argument("--stages", required=True, help="分段引擎清单（colorplan.py / stages.py 生成）")
        p.add_argument("--dir", default=None, help="节点上的权重目录（默认 = 清单的 model_dir）")
        p.add_argument("--nodes", default=None, help="只处理这些节点，逗号分隔")
        p.add_argument("--user", default="ubuntu")
        p.add_argument("--python", default=None,
                       help="节点上跑 weights_node.py 的 python（默认 = 清单命令里 venv 的；从 HF 下要有 huggingface_hub）")
        p.add_argument("--source", default="hf", help="hf（默认）或 dir:<每台节点都能看到的完整 checkpoint 目录>")
        p.add_argument("--repo", default="Qwen/Qwen3-30B-A3B")
        p.add_argument("--revision", default="main")
        p.add_argument("--no-verify", action="store_true", help="不算 sha256，只比大小")
        p.add_argument("--curl", action="store_true", help="节点上用 curl 下载（可续传）")
        p.add_argument("--no-xet", action="store_true", help="HF 走普通 HTTPS、不走 Xet")
        if name == "check":
            p.add_argument("--timeout", type=float, default=1800.0,
                           help="每台节点核对的时限（秒；要算 sha256 的分片多时会久一点）")
            p.add_argument("--json", default=None, help="把各节点的核对结果写成 JSON")
        if name == "fetch":
            p.add_argument("--keep-extra", action="store_true",
                           help="留着本节点用不到的分片（默认删掉，各节点只留 mbf 要的）")
            p.add_argument("--clean", action="store_true", help="先清空各节点的整个权重目录")
            p.add_argument("--max-parallel", type=int, default=15)
            p.add_argument("--no-peers", dest="peers", action="store_false",
                           help="不先从集群里已有该分片的节点互拷（默认先互拷，拷不到的再从源下）")
            p.add_argument("--peer-parallel", type=int, default=4, help="互拷同时几个分片（经控制机中转）")
            p.add_argument("--interval", type=float, default=10.0, help="进度表刷新间隔（秒）")
            p.add_argument("--log-dir", default="logs")
        if name == "status":
            p.add_argument("--watch", type=float, default=0.0, help="每隔几秒刷新；0 = 只看一次")
    a = ap.parse_args(argv)
    sj = json.loads(Path(a.stages).read_text(encoding="utf-8"))
    a.dir = a.dir or sj.get("model_dir")
    if not a.dir:
        raise SystemExit("清单里没有 model_dir，用 --dir 给")
    all_nodes = node_specs(sj)
    nodes = _select(all_nodes, a.nodes)
    if a.cmd == "check":
        return cmd_check(nodes, a)
    if a.cmd == "fetch":
        return cmd_fetch(nodes, a, all_nodes)
    return cmd_status(nodes, a)


if __name__ == "__main__":
    sys.exit(main())
