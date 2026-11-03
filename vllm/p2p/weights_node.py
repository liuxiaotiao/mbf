# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""在**一台节点**上核对 / 下载本节点要的 safetensors 分片。**只用标准库**（从 HF 下载时用 huggingface_hub）。

通常由控制机上的 `weights.py check|fetch` 经 ssh 拉起（脚本走 ssh 的标准输入，节点上不用先同步代码）；
也可以手动跑：

    python weights_node.py --dir /home/ubuntu/p2p-framework/weights-q3-30b --spec <base64 JSON> --check
    python weights_node.py --dir … --spec … [--keep-extra] [--source hf|dir:/nfs/qwen3-30b] [--curl] [--no-xet]

spec（`weights.py` 按 stages.json 算好、base64 后传进来）::

    {"node": "N05", "layers": [[8, 21]],          # 本节点的层，0-based 半开区间（可多段）
     "embed": false, "head": false,               # 要不要 embed_tokens / final norm + lm_head
     "experts": {"8": [2, 3, …], …}}              # 0-based 层 → 驻留专家；没列的层专家全要

「本地需要」精确到张量：本节点各层的注意力 / 层归一化 / 路由门，驻留专家的权重，前段首级的
embed_tokens，后段末级的 model.norm 与 lm_head（与 `qwen3_moe_partial.py` 各级实际加载的一致）。
分片是整个文件下载的：一个分片里只要有一个要的张量，整个分片都要。

核对（`--check` 只报告，不改动）与修复（默认）：

* **模型不对**：本地 config.json 与源（HF 或 --source dir:）的模型类型 / 层数 / 专家数 / hidden 不同，
  或本地 index 里有源没有的张量、同一张量在不同的分片 → 删掉目录里的权重分片与模型文件（config / index /
  tokenizer、本脚本的记录、下载残留），**其余文件不动**，再整套重下。本地 index 只是源的子集
  （部分 checkpoint）不算不对；
* 已有的 config / index / tokenizer 不覆盖（只补缺的）；
* **要的分片**：不在 → 下载；大小不对或 sha256 不对 → 删掉重下；大小对、还没校验过的 → 算一次
  sha256（结果记在 `.p2p_weights_verified.json`，以后不再算）；
* **用不到的分片**：删掉（目录里只留 mbf 在这台节点上要的；`--keep-extra` 留着）；
* 最后把 model.safetensors.index.json 改成只列留下的分片（vLLM 只加载 index 里出现的分片文件）；
* 残留的 `*.part` / `*.incomplete`：下载前清掉。

下载过程写 `<dir>/.p2p_weights_progress.json`（控制机据此画进度表）；成功时最后打印 P2P_WEIGHTS_OK，
`--check` 打印一行 `P2P_WEIGHTS_REPORT <JSON>`。思路与 Helix 的 tools/fetch_weights.py 相同
（按层取分片、大小 + sha256 校验、断点续传、进度文件），多了张量级的「是否本地需要」与自动修复。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import socket
import sys
import time

SMALL_FILES = ["config.json", "generation_config.json", "model.safetensors.index.json",
               "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
REQUIRED_SMALL = ("config.json", "model.safetensors.index.json")
ID_KEYS = ("model_type", "num_hidden_layers", "num_experts", "hidden_size", "num_experts_per_tok")
PROGRESS = ".p2p_weights_progress.json"
VERIFIED = ".p2p_weights_verified.json"
MANIFEST = ".p2p_weights.json"
OK_MARK = "P2P_WEIGHTS_OK"
REPORT_MARK = "P2P_WEIGHTS_REPORT"
_LAYER = re.compile(r"^model\.layers\.(\d+)\.")
_EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")


# --------------------------------------------------------------------------- #
# 本地需要哪些张量 / 分片
# --------------------------------------------------------------------------- #
def tensor_needed(name: str, spec: dict) -> bool:
    if name.startswith("model.embed_tokens."):
        return bool(spec.get("embed"))
    if name.startswith("model.norm.") or name.startswith("lm_head."):
        return bool(spec.get("head"))
    m = _LAYER.match(name)
    if not m:
        return False
    layer = int(m.group(1))
    if not any(lo <= layer < hi for lo, hi in spec.get("layers") or []):
        return False
    e = _EXPERT.match(name)
    if e is None:
        return True                              # 注意力、层归一化、路由门：都要
    keep = (spec.get("experts") or {}).get(str(layer))
    return keep is None or int(e.group(2)) in set(int(x) for x in keep)


def needed_shards(weight_map: dict, spec: dict) -> tuple[list[str], int]:
    """→ (要的分片, 要的张量数)。"""
    want, n = set(), 0
    for name, shard in weight_map.items():
        if tensor_needed(name, spec):
            want.add(shard)
            n += 1
    return sorted(want), n


# --------------------------------------------------------------------------- #
# 源：HuggingFace（含镜像 HF_ENDPOINT、HF_TOKEN）或一个目录（本地 / NFS 上的完整 checkpoint）
# --------------------------------------------------------------------------- #
class DirSource:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.label = f"dir:{self.path}"

    def fetch_small(self, f: str, dst_dir: str) -> str:
        src = os.path.join(self.path, f)
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        dst = os.path.join(dst_dir, f)
        shutil.copyfile(src, dst + ".part")
        os.replace(dst + ".part", dst)
        return dst

    def sizes(self, files):
        return {f: (os.path.getsize(os.path.join(self.path, f))
                    if os.path.exists(os.path.join(self.path, f)) else None) for f in files}

    def shas(self, files):
        return {f: None for f in files}          # 目录源只比大小（要 sha 就把 HF 当源）

    def download(self, f: str, dst_dir: str, expected_size=None) -> str:
        src, dst = os.path.join(self.path, f), os.path.join(dst_dir, f)
        part = dst + ".part"
        with open(src, "rb") as fi, open(part, "wb") as fo:
            while True:
                b = fi.read(16 << 20)
                if not b:
                    break
                fo.write(b)
        os.replace(part, dst)
        return dst


class HFSource:
    def __init__(self, repo: str, revision: str, curl: bool = False):
        self.repo, self.revision, self.curl = repo, revision, curl
        self.label = f"hf:{repo}@{revision}" + (" (curl)" if curl else "")

    def _url(self, f):
        ep = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        return f"{ep}/{self.repo}/resolve/{self.revision}/{f}"

    def fetch_small(self, f, dst_dir):
        return self.download(f, dst_dir, attempts=3)

    def _infos(self, files):
        from huggingface_hub import HfApi
        try:
            return {i.path: i for i in HfApi().get_paths_info(self.repo, files, revision=self.revision)}
        except Exception:  # noqa: BLE001
            return {}

    def sizes(self, files):
        got = self._infos(files)
        return {f: getattr(got.get(f), "size", None) for f in files}

    def shas(self, files):
        got = self._infos(files)

        def sha(i):
            lfs = getattr(i, "lfs", None)
            if lfs is None:
                return None
            return lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        return {f: (sha(got[f]) if f in got else None) for f in files}

    def download(self, f, dst_dir, expected_size=None, attempts=8):
        if self.curl:
            return self._curl(f, dst_dir, expected_size, attempts=max(attempts, 30))
        from huggingface_hub import hf_hub_download
        for k in range(1, attempts + 1):
            try:
                return hf_hub_download(repo_id=self.repo, filename=f, revision=self.revision,
                                       local_dir=dst_dir)
            except Exception as e:  # noqa: BLE001
                if k == attempts:
                    raise
                wait = min(120, 15 * k)
                print(f"  重试 {k}/{attempts - 1}（{f}，{wait}s 后）：{type(e).__name__}: {str(e)[:200]}",
                      flush=True)
                time.sleep(wait)

    def _curl(self, f, dst_dir, expected_size, attempts):
        import subprocess
        dst = os.path.join(dst_dir, f)
        part = dst + ".part"
        cmd = ["curl", "-fsSL", "--connect-timeout", "30", "--speed-limit", "1000000",
               "--speed-time", "60", "-C", "-", "-o", part, self._url(f)]
        if os.environ.get("HF_TOKEN"):
            cmd[1:1] = ["-H", f"Authorization: Bearer {os.environ['HF_TOKEN']}"]
        for k in range(1, attempts + 1):
            if expected_size and os.path.exists(part) and os.path.getsize(part) >= expected_size:
                break
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                break
            if k == attempts:
                raise RuntimeError(f"curl {f} 失败（exit {r.returncode}）：{r.stderr.strip()[-300:]}")
            time.sleep(min(60, 5 * k))
        os.replace(part, dst)
        return dst


def make_source(a):
    if a.source.startswith("dir:"):
        return DirSource(a.source[4:])
    if a.source != "hf":
        raise SystemExit(f"--source 只能是 hf 或 dir:<目录>，给的是 {a.source!r}")
    return HFSource(a.repo, a.revision, curl=a.curl)


# --------------------------------------------------------------------------- #
def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_size(d: str) -> int:
    tot = 0
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            if not os.path.islink(p):
                try:
                    tot += os.path.getsize(p)
                except OSError:
                    pass
    return tot


def managed(name: str) -> bool:
    """本脚本管的文件：权重分片、模型的小文件、本脚本的记录、下载残留。别的文件不碰。"""
    return (name.endswith((".safetensors", ".part", ".incomplete")) or name in SMALL_FILES
            or name.startswith(".p2p_weights") or name in ("generation_config.json", "special_tokens_map.json"))


def clean_managed(d: str) -> int:
    """「模型不对」时用：只删本脚本管的文件（分片、config / index / tokenizer、记录），其余原样留着。"""
    freed = 0
    if not os.path.isdir(d):
        return 0
    for name in os.listdir(d):
        p = os.path.join(d, name)
        if os.path.isfile(p) and managed(name):
            freed += os.path.getsize(p)
            os.remove(p)
    return freed


def clean_dir(d: str) -> int:
    real = os.path.realpath(d)
    home = os.path.realpath(os.path.expanduser("~"))
    if real in ("/", home) or real.count(os.sep) < 2:
        raise SystemExit(f"拒绝清空 {real}")
    if not os.path.isdir(real):
        return 0
    freed = dir_size(real)
    for name in os.listdir(real):
        p = os.path.join(real, name)
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p)
        else:
            os.remove(p)
    return freed


def _load_json(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def identity_problem(local_dir: str, ref_dir: str) -> str | None:
    """本地 config / index 与源不一致 → 原因；一致或本地还没有 → None。"""
    lc = _load_json(os.path.join(local_dir, "config.json"))
    li = _load_json(os.path.join(local_dir, "model.safetensors.index.json"))
    rc = _load_json(os.path.join(ref_dir, "config.json"))
    ri = _load_json(os.path.join(ref_dir, "model.safetensors.index.json"))
    if lc is None and li is None:
        return None
    if lc is None or li is None:
        return "config.json / index 只有一个"
    diff = [k for k in ID_KEYS if lc.get(k) != rc.get(k)]
    if diff:
        return "config.json 与源不同：" + ", ".join(f"{k} {lc.get(k)}≠{rc.get(k)}" for k in diff)
    lw, rw = li.get("weight_map") or {}, ri.get("weight_map") or {}
    # 本地 index 是源的子集（部分 checkpoint 只列了自己有的张量）没问题；张量在源里没有、
    # 或同一个张量指向不同的分片文件，才说明不是同一个模型 / 同一种切分
    unknown = [n for n in lw if n not in rw]
    moved = [n for n in lw if n in rw and lw[n] != rw[n]]
    if unknown or moved:
        return (f"index 与源不一致：{len(unknown)} 个张量源里没有、{len(moved)} 个张量所在分片不同"
                f"（如 {(unknown or moved)[0]}）")
    return None


def fmt_gb(x) -> str:
    return "?" if x is None else f"{x / 1e9:.1f} GB"


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--spec", required=True, help="base64(JSON)，见文件头")
    ap.add_argument("--source", default="hf", help="hf（默认，HF_ENDPOINT / HF_TOKEN 生效）或 dir:<目录>")
    ap.add_argument("--repo", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--check", action="store_true", help="只核对、报告，不改动任何文件")
    ap.add_argument("--keep-extra", action="store_true",
                    help="留着本节点用不到的分片（默认删掉：目录里只留 mbf 在这台节点上要的）")
    ap.add_argument("--prune", action="store_true", help=argparse.SUPPRESS)   # 旧参数，现在是默认行为
    ap.add_argument("--clean", action="store_true", help="先清空整个目录")
    ap.add_argument("--no-verify", action="store_true", help="不算 sha256（只比大小）")
    ap.add_argument("--curl", action="store_true", help="用 curl 下载（可续传）")
    ap.add_argument("--no-xet", action="store_true", help="HF 走普通 HTTPS、不走 Xet")
    ap.add_argument("--reserve-gb", type=float, default=2.0)
    a = ap.parse_args(argv)
    if a.no_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    prune = not a.keep_extra
    spec = json.loads(base64.b64decode(a.spec).decode("utf-8"))
    node = spec.get("node") or socket.gethostname()
    d = os.path.abspath(os.path.expanduser(a.dir))
    src = make_source(a)
    print(f"[weights] {node}（{socket.gethostname()}）：{src.label} → {d}；层 {spec.get('layers')}"
          f"{' + embed' if spec.get('embed') else ''}{' + norm/lm_head' if spec.get('head') else ''}"
          f"{'（只核对）' if a.check else ''}", flush=True)

    # 源的 config / index 先拉到临时目录，用来判「本地是不是这个模型」与算要哪些分片
    import tempfile
    ref = tempfile.mkdtemp(prefix="p2p_weights_ref_")
    for f in REQUIRED_SMALL:
        try:
            src.fetch_small(f, ref)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"从 {src.label} 拿不到 {f}：{e}（没有外网？设 HF_ENDPOINT 用镜像，或 --source dir:）")
    index = _load_json(os.path.join(ref, "model.safetensors.index.json"))
    wm = index["weight_map"]
    shards, n_tensors = needed_shards(wm, spec)
    all_shards = sorted(set(wm.values()))
    sizes = src.sizes(shards)
    shas = {} if a.no_verify else src.shas(shards)

    os.makedirs(d, exist_ok=True) if not a.check else None
    wrong = None if a.clean else identity_problem(d, ref)
    verified = _load_json(os.path.join(d, VERIFIED)) or {}

    # ---- 逐分片的状态 ----
    status = {}
    for s in shards:
        p = os.path.join(d, s)
        if a.clean or wrong:
            status[s] = "missing"
        elif not os.path.exists(p):
            status[s] = "missing"
        elif sizes.get(s) is not None and os.path.getsize(p) != sizes[s]:
            status[s] = "bad_size"
        elif shas.get(s) and verified.get(s) == shas[s]:
            status[s] = "ok"
        elif shas.get(s):
            status[s] = "unverified"             # 大小对、还没算过 sha
        else:
            status[s] = "ok"                     # 源不给 sha：只能比大小
    present = set(os.listdir(d)) if os.path.isdir(d) else set()
    extra = sorted(f for f in present if f.endswith(".safetensors") and f not in shards
                   and not (a.clean or wrong))
    stale = sorted(f for f in present if f.endswith((".part", ".incomplete")))
    todo = [s for s in shards if status[s] in ("missing", "bad_size")]
    need = sum(sizes.get(s) or 4_000_000_000 for s in todo)
    have = sum(os.path.getsize(os.path.join(d, s)) for s in shards if status[s] in ("ok", "unverified"))
    free = shutil.disk_usage(d if os.path.isdir(d) else os.path.dirname(d) or "/").free
    report = {"node": node, "host": socket.gethostname(), "dir": d, "source": src.label,
              "layers": spec.get("layers"), "embed": bool(spec.get("embed")), "head": bool(spec.get("head")),
              "n_tensors": n_tensors, "shards_needed": len(shards), "shards_total": len(all_shards),
              "status": status, "extra": extra, "extra_bytes": sum(os.path.getsize(os.path.join(d, f)) for f in extra),
              "stale": stale, "wrong_model": wrong, "need_bytes": need, "have_bytes": have,
              "needed_bytes": sum(sizes.get(s) or 0 for s in shards), "free_bytes": free,
              "prune": prune,
              "ok": not todo and not wrong and all(v == "ok" for v in status.values()) and not (prune and extra)}
    print(f"[weights] 要 {len(shards)}/{len(all_shards)} 个分片（{n_tensors} 个张量，"
          f"{fmt_gb(report['needed_bytes'])}）：已有 {sum(v == 'ok' for v in status.values())}、"
          f"待校验 {sum(v == 'unverified' for v in status.values())}、大小不对 "
          f"{sum(v == 'bad_size' for v in status.values())}、缺 {sum(v == 'missing' for v in status.values())}"
          f"（要下 {fmt_gb(need)}）；用不到的 {len(extra)} 个（{fmt_gb(report['extra_bytes'])}，"
          f"{'留着' if a.keep_extra else 'fetch 时删掉'}）；"
          f"空闲 {fmt_gb(free)}" + (f"\n[weights] 模型不对：{wrong} → 清空重下" if wrong else ""), flush=True)
    if a.check:
        print(f"{REPORT_MARK} {json.dumps(report, ensure_ascii=False)}", flush=True)
        shutil.rmtree(ref, ignore_errors=True)
        return 0

    # ---- 修复 ----
    prog = {"state": "checking", "need": need, "base": 0, "shards_total": len(todo), "shards_done": 0,
            "current": None, "started": time.time(), "updated": time.time()}

    def save_progress(**kw):
        prog.update(kw, updated=time.time())
        tmp = os.path.join(d, PROGRESS + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(prog, f)
        os.replace(tmp, os.path.join(d, PROGRESS))

    try:
        if a.clean:
            freed = clean_dir(d)
            print(f"[weights] 清空 {d}（--clean），释放 {fmt_gb(freed)}", flush=True)
            verified, free = {}, shutil.disk_usage(d).free
        elif wrong:
            freed = clean_managed(d)
            print(f"[weights] 模型不对：删掉 {d} 里的权重分片与模型文件（其余文件不动），释放 {fmt_gb(freed)}",
                  flush=True)
            verified, free = {}, shutil.disk_usage(d).free
        save_progress()
        for f in stale:
            p = os.path.join(d, f)
            if os.path.exists(p):
                os.remove(p)
        for s in [s for s in shards if status[s] == "bad_size"]:
            print(f"[weights] {s} 大小不对，删掉重下", flush=True)
            os.remove(os.path.join(d, s))
        if prune:
            for f in extra:
                os.remove(os.path.join(d, f))
                verified.pop(f, None)
            if extra:
                print(f"[weights] 删掉用不到的 {len(extra)} 个分片，释放 {fmt_gb(report['extra_bytes'])}", flush=True)
            free = shutil.disk_usage(d).free

        def mark_verified(s):
            verified[s] = shas.get(s)
            with open(os.path.join(d, VERIFIED), "w", encoding="utf-8") as f:
                json.dump(verified, f, indent=1)

        # 已有但没校验过的：算一次 sha256，不对就删掉加进待下载
        for s in [s for s in shards if status[s] == "unverified"]:
            save_progress(current=f"{s}（校验 sha256）")
            if sha256_of(os.path.join(d, s)) == shas[s]:
                mark_verified(s)
            else:
                print(f"[weights] {s} sha256 不对，删掉重下", flush=True)
                os.remove(os.path.join(d, s))
                todo.append(s)
                need += sizes.get(s) or 4_000_000_000
        if need + a.reserve_gb * 1e9 > free:
            raise SystemExit(f"磁盘不够：要 {fmt_gb(need)}（另留 {a.reserve_gb:g} GB），空闲 {fmt_gb(free)}")
        # 小文件（config / index / tokenizer …）：只补缺的，已有的不覆盖（本地 index 可能是部分 checkpoint
        # 自己的版本，别的程序在用；模型不对 / --clean 时前面已经删掉，这里会重新拿）
        for f in SMALL_FILES:
            if os.path.exists(os.path.join(d, f)):
                continue
            try:
                src.fetch_small(f, d)
            except Exception as e:  # noqa: BLE001
                if f in REQUIRED_SMALL:
                    raise
                print(f"[weights] 可选文件 {f} 没拿到（{type(e).__name__}）", flush=True)
        save_progress(state="downloading", need=need, base=dir_size(d), shards_total=len(todo))
        t0 = time.time()
        for k, s in enumerate(todo, 1):
            save_progress(current=s)
            t = time.time()
            src.download(s, d, expected_size=sizes.get(s))
            p = os.path.join(d, s)
            sz = os.path.getsize(p)
            if sizes.get(s) is not None and sz != sizes[s]:
                os.remove(p)
                raise SystemExit(f"{s}：下到 {sz} 字节，应为 {sizes[s]}（已删，重跑会重下）")
            if shas.get(s):
                save_progress(current=f"{s}（校验 sha256）")
                if sha256_of(p) != shas[s]:
                    os.remove(p)
                    raise SystemExit(f"{s}：sha256 不对（已删，重跑会重下）")
                mark_verified(s)
            save_progress(shards_done=k)
            print(f"[weights] {k}/{len(todo)} {s} {fmt_gb(sz)}，{time.time() - t:.0f}s"
                  f"（{sz / max(time.time() - t, 1e-3) / 1e6:.0f} MB/s）", flush=True)
    except BaseException as e:
        try:
            save_progress(state="failed", error=f"{type(e).__name__}: {str(e)[:300]}")
        except OSError:
            pass
        raise
    finally:
        shutil.rmtree(ref, ignore_errors=True)
    # index 改成只列本节点留下的分片：vLLM 只加载 index 里出现的分片文件，旧的完整 / 别人裁剪过的
    # index 与目录里实际的分片会对不上（留着用不到的分片时把它们也列进去）
    keep = set(shards) | (set() if prune else
                          {f for f in os.listdir(d) if f.endswith(".safetensors") and f in set(wm.values())})
    sub = {n: sh for n, sh in wm.items() if sh in keep}
    tmp = os.path.join(d, "model.safetensors.index.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"metadata": index.get("metadata", {}), "weight_map": sub}, f, indent=2)
    os.replace(tmp, os.path.join(d, "model.safetensors.index.json"))
    save_progress(state="done", current=None)
    with open(os.path.join(d, MANIFEST), "w", encoding="utf-8") as f:
        json.dump({"source": src.label, "spec": spec, "shards": shards, "pruned": prune,
                   "time": time.strftime("%F %T")}, f, indent=1, ensure_ascii=False)
    print(f"[weights] 完成，用时 {time.time() - t0:.0f}s；{len(shards)} 个分片在 {d}", flush=True)
    print(OK_MARK, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
