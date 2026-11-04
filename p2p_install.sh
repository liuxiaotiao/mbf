#!/usr/bin/env bash
# P2P 移植：在一台节点上安装本目录（修改后的 vLLM 0.14.0）。**逐步检查，缺哪一步补哪一步**，
# 重复运行是安全的 —— 已经就绪的步骤直接跳过。
#
#     bash /home/ubuntu/vllm-p2p/p2p_install.sh            # 检查 + 补装
#     bash /home/ubuntu/vllm-p2p/p2p_install.sh --verify   # 只检查，不装（不就绪则退出码 1）
#
# 步骤（每步先检查，就绪就跳过）：
#   1 venv        用 moe 的 Python 解释器建一个独立 venv（不复用 moe 的任何包，moe 本身不改）
#   2 torch       torch 2.9.1 + CUDA 12 库能 import（vLLM 0.14 钉死这个版本）
#   3 构建工具    requirements/build.txt 都满足（关了构建隔离，所以要自己装）
#   4 vLLM        以可编辑方式装的是本目录、编译好的 .so 在、依赖齐全 ——
#                 只从官方 0.14.0 wheel 里取编译好的 CUDA 扩展，Python 代码全用本目录的
#   5 验证        torch/vLLM 来源正确、部分专家模型已注册、扩展能加载
#
# 包从哪来：有本地 wheel 仓库（控制节点用 p2p_distribute.sh 推过来的，含 constraints.txt）
# 就**离线**从它装，版本与控制节点逐个一致；没有才联网从 PyPI 装。
#
# 为什么不复用 moe 里已有的 CUDA 库（--system-site-packages）：torch 找 libcublas / libcusparseLt 等
# 靠的是写在 .so 里的相对路径 $ORIGIN/../../nvidia/*/lib —— 必须和 torch 装在同一个 site-packages。
#
# 可覆盖的变量（一般不用改）：
#   SRC        本目录（修改后的 vLLM）    默认 /home/ubuntu/vllm-p2p
#   MOE_PY     moe 环境的 python          默认 /home/ubuntu/miniforge3/envs/moe/bin/python
#   VENV       venv                       默认 /home/ubuntu/venvs/vllm-p2p
#   WHEELHOUSE 本地 wheel 仓库            默认 /home/ubuntu/vllm-p2p-wheels（不存在就联网）
#   WHEEL      官方 vllm-0.14.0 wheel     默认在 WHEELHOUSE 里找，找不到就下载
set -euo pipefail

SRC=${SRC:-/home/ubuntu/vllm-p2p}
MOE_PY=${MOE_PY:-/home/ubuntu/miniforge3/envs/moe/bin/python}
VENV=${VENV:-/home/ubuntu/venvs/vllm-p2p}
WHEELHOUSE=${WHEELHOUSE:-/home/ubuntu/vllm-p2p-wheels}
WHEEL=${WHEEL:-}
VERIFY_ONLY=0
[ "${1:-}" = "--verify" ] && VERIFY_ONLY=1
TMPW=$(mktemp -d /tmp/vllm-wheel.XXXXXX)
trap 'rm -rf "$TMPW"' EXIT

say()  { printf '\n== %s\n' "$*"; }
ok()   { printf '  ✓ %s\n' "$*"; }
todo() { printf '  → %s\n' "$*"; }
die()  { printf '✗ %s\n' "$*" >&2; exit 1; }
check() { python -I "$SRC/vllm/p2p/envcheck.py" "$@"; }   # 0 = 就绪

# ---- pip 的来源：本地 wheel 仓库（离线、版本钉死）或 PyPI ------------------ #
PIP_SRC=()
if [ -f "$WHEELHOUSE/constraints.txt" ]; then
  PIP_SRC=(--no-index --find-links "$WHEELHOUSE" -c "$WHEELHOUSE/constraints.txt")
  SRC_DESC="本地 wheel 仓库 $WHEELHOUSE（离线，版本与控制节点一致）"
else
  SRC_DESC="PyPI（联网）"
fi
export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
pipi() { pip install ${PIP_SRC[@]+"${PIP_SRC[@]}"} "$@"; }   # 空数组在 set -u 下也安全

verify() {
  # venv 已激活的前提下：torch 2.9.1 来自 venv、vllm 来自本目录、模型已注册、扩展能加载
  ( cd / && python - <<'PY'
import sys
import torch, vllm
from vllm import ModelRegistry
print(f"  torch {torch.__version__}  ←  {torch.__file__}")
print(f"  vllm  {vllm.__version__}  ←  {vllm.__file__}")
assert torch.__version__.startswith("2.9.1"), "用到的不是 venv 里的 torch 2.9.1"
assert "/site-packages/" not in vllm.__file__, "import 到的不是本目录的 vllm"
assert torch.__file__.startswith(sys.prefix), "torch 不是 venv 里的那份"
assert "Qwen3MoePartialForCausalLM" in ModelRegistry.get_supported_archs(), "部分专家模型没注册上"
import vllm._C  # noqa: F401  —— 编译好的扩展能加载
import vllm._moe_C  # noqa: F401
print("  Qwen3MoePartialForCausalLM 已注册，vllm._C 可加载 ✓")
print(f"  CUDA 可用: {torch.cuda.is_available()}" + (f"，{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
PY
  )
}

# ========================================================================= #
say "0/5 预检（$(hostname)）  包来源：$SRC_DESC"
[ -f "$SRC/vllm/model_executor/models/qwen3_moe_partial.py" ] \
  || die "$SRC 里没有 qwen3_moe_partial.py —— 不是修改后的 vLLM，或者还没同步过来"
[ -f "$SRC/vllm/p2p/envcheck.py" ] || die "$SRC 里没有 vllm/p2p/envcheck.py —— 代码目录不是最新的"
[ -x "$MOE_PY" ] || die "找不到 moe 的 python：$MOE_PY（用 MOE_PY=… 指定）"
pyv=$("$MOE_PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
case "$pyv" in 3.10|3.11|3.12|3.13) ok "moe Python $pyv" ;;
  *) die "moe 是 Python $pyv，vLLM 0.14 要 3.10–3.13" ;; esac
glibc=$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo 0)
awk -v v="$glibc" 'BEGIN{split(v,a,"."); exit !(a[1]>2 || (a[1]==2 && a[2]>=31))}' \
  || die "glibc $glibc < 2.31，官方 0.14.0 wheel（manylinux_2_31）装不上"
ok "glibc $glibc"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/    GPU /'
NEED=0     # 有没有任何一步要装 —— 只检查模式下用它决定退出码

# ---- 1 venv --------------------------------------------------------------- #
say "1/5 venv  $VENV"
venv_ok=1
[ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c 'import sys' 2>/dev/null || venv_ok=0
if [ "$venv_ok" = 1 ] && grep -qiE '^include-system-site-packages *= *true' "$VENV/pyvenv.cfg" 2>/dev/null; then
  todo "是旧的 --system-site-packages 版本（会找不到 CUDA 库）"; venv_ok=0
fi
if [ "$venv_ok" = 1 ] && [ "$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" != "$pyv" ]; then
  todo "venv 的 Python 版本与 moe 不一致"; venv_ok=0
fi
if [ "$venv_ok" = 1 ]; then
  ok "已就绪"
else
  NEED=1
  [ "$VERIFY_ONLY" = 1 ] && { echo "✗ venv 不可用（--verify 模式，不修）"; exit 1; }
  d=$(dirname "$VENV"); while [ ! -d "$d" ]; do d=$(dirname "$d"); done
  free_gb=$(df -Pk "$d" | awk 'NR==2{printf "%d", $4/1048576}')
  [ "${free_gb:-0}" -ge 12 ] || die "$d 只剩 ${free_gb}GB，从头装至少要 12GB（装完约 10GB）"
  todo "（重）建 venv（磁盘剩 ${free_gb}GB）"
  rm -rf "$VENV"; mkdir -p "$(dirname "$VENV")"
  "$MOE_PY" -m venv "$VENV"
  ok "已建好"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
[ "$(python -c 'import sys; print(sys.prefix)')" = "$VENV" ] || die "venv 没激活成功"

# ---- 2 torch -------------------------------------------------------------- #
say "2/5 torch 2.9.1 + CUDA 12 库"
if check torch; then
  ok "已就绪"
else
  NEED=1
  [ "$VERIFY_ONLY" = 1 ] && { echo "✗ torch 不就绪（--verify 模式，不修）"; exit 1; }
  todo "安装"
  pipi torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
  check torch || die "装完 torch 还是 import 不了 —— 看上面的报错"
  ok "已装好"
fi

# ---- 3 构建工具 ------------------------------------------------------------ #
say "3/5 构建工具（requirements/build.txt）"
if check reqs "$SRC/requirements/build.txt"; then
  ok "已就绪"
else
  NEED=1
  [ "$VERIFY_ONLY" = 1 ] && { echo "✗ 构建工具不齐（--verify 模式，不修）"; exit 1; }
  todo "安装"
  pipi -r "$SRC/requirements/build.txt"
  check reqs "$SRC/requirements/build.txt" || die "装完构建工具还是不齐"
  ok "已装好"
fi

# ---- 4 vLLM（可编辑安装本目录） ------------------------------------------- #
say "4/5 vLLM：可编辑安装 $SRC"
if check vllm "$SRC"; then
  ok "已就绪"
else
  NEED=1
  [ "$VERIFY_ONLY" = 1 ] && { echo "✗ vLLM 不就绪（--verify 模式，不修）"; exit 1; }
  if [ -z "$WHEEL" ]; then
    WHEEL=$(ls "$WHEELHOUSE"/vllm-0.14.0-*x86_64.whl 2>/dev/null | head -1 || true)
  fi
  if [ -z "$WHEEL" ]; then
    [ ${#PIP_SRC[@]} -gt 0 ] && die "本地 wheel 仓库里没有官方 vllm-0.14.0 wheel —— 在控制节点重跑 p2p_distribute.sh"
    todo "下载官方 0.14.0 wheel（只取编译好的 .so）"
    pip download vllm==0.14.0 --no-deps -d "$TMPW"
    WHEEL=$(ls "$TMPW"/vllm-0.14.0-*x86_64.whl)
  fi
  [ -f "$WHEEL" ] || die "wheel 不存在：$WHEEL"
  todo "安装（预编译扩展取自 $(basename "$WHEEL")）"
  ( cd "$SRC" && VLLM_VERSION_OVERRIDE=0.14.0 VLLM_USE_PRECOMPILED=1 \
      VLLM_PRECOMPILED_WHEEL_LOCATION="$WHEEL" \
      pipi --no-build-isolation -e . )
  check vllm "$SRC" || die "装完 vLLM 检查还是不过 —— 看上面缺了什么"
  ok "已装好"
fi

# ---- 5 验证 ---------------------------------------------------------------- #
say "5/5 验证"
verify
echo "  moe 的 torch 仍是 $("$MOE_PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '?')（应为 2.9.0）"
echo
if [ "$VERIFY_ONLY" = 1 ]; then
  echo "自检通过（$(hostname)）"
elif [ "$NEED" = 0 ]; then
  echo "本来就装好了，什么都没动（$(hostname)）"
else
  echo "装好了（$(hostname)）。以后用：source $VENV/bin/activate"
  echo "远程起服务时用绝对路径：$VENV/bin/vllm（ssh 非交互 shell 不会激活 venv）"
fi
