#!/usr/bin/env bash
# 在**已经装好的那台**（控制节点）上运行，三步：
#   1. 建本地 wheel 仓库：按本机 venv 里**实际装着的版本**逐个下载 wheel（只在第一次或版本变了时做）
#   2. 把代码目录 + wheel 仓库推给各节点
#   3. 在各节点上跑 p2p_install.sh —— 它逐步检查，缺哪一步补哪一步，从本地 wheel 仓库**离线**装
#      （已经装好的节点什么都不动；装了一半的节点只补缺的那几步）
#
#     bash /home/ubuntu/vllm-p2p/p2p_distribute.sh                          # hosts.txt 里除本机外的全部
#     bash /home/ubuntu/vllm-p2p/p2p_distribute.sh 192.168.1.6 192.168.1.4  # 只处理这几台
#     REBUILD=1 bash /home/ubuntu/vllm-p2p/p2p_distribute.sh                # 强制重建 wheel 仓库
#
# 各节点最终用的包版本与本机逐个一致（wheel 仓库带 constraints.txt，安装时钉死）。
#
# 可覆盖的变量：
#   SRC        代码目录（修改后的 vLLM）  默认 /home/ubuntu/vllm-p2p
#   VENV       本机装好的 venv            默认 /home/ubuntu/venvs/vllm-p2p
#   WHEELHOUSE wheel 仓库（本机与各节点同一路径）默认 /home/ubuntu/vllm-p2p-wheels
#   MOE_PY     moe 的 python              默认 /home/ubuntu/miniforge3/envs/moe/bin/python
#   HOSTS      节点表                     默认 /home/ubuntu/p2p-framework/task/hosts.txt
#   SSH_USER                              默认 ubuntu
#   JOBS       同时处理几台               默认 4（推 wheel 仓库时本机网卡是瓶颈）
set -euo pipefail

SRC=${SRC:-/home/ubuntu/vllm-p2p}
VENV=${VENV:-/home/ubuntu/venvs/vllm-p2p}
WHEELHOUSE=${WHEELHOUSE:-/home/ubuntu/vllm-p2p-wheels}
MOE_PY=${MOE_PY:-/home/ubuntu/miniforge3/envs/moe/bin/python}
HOSTS=${HOSTS:-/home/ubuntu/p2p-framework/task/hosts.txt}
SSH_USER=${SSH_USER:-ubuntu}
JOBS=${JOBS:-4}
REBUILD=${REBUILD:-0}
LOGDIR=${LOGDIR:-$HOME/p2p_distribute_logs}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"

die() { printf '✗ %s\n' "$*" >&2; exit 1; }

# ---- 0. 本机必须是装好的 -------------------------------------------------- #
echo "== 0. 本机自检（$(hostname)）"
bash "$SRC/p2p_install.sh" --verify >/dev/null 2>&1 \
  || die "本机的安装没通过自检 —— 先在本机跑 bash $SRC/p2p_install.sh 装好"
PYV=$("$MOE_PY" -c 'import sys; print(sys.version.split()[0])')
echo "  ✓ 本机已装好（moe Python $PYV）"

# ---- 1. wheel 仓库 --------------------------------------------------------- #
echo "== 1. 本地 wheel 仓库 $WHEELHOUSE"
TMPC=$(mktemp)
trap 'rm -f "$TMPC"' EXIT
# 本机 venv 里实际装着的版本（不含可编辑安装的 vllm 本身、不含 pip）
"$VENV/bin/pip" freeze --all --exclude-editable 2>/dev/null \
  | grep -v -E '^pip==' | grep -v ' @ ' | sort > "$TMPC"
n_pkg=$(wc -l < "$TMPC")
skipped=$("$VENV/bin/pip" freeze --all --exclude-editable 2>/dev/null | grep -c ' @ ' || true)
[ "$skipped" = 0 ] || echo "  ⚠ 有 $skipped 个包是从本地路径/URL 装的，不进 wheel 仓库（节点上会缺）"
if [ "$REBUILD" = 0 ] && [ -f "$WHEELHOUSE/constraints.txt" ] \
   && cmp -s "$TMPC" "$WHEELHOUSE/constraints.txt" \
   && ls "$WHEELHOUSE"/vllm-0.14.0-*x86_64.whl >/dev/null 2>&1; then
  echo "  ✓ 已是最新（$n_pkg 个包，$(du -sh "$WHEELHOUSE" | cut -f1)），跳过下载"
else
  echo "  → 按本机版本下载 $n_pkg 个包（第一次约 5GB，之后版本不变就不再下载）"
  mkdir -p "$WHEELHOUSE"
  "$VENV/bin/pip" download --no-deps -d "$WHEELHOUSE" -r "$TMPC"
  "$VENV/bin/pip" download --no-deps -d "$WHEELHOUSE" vllm==0.14.0   # 只取里面编译好的 .so
  cp "$TMPC" "$WHEELHOUSE/constraints.txt"
  n_sdist=$(ls "$WHEELHOUSE"/*.tar.gz "$WHEELHOUSE"/*.zip 2>/dev/null | wc -l || true)
  [ "$n_sdist" = 0 ] || echo "  ⚠ 有 $n_sdist 个包只有源码包（.tar.gz），节点上离线装时要现场编译"
  echo "  ✓ 建好（$(du -sh "$WHEELHOUSE" | cut -f1)）"
fi

# ---- 2. 目标列表 ----------------------------------------------------------- #
if [ $# -gt 0 ]; then
  TARGETS="$*"
else
  [ -f "$HOSTS" ] || die "找不到 $HOSTS（用 HOSTS=… 指定，或在命令行直接给 IP）"
  MINE=" $(hostname -I 2>/dev/null) "
  TARGETS=$(awk '{sub(/#.*/,"")} NF>=2 {split($2,a,":"); print a[1]}' "$HOSTS" \
            | while read -r ip; do case "$MINE" in *" $ip "*) ;; *) echo "$ip" ;; esac; done)
fi
[ -n "$TARGETS" ] || die "没有要处理的节点"
echo "== 2. 目标 $(echo $TARGETS | wc -w) 台：$(echo $TARGETS)"
mkdir -p "$LOGDIR"

one() {   # 单台：预检 → 推代码 → 推 wheel 仓库 → 远端检查 + 补装。输出写进 $LOGDIR/<ip>.log
  # 每步都显式 `|| return 1`：本函数在 `one … && …` 里被调用，那种上下文里 bash 会忽略 set -e
  local ip=$1 t=$SSH_USER@$1
  {
    echo "== $ip  $(date '+%F %T')"
    rpy=$($SSH "$t" "$MOE_PY -c 'import sys; print(sys.version.split()[0])'" 2>&1) \
      || { echo "✗ 连不上或没有 $MOE_PY：$rpy"; return 1; }
    [ "$rpy" = "$PYV" ] || { echo "✗ 那边 moe 是 Python $rpy，本机是 $PYV —— wheel 是按本机版本下的，不通用"; return 1; }
    free=$($SSH "$t" "d=$(dirname "$WHEELHOUSE"); while [ ! -d \$d ]; do d=\$(dirname \$d); done; df -Pk \$d | awk 'NR==2{printf \"%d\", \$4/1048576}'")
    echo "  磁盘剩 ${free}GB"
    $SSH "$t" "mkdir -p $SRC $WHEELHOUSE" || { echo "✗ 建目录失败"; return 1; }
    # 代码目录**不加 --delete**：那边已有的 .so / _version.py 不能被删
    rsync -a -e "$SSH" "$SRC/" "$t:$SRC/" || { echo "✗ 推代码目录失败"; return 1; }
    # wheel 仓库是我们自己的，镜像过去（--delete 清掉旧版本的 wheel）
    rsync -a --delete -e "$SSH" "$WHEELHOUSE/" "$t:$WHEELHOUSE/" || { echo "✗ 推 wheel 仓库失败"; return 1; }
    # 远端：逐步检查，缺哪步补哪步（离线，因为 wheel 仓库在默认路径）
    $SSH "$t" "SRC=$SRC VENV=$VENV MOE_PY=$MOE_PY WHEELHOUSE=$WHEELHOUSE bash $SRC/p2p_install.sh" \
      || { echo "✗ 远端安装 / 检查没通过"; return 1; }
    echo "✓ 完成"
  } > "$LOGDIR/$ip.log" 2>&1
}
export -f one
export SSH SSH_USER MOE_PY VENV SRC WHEELHOUSE PYV LOGDIR

echo "== 3. 推送 + 各节点检查/补装（同时 $JOBS 台，日志在 $LOGDIR/）"
for ip in $TARGETS; do echo "$ip"; done \
  | xargs -P "$JOBS" -I{} bash -c 'one {} && echo "  ✓ {}  $(grep -m1 -E "本来就装好了|装好了" $LOGDIR/{}.log | sed "s/（.*//")" || echo "  ✗ {}（见 $LOGDIR/{}.log）"' \
  | tee "$LOGDIR/summary.txt"
n_bad=$(grep -c '✗' "$LOGDIR/summary.txt" || true)
echo
if [ "$n_bad" = 0 ]; then
  echo "全部就绪。各节点用：source $VENV/bin/activate"
  echo "（各节点上的 $WHEELHOUSE 约 5GB；不再需要自动修复的话可以删掉）"
else
  echo "$n_bad 台没成功 —— 看对应日志；修好后只重跑那几台：bash $0 <ip> …"
  exit 1
fi
