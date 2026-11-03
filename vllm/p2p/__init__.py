# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P2P MoE 移植：把 p2p-framework 的「双段放置 + 只装部分专家」搬到 vLLM 上。

计划与验收标准见 p2p-framework 的 `VLLM_MIGRATION.md`。本源码树里的改动：

* `vllm/model_executor/models/qwen3_moe_partial.py` —— `Qwen3MoePartialForCausalLM`：
  每层只为驻留专家分配 / 加载权重，路由等价于 p2p-framework 的 `local_topk`；
  已登记在 `vllm/model_executor/models/registry.py`（测试侧 `tests/models/registry.py`）。
* `vllm/p2p/layout.py` —— p2p 部署清单 → 每条静态通道的切层、逐层专家（`--hf-overrides`）
  与逐节点 `vllm serve` 命令。纯 Python，控制机上不装 vLLM 也能按文件路径直接跑。
* `vllm/p2p/parity.py` —— 单卡对拍（vLLM 原生 / 部分专家模型 / p2p-framework 参考实现）。
* `vllm/p2p/envcheck.py` —— 安装脚本每一步的「就绪了吗」检查。
* `vllm/p2p/launch.py` —— 按 `channels.json` 拉起 / 查看 / 停止通道与识别服务（控制机，纯 Python）。
* `vllm/p2p/gateway.py` —— A3/A4：按 task 分发的 OpenAI 兼容网关（oracle / lr / shadow）。
* `vllm/p2p/identifier.py` —— A4：前 L₀ 层 + LR 的识别服务（复用 p2p-framework 的执行层与 LR）。
* `vllm/p2p/bench.py` —— A3：闭环压测，结果与 p2p-framework 的 results/*.json 同构。
* `vllm/p2p/stats.py` —— A5：worker 内的逐请求路由统计（全量 top-k → miss、LR 特征）；
  模型里的写入在 `qwen3_moe_partial.py`（自定义算子 `vllm.p2p_route_select`），
  逐请求切分在 `vllm/v1/worker/gpu_model_runner.py` 的两处钩子（`p2p_stats`）。
* `vllm/entrypoints/openai/serving_completion.py` —— 网关发来的请求在最后的 usage 块里附
  引擎侧的逐请求时间（`p2p_metrics`：排队 / prefill / decode）。
* `vllm/p2p/stage.py` —— B-1 分段引擎（前段 = 前 L₀ 层、后段 = 其余层，各是一个独立的 vLLM
  引擎）的协议与公共件；`hf_overrides` 里有 `p2p_stage` 才生效。引擎内的改动：
  模型（`qwen3_moe_partial.py`：只建本段的层、前段交出残差流、后段零 embed）、调度器
  （`v1/core/sched/scheduler.py`：后段按「已收到 hidden 的位置」设门、前段随调度结果送输入
  token、`p2p_append` / `p2p_feed`）、EngineCore（`v1/engine/core.py`：两个 utility、
  等输入时不空转）、runner（`v1/worker/gpu_model_runner.py`：前段不采样、后段按位置喂 hidden），
  以及 `EngineCoreOutput` / `ModelRunnerOutput` / `SchedulerOutput` / `Request` 上各一个字段。
* `vllm/p2p/tiered.py` —— B-1 的协调器（按请求绑定后段、两个方向搬 token / hidden）与 B0
  原型（1 前段 + 2 同 task 后段，单卡微型 checkpoint，对拍整模型 + 开销）。
* B1（真模型、跨机）：`vllm/p2p/stages.py`（channels.json → stages.json：每个前段节点一个前段引擎、
  每条通道的后段节点组成一个后段引擎）、`vllm/p2p/stage_agent.py`（每个分段引擎首节点上的代理：
  本机 EngineCore + 给协调器的 ZMQ 入口；其余节点 headless worker）、`vllm/p2p/remote.py`
  （协调器 ↔ 代理的线路，`RemoteClient` 与 `AsyncMPClient` 同接口）、`vllm/p2p/tiered_gateway.py`
  （OpenAI 兼容网关 + 协调器，bench / report / compare 原样可用；`--mode lr` = 前段出口识别：前段在
  prompt 算完时交出前 L₀ 层的路由统计（`stats.RouteStats.take_lr` → `EngineCoreOutput.p2p_feat`），
  协调器跑 LR 定 task、再绑该 task 最闲的后段，路径此后不变）。
* 多级（中段，`role = mid`）：输入像后段、输出像前段、不采样（调度器在 p2p_feed 里补占位 token
  让序列变长）。协调器绑定的是一条路线（`tiered.Route`：0 个或多个中段 + 1 个后段），逐段 prefill，
  之后 hidden 一段段往下传，同 task 的多条路线可以分叉 / 汇合、按规划的 weight 分请求；
  `kv_guard` 按 KV 容量预留，避免各段互相等死。清单由 `colorplan.py` 从
  `moe_color_planner.py --deploy --interleave` 的结果生成（一张卡可以有几个引擎）。
* `vllm/p2p/layer_bench.py` —— 各型 GPU 上「一层」的实测耗时（同一套 vLLM kernel、随机权重、
  两种层数相减），写进 p2p-framework 规划器的拓扑文件（节点的 `ms_per_layer`）。
* `tests/p2p/` —— 以上各件的单测与端到端测试（假的 vLLM 通道，不要 GPU）。

在集群上安装：先在一台节点上（本目录在 /home/ubuntu/vllm-p2p）

    bash /home/ubuntu/vllm-p2p/p2p_install.sh       # 逐步检查，缺哪步补哪步；可重复运行

再在这台上把其余节点一起装好（建本地 wheel 仓库 → 推代码 + 仓库 → 各节点离线补装）::

    bash /home/ubuntu/vllm-p2p/p2p_distribute.sh

它用 moe 环境的 Python 解释器建一个**独立** venv（/home/ubuntu/venvs/vllm-p2p，不复用 moe 的包，
moe 本身不改；装完约 10 GiB），装 torch 2.9.1（含 CUDA 12 库）与构建工具，再以可编辑方式安装
本目录，最后验证。不用 --system-site-packages 复用 moe 的 CUDA 库：torch 按 .so 里的相对路径
（$ORIGIN/../../nvidia/*/lib）找库，库必须和 torch 在同一个 site-packages。核心一步等价于::

    VLLM_VERSION_OVERRIDE=0.14.0 VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_LOCATION=<官方 vllm-0.14.0 wheel> \
        pip install --no-cache-dir --no-build-isolation -e .

三个参数各有原因：

* `VLLM_VERSION_OVERRIDE` —— 本目录不是 git 仓库，setuptools-scm 取不到版本号会直接报错；
* `VLLM_USE_PRECOMPILED` + `VLLM_PRECOMPILED_WHEEL_LOCATION` —— 复用官方 0.14.0 wheel 里编好的
  CUDA 扩展（只改了 Python 文件，扩展与源码匹配），不在节点上编译；
* `--no-build-isolation` —— 构建依赖里有 `torch==2.9.1`，隔离构建会在临时目录里再下一整套
  torch + CUDA 库（约 4 GiB）。先装好 torch 再关掉隔离就不会。

venv 只借用 moe 的 Python 解释器（不读它的包），所以不要删 moe 环境。ssh 非交互 shell 不会 activate venv，
所以 `vllm/p2p/layout.py` 生成命令时用 `--vllm-bin /home/ubuntu/venvs/vllm-p2p/bin/vllm`。

本包刻意不在 `__init__` 里 import 任何东西 —— `layout.py` / `launch.py` 要能在没有 torch 的
控制机上跑；`gateway.py` / `bench.py` / `identifier.py` 也按文件路径运行（同目录互相 import）。
"""
