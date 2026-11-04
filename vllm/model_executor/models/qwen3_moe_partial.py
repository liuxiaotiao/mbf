# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-MoE 的 vLLM 实现，**每层只装指定的那些专家** —— p2p-framework `PartialExpertMoEBlock`
在 vLLM 0.14 上的对应物。

    vllm serve <ckpt> --hf-overrides '{"architectures": ["Qwen3MoePartialForCausalLM"],
                                       "p2p_layer_experts": {"8": [2, 3, 5, …], …}}'

配置（经 `--hf-overrides` 写进 HF config，所有 PP 级都读得到）：

* `p2p_layer_experts`：{0-based 层号字符串: 驻留专家 id 列表}。**没列出的层 = 全装**，
  走 vLLM 原生的 `Qwen3MoeSparseMoeBlock`，与官方实现逐位相同。
* `p2p_miss_policy`：目前只支持 `local_topk`（p2p-framework 的实验一直用它）。

为什么不用 EP 的 expert_map
---------------------------
`FusedMoE.__init__` 只在开启专家并行时才让 `local_num_experts` 小于全局数，权重又是
**在构造时**按 `local_num_experts` 分配的。单卡上 expert_map 为 None、权重按 128 个
分配 —— 事后再改 map 来不及，40 层全专家 24GB 装不下。所以这里每层的 `FusedMoE`
**直接以驻留专家数构造**：对它来说，这一层就是一个只有 n 个专家的 MoE。

路由：local_topk 的等价性
-------------------------
路由器 `mlp.gate` 仍是全量 E 维（统计要看全量路由，见 A5）。前向时把它的输出按驻留集
**取列**，再交给 `FusedMoE`（softmax → top-k → 重归一，`renormalize=True`）。
p2p-framework 的 local_topk 是：全量 softmax → 非驻留置 −∞ → top-k → 重归一。两者相同：

* 子集上的 softmax = 全量 softmax 限制到子集再除以子集总和 —— 单调、保比例；
* 所以 top-k 选出的是同一批专家，重归一后的权重也逐个相等。

于是「驻留集覆盖实际路由时与全装逐位一致」这条性质在这里同样成立（不同 kernel 的
累加顺序除外）。驻留专家少于 top_k 时取 min(top_k, n)，与 p2p-framework 相同。

PP 各级只建自己用得到的东西
---------------------------
原生 Qwen3-MoE 在**每一级**都建 embed_tokens、norm、lm_head。而 `p2pmoe.deploy.fetch`
拉下来的部分 checkpoint 里，前段节点没有 lm_head、后段节点没有 embed_tokens ——
vLLM 的加载器是严格的（`default_loader.py`：每个参数都要有权重），会直接报错。
这里按 Llama 的做法把用不到的换成 `PPMissingLayer`（加载器遇到它就跳过）。

A5 统计钩子（hf_overrides 里有 `p2p_stats` 才开）
----------------------------------------------
每个 MoE 层（装全的层也算）在路由之后多走一个自定义算子 `vllm.p2p_route_select`：
它返回交给 `FusedMoE` 的那份路由打分（与不开钩子时同一个张量运算，数值不变），顺带把
**全量** softmax 的 top-k 写进 `vllm/p2p/stats.py` 的固定显存缓冲区。算子对 torch.compile
是黑盒、在 CUDA Graph 里随图重放，所以开着钩子也能用 CUDA Graph。逐请求的切分与落盘在
model runner 里做（见 `stats.py`）。

B-1 分段引擎（hf_overrides 里有 `p2p_stage` 才开）
----------------------------------------------
`{"role": "front"|"mid"|"back", "layers": [lo, hi]}`：本引擎只建 [lo, hi) 这些层（再按 PP 切给各级，
层号仍是全局的，权重按原名加载）。前段：final norm 换成 `AddResidual`（交出残差流）、没有
lm_head；后段：embed_tokens 换成 `ZeroEmbed`（每个位置都由前段的 hidden 喂入）；中段两样都换
（输入是上一段的 hidden、输出交给下一段）。
协议与调度见 `vllm/p2p/stage.py`。

Turing 上的 MoE（`p2p_moe_impl`）
------------------------------
vLLM 0.14 的 Triton `fused_experts` 在 sm75（RTX 6000 / T4）上极慢：单卡实测 E=64、T=320 时
RTX 6000 27 ms、T4 46 ms（折合 0.5–0.9 TFLOPS，不到卡的 2%），而 A30 1.2 ms
（`vllm/p2p/moe_kernel_bench.py`）。所以 sm < 8.0 时改走自定义算子 `vllm.p2p_dense_moe`：

* 一步 ≤ `VLLM_P2P_DENSE_MOE_MAX_TOKENS`（默认 512，覆盖 CUDA Graph 抓的 decode 形状）：
  每个 token 过本层**全部**驻留专家（cuBLAS 批量矩阵乘，走 tensor core），再按路由权重
  （非 top-k 为 0）加权求和 —— 多算 E/top-k 倍，但形状固定、能进 CUDA Graph；
  同样形状 RTX 6000 3.7 ms、T4 12.6 ms；
* 更大的一步（prefill / 混合批，不在 CUDA Graph 里）：按专家分组、每个专家一次 cuBLAS 矩阵乘。

路由与 FusedMoE 相同：softmax → top-k → 重归一（`renormalize`）。`p2p_moe_impl`（hf_overrides）
或环境变量 `VLLM_P2P_MOE_IMPL` 取 `auto`（默认：sm < 8.0 用 dense）/ `dense` / `triton`。
只支持 TP = 1、非 sequence-parallel（否则退回 FusedMoE）。

逐专家核对
----------
严格检查是**按参数**的：`experts.w13_weight` 是 n 个专家堆在一起的一个参数，只要其中
一个专家加载过，它就算「已加载」。某台节点的部分 checkpoint 若缺了某个驻留专家，那一行
会是未初始化的显存 —— 不报错，只是输出悄悄变差。所以 `load_weights` 自己逐
(层, 专家, gate/up/down) 核对，缺一个就报。
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models import qwen3_moe as _q
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    maybe_offload_to_cpu,
)
from vllm.p2p import stage as _p2p_stage
from vllm.p2p import stats as _p2p_stats
from vllm.p2p.layout import (
    LayoutError,
    expert_key,
    localize_expert_name,
    parse_layer_experts,
)
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

__all__ = ["Qwen3MoePartialForCausalLM", "PartialQwen3MoeSparseMoeBlock",
           "SUPPORTED_MISS_POLICIES"]

SUPPORTED_MISS_POLICIES = ("local_topk",)
# TODO(drop / drop_noscale)：全量 softmax 取 top-k、缺失的权重置 0（drop_noscale）或
# 再重归一（drop）。走 FusedMoE 的 custom_routing_function，把全局专家号映射成本地号、
# 缺失的映到任一本地专家且权重为 0。实验没用到，先不做。


# --------------------------------------------------------------------------- #
# A5：路由打分 → FusedMoE 的那一步，顺带记全量 top-k
# --------------------------------------------------------------------------- #
def p2p_route_select(router_logits: torch.Tensor, resident_idx: torch.Tensor,
                     slot: int) -> torch.Tensor:
    """返回 `router_logits[:, resident_idx]`（交给 FusedMoE），并把全量 top-k 记进统计缓冲区。

    做成自定义算子是为了让「写缓冲区」这个副作用对 torch.compile 不可见、又不会被当成死代码
    删掉（输出被 FusedMoE 用着）。装全的层传 arange(E)，返回的就是同一份打分。"""
    st = _p2p_stats.get_collector()
    if st is not None and slot >= 0:
        st.capture(slot, router_logits)
    return router_logits.index_select(-1, resident_idx)


def _p2p_route_select_fake(router_logits: torch.Tensor, resident_idx: torch.Tensor,
                           slot: int) -> torch.Tensor:
    return router_logits.new_empty((router_logits.shape[0], resident_idx.shape[0]))


direct_register_custom_op(
    op_name="p2p_route_select",
    op_func=p2p_route_select,
    mutates_args=[],
    fake_impl=_p2p_route_select_fake,
)


# --------------------------------------------------------------------------- #
# Turing（sm < 8.0）：不走 Triton fused_experts 的 MoE（见模块文档）
# --------------------------------------------------------------------------- #
MOE_IMPLS = ("auto", "dense", "triton")


def dense_max_tokens() -> int:
    return int(os.environ.get("VLLM_P2P_DENSE_MOE_MAX_TOKENS", "512"))


def route_topk(router_logits: torch.Tensor, top_k: int,
               renormalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """与 FusedMoE 的 Renormalize 路由相同：softmax → top-k →（重归一）。→ (权重 fp32, 本地专家号)。"""
    w, i = router_logits.float().softmax(-1).topk(top_k, dim=-1)
    if renormalize:
        w = w / w.sum(-1, keepdim=True)
    return w, i


def moe_dense(x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor,
              tw: torch.Tensor, ti: torch.Tensor) -> torch.Tensor:
    """每个 token 过全部 E 个专家（批量矩阵乘），按路由权重加权求和。形状只依赖 T，能进 CUDA Graph。
    w13 [E, 2I, H]（前一半 gate、后一半 up），w2 [E, H, I]（FusedMoE 的布局）。"""
    E, inter = w13.shape[0], w2.shape[2]
    h = torch.bmm(x.unsqueeze(0).expand(E, -1, -1), w13.transpose(1, 2))      # [E, T, 2I]
    a = torch.nn.functional.silu(h[..., :inter]) * h[..., inter:]              # [E, T, I]
    y = torch.bmm(a, w2.transpose(1, 2))                                       # [E, T, H]
    wd = torch.zeros(x.shape[0], E, device=x.device, dtype=torch.float32).scatter_(1, ti.long(), tw)
    return torch.einsum("te,eth->th", wd.to(x.dtype), y)


def moe_grouped(x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor,
                tw: torch.Tensor, ti: torch.Tensor) -> torch.Tensor:
    """按专家分组、每个专家一次矩阵乘。有主机同步（每个专家多少 token），只用在不进 CUDA Graph 的大步。"""
    T, H = x.shape
    inter = w2.shape[2]
    flat = ti.reshape(-1).long()
    tok = torch.arange(T, device=x.device).repeat_interleave(ti.shape[1])
    wt = tw.reshape(-1)
    order = flat.argsort()
    flat, tok, wt = flat[order], tok[order], wt[order]
    counts = torch.bincount(flat, minlength=w13.shape[0]).tolist()
    out = torch.zeros(T, H, device=x.device, dtype=torch.float32)
    s = 0
    for e, c in enumerate(counts):
        if c:
            idx = tok[s:s + c]
            h = x.index_select(0, idx) @ w13[e].t()
            a = torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]
            out.index_add_(0, idx, (a @ w2[e].t()).float() * wt[s:s + c, None])
            s += c
    return out.to(x.dtype)


def p2p_dense_moe(hidden_states: torch.Tensor, router_logits: torch.Tensor, w13: torch.Tensor,
                  w2: torch.Tensor, top_k: int, renormalize: bool) -> torch.Tensor:
    if hidden_states.shape[0] == 0:
        return torch.empty_like(hidden_states)
    tw, ti = route_topk(router_logits, top_k, renormalize)
    if hidden_states.shape[0] <= dense_max_tokens():
        return moe_dense(hidden_states, w13, w2, tw, ti)
    return moe_grouped(hidden_states, w13, w2, tw, ti)


def _p2p_dense_moe_fake(hidden_states: torch.Tensor, router_logits: torch.Tensor,
                        w13: torch.Tensor, w2: torch.Tensor, top_k: int,
                        renormalize: bool) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="p2p_dense_moe",
    op_func=p2p_dense_moe,
    mutates_args=[],
    fake_impl=_p2p_dense_moe_fake,
)


def resolve_moe_impl(config, capability: tuple[int, int] | None) -> str:
    """→ "dense" 或 "triton"。环境变量优先于 hf_overrides 的 `p2p_moe_impl`。"""
    want = (os.environ.get("VLLM_P2P_MOE_IMPL") or getattr(config, "p2p_moe_impl", None)
            or "auto").lower()
    if want not in MOE_IMPLS:
        raise ValueError(f"p2p_moe_impl={want!r}；可选 {MOE_IMPLS}")
    if want == "auto":
        return "dense" if capability is not None and tuple(capability) < (8, 0) else "triton"
    return want


def _device_capability() -> tuple[int, int] | None:
    try:
        if torch.cuda.is_available():
            return tuple(torch.cuda.get_device_capability())
    except Exception:          # noqa: BLE001
        pass
    return None


_LOGGED_IMPL: set[str] = set()


def _pick_dense(block, config) -> bool:
    impl = resolve_moe_impl(config, _device_capability())
    if impl == "dense" and (get_tensor_model_parallel_world_size() != 1
                            or getattr(block, "is_sequence_parallel", False)):
        impl = "triton"            # dense 路径不做 TP 规约 / sequence-parallel 切分
    if impl not in _LOGGED_IMPL:
        _LOGGED_IMPL.add(impl)
        logger.info("p2p MoE：%s（sm=%s；dense = cuBLAS 批量矩阵乘，≤ %d token 一步全专家，"
                    "更大按专家分组）", impl, _device_capability(), dense_max_tokens())
    return impl == "dense"


_PARSED: dict[int, dict[int, tuple[int, ...]]] = {}


def _layer_experts_of(config) -> dict[int, tuple[int, ...]]:
    """从 HF config 读逐层驻留集（0-based）。每个 config 对象只解析一次。

    缓存放在模块里而不是挂到 config 上：vLLM 会序列化 / 哈希 HF config（编译缓存的键），
    往上挂私有字段可能改变哈希或序列化结果。"""
    key = id(config)
    if key not in _PARSED:
        _PARSED[key] = parse_layer_experts(getattr(config, "p2p_layer_experts", None),
                                           int(config.num_hidden_layers),
                                           int(config.num_experts))
    return _PARSED[key]


# --------------------------------------------------------------------------- #
class PartialQwen3MoeSparseMoeBlock(_q.Qwen3MoeSparseMoeBlock):
    """一层 MoE。层号在驻留表里 → 只建 n 个专家；不在 → 完全走原生构造。

    继承原生类是为了让 `Qwen3MoeForCausalLM.__init__` 里的
    `isinstance(layer.mlp, Qwen3MoeSparseMoeBlock)` 照常成立（它靠这个找 MoE 层）。
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        layer = extract_layer_index(prefix)
        resident = _layer_experts_of(config).get(layer)
        st = _p2p_stats.get_collector()
        if resident is None:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            self.resident: tuple[int, ...] | None = None
            self.p2p_slot: int | None = None
            self.p2p_dense = _pick_dense(self, config)
            if st is not None:
                if self.is_sequence_parallel or self.enable_eplb or self.ep_size != 1:
                    raise ValueError("A5 统计钩子不支持 sequence-parallel MoE / EP / EPLB")
                self.p2p_slot = st.register_layer(layer, None)
                self.register_buffer(
                    "resident_idx", torch.arange(int(config.num_experts), dtype=torch.long),
                    persistent=False)
            return

        nn.Module.__init__(self)
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        # 下面这几种并行方式都会再切一遍专家 / token，与「本层只有这 n 个」的语义冲突
        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_group = _q.get_ep_group().device_group
        self.ep_rank = _q.get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        if self.ep_size != 1 or parallel_config.enable_eplb:
            raise ValueError("部分专家层不支持专家并行 / EPLB（它们会再切一遍专家）")
        if parallel_config.use_sequence_parallel_moe:
            raise ValueError("部分专家层不支持 sequence-parallel MoE")
        if quant_config is not None:
            raise ValueError("部分专家层暂不支持量化 checkpoint")

        self.is_sequence_parallel = False
        self.enable_eplb = False
        self.resident = tuple(resident)
        n = len(self.resident)
        E = int(config.num_experts)
        self.n_routed_experts = E
        self.n_logical_experts = E
        self.n_redundant_experts = 0
        self.n_physical_experts = E
        self.n_local_physical_experts = n
        self.physical_expert_start = 0
        self.physical_expert_end = n

        # 对 FusedMoE 来说这一层就是一个 n 专家的 MoE：权重只分配 n 份。
        # renormalize 恒为 True —— local_topk 的定义里就有重归一（见模块文档）。
        self.experts = FusedMoE(
            num_experts=n,
            top_k=min(int(config.num_experts_per_tok), n),
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=True,
            renormalize=True,
            quant_config=None,
            prefix=f"{prefix}.experts",
            routing_method_type=RoutingMethodType.Renormalize,
        )
        # 路由器保持全量 E 维：统计口径永远是「路由真正想要谁」
        self.gate = ReplicatedLinear(
            config.hidden_size, E, bias=False, quant_config=None, prefix=f"{prefix}.gate",
        )
        # 本地号 i ↔ 全局号 resident[i]（升序）。权重加载用的是同一个顺序
        # （layout.localize_expert_name），两处必须一致。
        self.register_buffer(
            "resident_idx", torch.tensor(self.resident, dtype=torch.long), persistent=False)
        self.p2p_slot = st.register_layer(layer, self.resident) if st is not None else None
        self.p2p_dense = _pick_dense(self, config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.resident is None and self.p2p_slot is None and not self.p2p_dense:
            return super().forward(hidden_states)
        assert hidden_states.dim() <= 2
        is_input_1d = hidden_states.dim() == 1
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits, _ = self.gate(hidden_states)                 # [T, E] 全量
        if self.p2p_slot is not None:
            local_logits = torch.ops.vllm.p2p_route_select(
                router_logits, self.resident_idx, self.p2p_slot)
        elif self.resident is not None:
            local_logits = router_logits.index_select(-1, self.resident_idx)   # [T, n]
        else:
            local_logits = router_logits                            # 装全的层
        if self.p2p_dense:
            ex = self.experts
            out = torch.ops.vllm.p2p_dense_moe(hidden_states, local_logits, ex.w13_weight,
                                               ex.w2_weight, int(ex.top_k), bool(ex.renormalize))
        else:
            out = self.experts(hidden_states=hidden_states, router_logits=local_logits)
        return out.squeeze(0) if is_input_1d else out


@contextmanager
def _partial_blocks():
    """构造期间把 qwen3_moe 模块里的 MoE 块换成部分专家版。

    `Qwen3MoeDecoderLayer.__init__` 按**模块全局名**取 `Qwen3MoeSparseMoeBlock`，所以
    换掉这个名字就能在权重分配之前接管 —— 不必复制一整份 DecoderLayer / Model 的代码
    （复制的话 vLLM 升级时两份会悄悄分叉）。只在本模型的构造期间换，退出即恢复，
    同一进程里的原生 Qwen3-MoE 不受影响。
    """
    orig = _q.Qwen3MoeSparseMoeBlock
    _q.Qwen3MoeSparseMoeBlock = PartialQwen3MoeSparseMoeBlock
    try:
        yield
    finally:
        _q.Qwen3MoeSparseMoeBlock = orig


@contextmanager
def _stage_layers(stage: "_p2p_stage.StageSpec | None"):
    """B-1：构造期间让 `Qwen3MoeModel` 只建本段（再按 PP 切）的层。

    同 `_partial_blocks` 的手法：`Qwen3MoeModel.__init__` 按模块全局名调 `make_layers`。
    段外的层是 `PPMissingLayer` —— vLLM 的 forward（islice start..end）、权重加载
    （is_pp_missing_parameter）、KV 缓存分配（只给建出来的 Attention）都已经认它。"""
    if stage is None:
        yield
        return
    orig = _q.make_layers

    def make_layers(num_hidden_layers, layer_fn, prefix):
        pp = get_pp_group()
        s, e = _p2p_stage.stage_layer_range(stage, pp.rank_in_group, pp.world_size)
        modules = nn.ModuleList(
            [PPMissingLayer() for _ in range(s)]
            + [maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{i}")) for i in range(s, e)]
            + [PPMissingLayer() for _ in range(e, num_hidden_layers)])
        return s, e, modules

    _q.make_layers = make_layers
    try:
        yield
    finally:
        _q.make_layers = orig


# --------------------------------------------------------------------------- #
class Qwen3MoePartialForCausalLM(_q.Qwen3MoeForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        policy = getattr(config, "p2p_miss_policy", "local_topk")
        if policy not in SUPPORTED_MISS_POLICIES:
            raise ValueError(f"p2p_miss_policy={policy!r} 暂不支持；可选 {SUPPORTED_MISS_POLICIES}")
        try:
            self_le = _layer_experts_of(config)
        except LayoutError as e:
            raise ValueError(f"hf_overrides 的 p2p_layer_experts 有误：{e}") from e
        try:
            stage = _p2p_stage.stage_of(config)
        except ValueError as e:
            raise ValueError(f"hf_overrides 的 p2p_stage 有误：{e}") from e
        st = self._make_stats(vllm_config, config)
        _p2p_stats.set_collector(st)
        with _partial_blocks(), _stage_layers(stage):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.p2p_layer_experts = self_le
        self.p2p_miss_policy = policy
        self.p2p_stage = stage
        if st is not None:
            sc = vllm_config.scheduler_config
            cc = vllm_config.compilation_config
            cap = max(int(sc.max_num_batched_tokens),
                      int(getattr(cc, "max_cudagraph_capture_size", None) or 0)) + 16
            dev = (torch.device("cuda", torch.cuda.current_device())
                   if torch.cuda.is_available() else torch.device("cpu"))
            st.allocate(cap, dev)
            logger.info("p2p 统计钩子：%d 个 MoE 层（%s），LR 层 %s，写 %s",
                        len(st.layers), f"{st.layers[0]}–{st.layers[-1]}" if st.layers else "无",
                        [l for l in st.lr_layers if l in st.layers] or "无", st.path)

        # PP：只建这一级用得到的（见模块文档「PP 各级只建自己用得到的东西」）
        pp = get_pp_group()
        tied = bool(config.tie_word_embeddings)
        if not tied:
            if not pp.is_first_rank:
                self.model.embed_tokens = PPMissingLayer()
            if not pp.is_last_rank:
                self.lm_head = PPMissingLayer()
        if not pp.is_last_rank:
            self.model.norm = PPMissingLayer()
        if stage is not None and stage.role in _p2p_stage.EMITS:
            # 前段 / 中段不出 logits：交出残差流（h + r），lm_head 不建
            if pp.is_last_rank:
                self.model.norm = _p2p_stage.AddResidual()
            self.lm_head = PPMissingLayer()
        if stage is not None and stage.role == "mid" and not pp.is_first_rank:
            # 中段不收 embed 权重（tie_word_embeddings 时上面没换掉，这里补上）
            self.model.embed_tokens = PPMissingLayer()
        if stage is not None and stage.role in _p2p_stage.FED and pp.is_first_rank:
            if tied and stage.role == "back":
                raise ValueError("后段不能用 tie_word_embeddings 的模型（lm_head 借的是 embed 权重）")
            self.model.embed_tokens = _p2p_stage.ZeroEmbed(
                config.hidden_size, vllm_config.model_config.dtype)
        if stage is not None:
            logger.info("p2p 分段引擎：%s，本段全局层 [%d, %d)，本 PP 级 [%d, %d)",
                        stage.role, stage.lo, stage.hi,
                        self.model.start_layer, self.model.end_layer)

        here = range(self.model.start_layer, self.model.end_layer)
        part = [l for l in here if l in self_le]
        if part:
            avg = sum(len(self_le[l]) for l in part) / len(part)
            logger.info(
                "p2p 部分专家：%s 层 [%d, %d)，其中 %d 层只装子集（层均 %.1f/%d 个专家），"
                "miss_policy=%s", getattr(config, "p2p_channel", ""), here.start, here.stop,
                len(part), avg, int(config.num_experts), policy)

    @staticmethod
    def _make_stats(vllm_config: VllmConfig, config) -> "_p2p_stats.RouteStats | None":
        raw = getattr(config, "p2p_stats", None)
        if not raw:
            return None
        raw = raw if isinstance(raw, dict) else {}
        return _p2p_stats.RouteStats(
            n_experts=int(config.num_experts), top_k=int(config.num_experts_per_tok),
            norm_topk=bool(getattr(config, "norm_topk_prob", True)),
            lr_layers=raw.get("lr_layers", ()), channel=getattr(config, "p2p_channel", ""),
            pp_rank=get_pp_group().rank_in_group,
            # B-1 前段：prompt 算完时把 LR 特征随输出交给协调器（不落盘、decode 步不拷贝）
            emit_lr=bool(raw.get("emit_lr", False)), write=bool(raw.get("write", True)),
            prefill_only=bool(raw.get("prefill_only", False)))

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if isinstance(self.lm_head, PPMissingLayer):
            return None
        return super().compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        le = self.p2p_layer_experts
        lo, hi = self.model.start_layer, self.model.end_layer
        seen: dict[int, set[tuple[int, str]]] = defaultdict(set)

        stage = self.p2p_stage
        # 分段引擎换掉了的模块：checkpoint 里有，但本段不收（norm/lm_head 在后段，embed 在前段）
        skip = (() if stage is None else
                {"front": ("model.norm.", "lm_head."), "back": ("model.embed_tokens.",),
                 "mid": ("model.norm.", "lm_head.", "model.embed_tokens.")}[stage.role])

        def localized():
            for name, w in weights:
                if skip and name.startswith(skip):
                    continue
                new = localize_expert_name(name, le)
                if new is None:
                    continue              # 不驻留的专家：不读进显存
                k = expert_key(new)        # 已换成本地号：(层, 本地专家号, 投影)
                if k is not None and lo <= k[0] < hi:
                    seen[k[0]].add((k[1], k[2]))
                yield new, w

        loaded = super().load_weights(localized())

        # 逐专家核对：本级每个 MoE 层的每个本地专家都要有 gate/up/down 三份
        bad = []
        for l in range(lo, hi):
            if not isinstance(self.model.layers[l].mlp, _q.Qwen3MoeSparseMoeBlock):
                continue                  # mlp_only_layers：稠密层，没有专家
            n = len(le[l]) if l in le else int(self.config.num_experts)
            want = {(i, s) for i in range(n) for s in ("gate_proj", "up_proj", "down_proj")}
            miss = want - seen.get(l, set())
            if miss:
                gl = (lambda i: le[l][i]) if l in le else (lambda i: i)
                ex = sorted({gl(i) for i, _ in miss})
                bad.append(f"第 {l} 层（0-based）缺 {len(miss)} 份专家权重，全局专家号 {ex[:8]}"
                           + ("…" if len(ex) > 8 else ""))
        if bad:
            raise ValueError(
                "checkpoint 里缺驻留专家的权重 —— 不拦的话那几行是未初始化的显存，"
                "输出会悄悄变差：\n  " + "\n  ".join(bad[:6])
                + "\n这台的部分 checkpoint 多半是按别的驻留集拉的：用同一份清单重跑 "
                  "p2pmoe.deploy.fetch")
        return loaded

