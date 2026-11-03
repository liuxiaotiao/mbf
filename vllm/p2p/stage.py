# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B-1 分段引擎（stage engine）的公共件：前段 = 第 1..L0 层，后段 = 第 L0+1..L 层。

一条请求在两个**独立的** vLLM 引擎上跑：前段引擎（全专家的前 L0 层，所有 task 共用）和它
被绑定的那个后段引擎（某个 task 的部分专家后 L−L0 层）。两个引擎各有自己的调度器、KV
缓存和连续批处理；一个协调器（`vllm/p2p/tiered.py`）在它们之间搬两样东西：

    后段采样出的 token  ──p2p_append──▶  前段（把它当作下一个输入 token，算出这一位置的 hidden）
    前段算出的 hidden   ──p2p_feed────▶  后段（把它当作这一位置的输入 embedding，采样下一个 token）

配置（`--hf-overrides` 写进 HF config，模型 / runner / 调度器都从这里读）：

    "p2p_stage": {"role": "front", "layers": [0, 8]}      # 0-based 半开区间
    "p2p_stage": {"role": "mid",   "layers": [8, 22]}     # 可选：0 个或多个中段
    "p2p_stage": {"role": "back",  "layers": [22, 48]}

中段（多级）：输入像后段（prompt 的 hidden 随请求以 prompt_embeds 来，之后每个位置等上一段
`p2p_feed`），输出像前段（不采样，交出残差流给下一段）。中段不知道 token：p2p_feed 收到的
位置超过序列长度时，调度器补占位 token（`PLACEHOLDER_TOKEN`）让序列变长，runner 把这些位置的
输入换成收到的 hidden。一个 token 的一生：

    后段 token ─p2p_append─▶ 前段 ─p2p_feed─▶ 中段₁ ─p2p_feed─▶ … ─p2p_feed─▶ 后段

两段之间传的是**残差流**：前段最后一层输出 (hidden, residual) 后不做 final norm，而是
`hidden + residual`（`AddResidual`）；后段第一层收到 residual=None 时恰好做
`residual = x; x = input_layernorm(x)` —— 与整模型里那一层做的 fused add-rmsnorm 相同
（只差一次 fp16 舍入的位置）。

为什么前段不采样也能「卡住」请求：前段 runner 对每条请求都不返回采样 token，于是 prefill
完以后 `num_tokens == num_computed_tokens`，调度器的 running 循环算出 num_new_tokens = 0、
跳过它；直到协调器 `p2p_append` 一个 token 进来才又排上。后段则相反：每条请求有一个
`p2p_ready`（已经收到 hidden 的位置数），调度时 num_new_tokens 被它截住。

这里只放与 vLLM 内部无关、可以单测的东西。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from torch import nn

__all__ = [
    "StageSpec", "stage_of", "role_of", "stage_layer_range", "AddResidual", "ZeroEmbed",
    "RowBuffer", "pack_rows", "unpack_rows", "RowStream", "STAGE_KEY", "ROLES", "EMITS", "FED",
    "PLACEHOLDER_TOKEN",
]

STAGE_KEY = "p2p_stage"
ROLES = ("front", "mid", "back")
PLACEHOLDER_TOKEN = 0       # 中段补的占位 token（输入 embedding 会被换成上一段的 hidden）
EMITS = ("front", "mid")    # 交出 hidden、不采样
FED = ("mid", "back")       # 输入是上一段的 hidden（prompt_embeds + p2p_feed）


@dataclass(frozen=True)
class StageSpec:
    role: str            # "front" | "mid" | "back"
    lo: int              # 0-based，含
    hi: int              # 0-based，不含

    @property
    def n_layers(self) -> int:
        return self.hi - self.lo


def stage_of(cfg: Any, n_layers: int | None = None) -> StageSpec | None:
    """从 HF config（或一个 dict）读 `p2p_stage`；没有 → None（普通整模型）。"""
    raw = cfg.get(STAGE_KEY) if isinstance(cfg, dict) else getattr(cfg, STAGE_KEY, None)
    if not raw:
        return None
    if n_layers is None:
        n_layers = (cfg.get("num_hidden_layers") if isinstance(cfg, dict)
                    else getattr(cfg, "num_hidden_layers", None))
    role = raw.get("role")
    if role not in ROLES:
        raise ValueError(f"p2p_stage.role 必须是 {ROLES} 之一，给的是 {role!r}")
    try:
        lo, hi = (int(x) for x in raw["layers"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"p2p_stage.layers 要写成 [lo, hi]（0-based 半开），给的是 "
                         f"{raw.get('layers')!r}") from e
    if not 0 <= lo < hi:
        raise ValueError(f"p2p_stage.layers=[{lo}, {hi}) 是空的或倒过来了")
    if n_layers is not None:
        n_layers = int(n_layers)
        if hi > n_layers:
            raise ValueError(f"p2p_stage.layers=[{lo}, {hi}) 超出模型的 {n_layers} 层")
        if role == "back" and hi != n_layers:
            raise ValueError(f"后段必须一直到最后一层（hi={hi}，模型 {n_layers} 层）："
                             "lm_head 与 final norm 在后段")
    if role == "front" and lo != 0:
        raise ValueError(f"前段必须从第 0 层开始（lo={lo}）：embed_tokens 在前段")
    if role in FED and lo == 0:
        raise ValueError(f"{role} 段不能从第 0 层开始：第 0 层的输入是 token（在前段）")
    if role == "mid" and n_layers is not None and hi >= n_layers:
        raise ValueError(f"中段不能到最后一层（hi={hi}，模型 {n_layers} 层）：最后一层在后段")
    return StageSpec(role, lo, hi)


def role_of(cfg: Any) -> str | None:
    """'front' / 'mid' / 'back' / None。调度器与 runner 用它决定走不走分段逻辑。"""
    st = stage_of(cfg)
    return st.role if st else None


def stage_layer_range(spec: StageSpec, pp_rank: int, pp_size: int) -> tuple[int, int]:
    """本段的 [lo, hi) 再按 PP 切给各级（与 vLLM 的 get_pp_indices 同一种切法，
    `VLLM_PP_LAYER_PARTITION` 指的是**本段**的层数划分）。返回全局 0-based 层号。"""
    from vllm.distributed.utils import get_pp_indices

    s, e = get_pp_indices(spec.n_layers, pp_rank, pp_size)
    return spec.lo + s, spec.lo + e


class AddResidual(nn.Module):
    """前段的「final norm」：不归一化，把残差流交出去 —— `(h, r) → (h + r, None)`。"""

    def forward(self, hidden_states: torch.Tensor,
                residual: torch.Tensor | None = None) -> tuple[torch.Tensor, None]:
        if residual is None:
            return hidden_states, None
        return hidden_states + residual, None


class ZeroEmbed(nn.Module):
    """后段的 embed_tokens：后段的每个位置都由前段的 hidden 喂入，按理永远用不到 token
    embedding；但 vLLM 在 dummy run / 混合批里仍可能调它，给全 0（不占显存、不读权重）。"""

    def __init__(self, hidden_size: int, dtype: torch.dtype):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.dtype = dtype

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*input_ids.shape, self.hidden_size), dtype=self.dtype,
                           device=input_ids.device)


class RowBuffer:
    """按偏移写入、可增长的 [n, H] CPU 缓冲（后段 runner 存「prompt 之后各位置的 hidden」）。

    整条请求的生命周期里都留着：被抢占后从头重算时，prompt 之后的位置还要再用一遍。"""

    __slots__ = ("buf", "n")

    def __init__(self, width: int, dtype: torch.dtype, cap: int = 64):
        self.buf = torch.empty((cap, width), dtype=dtype)
        self.n = 0                      # 高水位：[0, n) 都写过

    def put(self, off: int, rows: torch.Tensor) -> None:
        end = off + rows.shape[0]
        if end > self.buf.shape[0]:
            cap = self.buf.shape[0]
            while cap < end:
                cap *= 2
            nb = torch.empty((cap, self.buf.shape[1]), dtype=self.buf.dtype)
            nb[: self.n] = self.buf[: self.n]
            self.buf = nb
        if off > self.n:
            raise ValueError(f"RowBuffer 不能留空洞：已有 {self.n} 行，要从 {off} 写")
        self.buf[off:end] = rows
        self.n = max(self.n, end)

    def get(self, lo: int, hi: int) -> torch.Tensor:
        if hi > self.n:
            raise IndexError(f"RowBuffer 只有 {self.n} 行，要读 [{lo}, {hi})")
        return self.buf[lo:hi]


# --------------------------------------------------------------------------- #
# 协调器 → 后段的 utility 参数：hidden 走 bytes（utility 参数没有类型信息，张量解不回来）
# --------------------------------------------------------------------------- #
def pack_rows(req_id: str, pos: int, rows: torch.Tensor) -> list:
    rows = rows.detach().contiguous().cpu()
    return [req_id, int(pos), int(rows.shape[0]), str(rows.dtype).removeprefix("torch."),
            rows.view(torch.uint8).numpy().tobytes()]


def unpack_rows(item: Sequence, width: int) -> tuple[str, int, torch.Tensor]:
    req_id, pos, n, dtype, data = item
    dt = getattr(torch, dtype)
    t = torch.frombuffer(bytearray(data), dtype=dt)
    if t.numel() != int(n) * width:
        raise ValueError(f"p2p_feed：{req_id} 的 {n} 行 × {width} 与数据长度 {t.numel()} 不符")
    return str(req_id), int(pos), t.view(int(n), width)


class RowStream:
    """协调器侧：一条请求从前段陆续收到的 hidden 行，按位置去重、拼接。

    前段被抢占后重算会把已经送过的位置再送一遍；分块 prefill 会分几步送 prompt 的行。"""

    __slots__ = ("next", "_parts")

    def __init__(self) -> None:
        self.next = 0                  # 下一个要的位置
        self._parts: list[torch.Tensor] = []

    def add(self, pos: int, rows: torch.Tensor) -> tuple[int, torch.Tensor] | None:
        """收下 [pos, pos+n) 里新的那部分 → (新段起点, 新行)；全是旧的 → None。"""
        end = pos + rows.shape[0]
        if end <= self.next:
            return None
        if pos > self.next:
            raise ValueError(f"前段的 hidden 跳了位置：要 {self.next}，来的是 {pos}")
        new = rows[self.next - pos:]
        start = self.next
        self.next = end
        return start, new

    def keep(self, rows: torch.Tensor) -> None:
        self._parts.append(rows)

    def take(self) -> torch.Tensor:
        out = self._parts[0] if len(self._parts) == 1 else torch.cat(self._parts, 0)
        self._parts = []
        return out


def items_by(items: Iterable[tuple[Any, list]]) -> dict[Any, list]:
    """[(key, item), …] → {key: [item, …]}（按引擎分批发 utility 用）。"""
    out: dict[Any, list] = {}
    for k, it in items:
        out.setdefault(k, []).append(it)
    return out
