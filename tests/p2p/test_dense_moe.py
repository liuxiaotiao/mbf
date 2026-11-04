# SPDX-License-Identifier: Apache-2.0
"""Turing 上的 MoE 路径（qwen3_moe_partial.p2p_dense_moe）：与逐 token 的参考实现一致。"""
from types import SimpleNamespace

import pytest
import torch

m = pytest.importorskip("vllm.model_executor.models.qwen3_moe_partial")


def _ref(x, w13, w2, logits, k, renorm):
    p = logits.float().softmax(-1)
    tw, ti = p.topk(k, -1)
    if renorm:
        tw = tw / tw.sum(-1, keepdim=True)
    inter = w2.shape[2]
    out = torch.zeros_like(x, dtype=torch.float32)
    for t in range(x.shape[0]):
        for j in range(k):
            e = int(ti[t, j])
            h = w13[e].float() @ x[t].float()
            a = torch.nn.functional.silu(h[:inter]) * h[inter:]
            out[t] += tw[t, j] * (w2[e].float() @ a)
    return out


@pytest.mark.parametrize("renorm", [True, False])
@pytest.mark.parametrize("T", [1, 7, 33])
def test_dense_and_grouped_match_reference(monkeypatch, T, renorm) -> None:
    torch.manual_seed(T)
    E, H, inter, k = 9, 16, 8, 3
    w13 = torch.randn(E, 2 * inter, H) * 0.2
    w2 = torch.randn(E, H, inter) * 0.2
    x = torch.randn(T, H)
    logits = torch.randn(T, E)
    ref = _ref(x, w13, w2, logits, k, renorm)
    tw, ti = m.route_topk(logits, k, renorm)
    assert torch.allclose(m.moe_dense(x, w13, w2, tw, ti), ref, atol=1e-5)
    assert torch.allclose(m.moe_grouped(x, w13, w2, tw, ti), ref, atol=1e-5)
    # 算子按 T 选路径：阈值以下 dense、以上分组 —— 两边结果都对
    for cap in ("1000", "4"):
        monkeypatch.setenv("VLLM_P2P_DENSE_MOE_MAX_TOKENS", cap)
        assert torch.allclose(m.p2p_dense_moe(x, logits, w13, w2, k, renorm), ref, atol=1e-5)
    assert m.p2p_dense_moe(x[:0], logits[:0], w13, w2, k, renorm).shape == (0, H)


def test_resolve_moe_impl(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_P2P_MOE_IMPL", raising=False)
    cfg = SimpleNamespace()
    assert m.resolve_moe_impl(cfg, (7, 5)) == "triton"          # 默认不开（后续优化项）
    auto = SimpleNamespace(p2p_moe_impl="auto")
    assert m.resolve_moe_impl(auto, (7, 5)) == "dense"          # Turing
    assert m.resolve_moe_impl(auto, (8, 0)) == "triton"         # A30
    assert m.resolve_moe_impl(auto, None) == "triton"
    assert m.resolve_moe_impl(SimpleNamespace(p2p_moe_impl="triton"), (7, 5)) == "triton"
    assert m.resolve_moe_impl(SimpleNamespace(p2p_moe_impl="dense"), (8, 6)) == "dense"
    monkeypatch.setenv("VLLM_P2P_MOE_IMPL", "triton")           # 环境变量优先
    assert m.resolve_moe_impl(SimpleNamespace(p2p_moe_impl="dense"), (7, 5)) == "triton"
    monkeypatch.setenv("VLLM_P2P_MOE_IMPL", "bogus")
    with pytest.raises(ValueError):
        m.resolve_moe_impl(cfg, (7, 5))
