# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""单卡量 MoE 一层（只算专家部分）各种实现的耗时：找 Turing（RTX 6000 / T4）上慢在哪。

layer_bench 量出来 RTX 6000 每层比 A30 慢 10–25 倍（prefill 折合只有约 0.7 TFLOPS），远超硬件差距。
这里把 MoE 单独拿出来，同一组随机权重 / 路由比几种算法：

* `triton`：vLLM 部署时用的 `fused_experts`（Triton，默认配置 —— 部分专家层的 E 没有调好的配置文件）；
* `triton*`：`--tune` 时在一组 Triton 配置里挑最快的（看是不是配置的问题）；
* `dense`：每个 token 过本层全部 E 个驻留专家，cuBLAS 批量矩阵乘（Turing 上走 tensor core），
  再按路由权重（非 top-k 处为 0）加权求和 —— 多算 E/top-k 倍，但形状固定、能进 CUDA graph；
* `loop`：按专家分组、每个专家一次 cuBLAS 矩阵乘（形状随路由变，只作参照）。

另外量一次大的稠密 GEMM，确认这张卡的 tensor core 本身是正常的（报 TFLOPS）。
`dense` / `loop` 与 `triton` 的输出逐元素比对（fp16 容差）。

    cd /home/ubuntu/mbf
    PYTHONPATH=/home/ubuntu/mbf /home/ubuntu/venvs/vllm-p2p/bin/python -m vllm.p2p.moe_kernel_bench \\
        --experts 64 --tokens 1,32,64,128,256,320 --tune
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time


def _timeit(fn, iters: int, warmup: int = 3) -> float:
    """→ 每次调用的毫秒数（CUDA event，取均值）。"""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def make_inputs(E: int, T: int, H: int, inter: int, topk: int, seed: int = 0):
    import torch

    g = torch.Generator(device="cuda").manual_seed(seed)
    dt = torch.float16
    w13 = (torch.randn(E, 2 * inter, H, generator=g, device="cuda", dtype=torch.float32) * 0.02).to(dt)
    w2 = (torch.randn(E, H, inter, generator=g, device="cuda", dtype=torch.float32) * 0.02).to(dt)
    x = torch.randn(T, H, generator=g, device="cuda", dtype=torch.float32).to(dt)
    logits = torch.randn(T, E, generator=g, device="cuda", dtype=torch.float32)
    p = logits.softmax(-1)
    tw, ti = p.topk(min(topk, E), dim=-1)
    tw = (tw / tw.sum(-1, keepdim=True)).float()
    return x, w13, w2, tw, ti.to(torch.int32)


def moe_dense(x, w13, w2, tw, ti):
    """每个 token 过全部 E 个专家（cuBLAS bmm），按路由权重加权求和。"""
    import torch
    import torch.nn.functional as F

    E, T = w13.shape[0], x.shape[0]
    inter = w2.shape[2]
    h = torch.bmm(x.unsqueeze(0).expand(E, -1, -1), w13.transpose(1, 2))      # [E, T, 2I]
    a = F.silu(h[..., :inter]) * h[..., inter:]                                 # [E, T, I]
    y = torch.bmm(a, w2.transpose(1, 2))                                        # [E, T, H]
    wd = torch.zeros(T, E, device=x.device, dtype=torch.float32).scatter_(1, ti.long(), tw)
    return torch.einsum("te,eth->th", wd.to(x.dtype), y)                       # [T, H]


def moe_loop(x, w13, w2, tw, ti):
    """按专家分组，每个专家一次矩阵乘（形状随路由变）。"""
    import torch
    import torch.nn.functional as F

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
        if c == 0:
            continue
        idx = tok[s:s + c]
        h = x[idx] @ w13[e].t()
        a = F.silu(h[:, :inter]) * h[:, inter:]
        y = a @ w2[e].t()
        out.index_add_(0, idx, y.float() * wt[s:s + c, None])
        s += c
    return out.to(x.dtype)


def moe_triton(x, w13, w2, tw, ti):
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    return fused_experts(x, w13, w2, tw, ti)


TUNE_SPACE = {"BLOCK_SIZE_M": (16, 32, 64), "BLOCK_SIZE_N": (32, 64, 128), "BLOCK_SIZE_K": (32, 64),
              "GROUP_SIZE_M": (1, 8), "num_warps": (4, 8), "num_stages": (2, 3)}


def tune_triton(x, w13, w2, tw, ti, iters: int) -> tuple[float, dict | None]:
    from vllm.model_executor.layers.fused_moe import override_config

    best = (float("inf"), None)
    keys = list(TUNE_SPACE)
    for vals in itertools.product(*(TUNE_SPACE[k] for k in keys)):
        cfg = dict(zip(keys, vals))
        cfg["SPLIT_K"] = 1
        try:
            with override_config(cfg):
                ms = _timeit(lambda: moe_triton(x, w13, w2, tw, ti), iters, warmup=2)
        except Exception:      # noqa: BLE001 — 资源不够（共享内存等）的配置跳过
            continue
        if ms < best[0]:
            best = (ms, cfg)
    return best


def gemm_tflops(n: int = 4096, iters: int = 20) -> float:
    import torch

    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    ms = _timeit(lambda: a @ b, iters)
    return 2 * n ** 3 / (ms / 1e3) / 1e12


def moe_flops(T: int, H: int, inter: int, topk: int) -> float:
    """真正要算的量：每个 token top-k 个专家 × (gate+up+down)。"""
    return 2.0 * T * topk * 3 * H * inter


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--experts", type=int, default=64, help="本层驻留专家数 E")
    ap.add_argument("--tokens", default="1,32,64,128,256,320", help="一步的 token 数（decode = 在途条数）")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=768, help="moe_intermediate_size")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--tune", action="store_true", help="另在一组 Triton 配置里挑最快的")
    ap.add_argument("--no-loop", action="store_true", help="不量 loop（它最慢）")
    ap.add_argument("--json", default=None, help="结果另写一份 JSON")
    a = ap.parse_args(argv)

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("要 GPU")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    tf = gemm_tflops()
    print(f"{name}（sm{cap[0]}{cap[1]}），稠密 fp16 GEMM 4096³：{tf:.1f} TFLOPS；"
          f"E={a.experts} H={a.hidden} I={a.inter} top-{a.topk}", flush=True)
    rows = []
    hdr = f"{'T':>5} {'triton':>9} " + ("{:>9} ".format("triton*") if a.tune else "") + \
          f"{'dense':>9} " + ("" if a.no_loop else f"{'loop':>9} ") + " 有效 TFLOPS(triton/dense)  误差(dense)"
    print(hdr, flush=True)
    for T in [int(t) for t in a.tokens.split(",") if t]:
        x, w13, w2, tw, ti = make_inputs(a.experts, T, a.hidden, a.inter, a.topk)
        ref = moe_triton(x, w13, w2, tw, ti)
        t_tri = _timeit(lambda: moe_triton(x, w13, w2, tw, ti), a.iters)
        t_tune, cfg = tune_triton(x, w13, w2, tw, ti, max(5, a.iters // 4)) if a.tune else (None, None)
        den = moe_dense(x, w13, w2, tw, ti)
        err = (den.float() - ref.float()).abs().max().item() / max(ref.float().abs().max().item(), 1e-6)
        t_den = _timeit(lambda: moe_dense(x, w13, w2, tw, ti), a.iters)
        t_loop = None if a.no_loop else _timeit(lambda: moe_loop(x, w13, w2, tw, ti), max(3, a.iters // 4))
        fl = moe_flops(T, a.hidden, a.inter, min(a.topk, a.experts))
        r = {"T": T, "triton_ms": round(t_tri, 4), "triton_tuned_ms": None if t_tune is None else round(t_tune, 4),
             "triton_tuned_cfg": cfg, "dense_ms": round(t_den, 4),
             "loop_ms": None if t_loop is None else round(t_loop, 4), "dense_rel_err": round(err, 5)}
        rows.append(r)
        print(f"{T:>5} {t_tri:>9.3f} " + (f"{t_tune:>9.3f} " if a.tune else "") + f"{t_den:>9.3f} "
              + ("" if a.no_loop else f"{t_loop:>9.3f} ")
              + f" {fl / t_tri / 1e9:8.2f} / {fl / t_den / 1e9:8.2f}            {err:.1e}"
              + (f"   最快配置 {json.dumps(cfg)}" if cfg else ""), flush=True)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"gpu": name, "sm": f"{cap[0]}{cap[1]}", "gemm_tflops": round(tf, 2),
                       "E": a.experts, "H": a.hidden, "I": a.inter, "topk": a.topk, "rows": rows,
                       "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
