# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B1：一个分段引擎（前段 / 后段）在本机的代理。由 `launch.py --channels stages.json start …` 拉起。

    # 后段 gsm8k-0：N03（node_rank 0，起 EngineCore + 代理）+ N05（node_rank 1，headless worker）
    VLLM_PP_LAYER_PARTITION=13,27 <venv>/bin/python -m vllm.p2p.stage_agent \\
        --served-model-name back-gsm8k-0 --model <weights> --role back --layers 8,48 \\
        --pipeline-parallel-size 2 --nnodes 2 --node-rank 0 --master-addr 192.168.x.y \\
        --master-port 29600 --dtype half --max-model-len 4096 --gpu-memory-utilization 0.95 \\
        --attention-backend TRITON_ATTN --port 8300 --hf-overrides '{…}'

node_rank 0：起 vLLM 的 EngineCore（`AsyncMPClient`，PP 多机时 worker 分布在各节点），
引擎就绪后开 HTTP（`/health`、`/info`，端口 --port）与 ZMQ ROUTER（端口 --port + 1，协议见
`remote.py`）。node_rank ≥ 1：只起 headless worker（与 `vllm serve --headless` 同一条路径），
引擎参数由同一段代码生成，保证各节点逐字一致。

`--hf-overrides` 直接给 JSON（不读文件：各节点不必同步 overrides 文件），本程序再加上
`p2p_stage = {role, layers}`。

前段本机 LR：hf_overrides 里带 `p2p_lr`（`colorplan.py --lr-model …` 生成）时，前段代理在本机
加载 p2p-framework 的 LR 选择器，prompt 算完那一批输出里的前 L₀ 层统计在本机判出 task，只把
判定结果交给协调器（网关 `--lr-at front`）。`p2p_lr` 不进模型配置（这里先拿掉）：

    "p2p_lr": {"model": "<selector .npz>", "tasks": {"gsm8k": 1, …}, "p2p_framework": "<目录>",
               "classes": null, "features": null, "feature_fn": null, "tau_hi": null, "tau_lo": 0.4}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

__all__ = ["build_engine_args", "main"]


def build_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--served-model-name", required=True, help="分段引擎名（launch 按它认进程）")
    ap.add_argument("--model", required=True)
    ap.add_argument("--role", required=True, choices=("front", "back"))
    ap.add_argument("--layers", required=True, help="0-based 半开区间 lo,hi")
    ap.add_argument("--task", default=None, help="后段服务的 task（只写进 /info）")
    ap.add_argument("--pipeline-parallel-size", type=int, default=1)
    ap.add_argument("--nnodes", type=int, default=1)
    ap.add_argument("--node-rank", type=int, default=0)
    ap.add_argument("--master-addr", default="127.0.0.1")
    ap.add_argument("--master-port", type=int, default=29600)
    ap.add_argument("--dtype", default="half")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--max-num-batched-tokens", type=int, default=None)
    ap.add_argument("--kv-cache-memory-bytes", type=int, default=None)
    ap.add_argument("--attention-backend", default="TRITON_ATTN")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--port", type=int, default=8300, help="HTTP 端口；ZMQ 用它 + 1")
    ap.add_argument("--hf-overrides", required=True, help="JSON（本程序再加 p2p_stage）")
    return ap.parse_args(argv)


def build_engine_args(a: argparse.Namespace):
    from vllm.engine.arg_utils import EngineArgs

    lo, hi = (int(x) for x in a.layers.split(","))
    ov = json.loads(a.hf_overrides)
    ov.pop("p2p_lr", None)              # 前段本机 LR 的配置，代理自己用，不进模型配置
    ov["p2p_stage"] = {"role": a.role, "layers": [lo, hi]}
    kw = dict(
        model=a.model, served_model_name=[a.served_model_name], dtype=a.dtype, hf_overrides=ov,
        skip_tokenizer_init=True, enable_prompt_embeds=(a.role == "back"),
        async_scheduling=False, enable_prefix_caching=False,
        gpu_memory_utilization=a.gpu_memory_utilization, max_model_len=a.max_model_len,
        pipeline_parallel_size=a.pipeline_parallel_size, enforce_eager=a.enforce_eager, seed=0)
    if a.nnodes > 1:
        kw.update(nnodes=a.nnodes, node_rank=a.node_rank, master_addr=a.master_addr,
                  master_port=a.master_port, distributed_executor_backend="mp")
    if a.max_num_seqs:
        kw["max_num_seqs"] = a.max_num_seqs
    if a.max_num_batched_tokens:
        kw["max_num_batched_tokens"] = a.max_num_batched_tokens
    if a.kv_cache_memory_bytes:
        kw["kv_cache_memory_bytes"] = a.kv_cache_memory_bytes
    if a.attention_backend:
        kw["attention_backend"] = a.attention_backend
    return EngineArgs(**kw)


def make_front_lr(cfg: dict, *, l0: int, n_experts: int):
    """→ (feat_hook, 写进 /info 的描述)。前段代理在本机跑 LR：`feat_hook(前段交出的统计)` →
    `{"verdict": 判定}`（与网关里 `make_lr_classify` 同一个选择器、同一套阈值、同一格式）。"""
    from pathlib import Path

    from vllm.p2p.identifier import use_p2p_framework
    from vllm.p2p.tiered_gateway import make_lr_classify

    use_p2p_framework(cfg.get("p2p_framework"))
    tasks = {str(k): float(v) for k, v in (cfg.get("tasks") or {}).items()}
    if not tasks:
        raise SystemExit("p2p_lr 里没有 tasks（部署的 task 与先验）")
    classify, clf = make_lr_classify(
        lr_model=cfg["model"], l0=int(l0), n_experts=int(n_experts), tasks=tasks,
        lr_classes=cfg.get("classes"), lr_features=cfg.get("features"),
        lr_feature_fn=cfg.get("feature_fn"), tau_hi=cfg.get("tau_hi"),
        tau_lo=float(cfg.get("tau_lo", 0.40)))

    def hook(feat: dict) -> dict:
        if isinstance(feat, dict) and "verdict" in feat:      # 已经判过（不会发生，防重入）
            return feat
        t0 = time.perf_counter()
        try:
            v = dict(classify(feat, {}))
            for k in ("n_prompt", "n_body"):                   # 这两个由协调器按请求补
                v.pop(k, None)
        except Exception as e:                                 # noqa: BLE001 —— 判不出也不能卡住请求
            v = {"task": None, "error": f"{type(e).__name__}: {e}"}
        v["lr_ms_front"] = round((time.perf_counter() - t0) * 1e3, 3)
        return {"verdict": v}

    desc = {"model": Path(cfg["model"]).name, "needs_body": bool(clf.needs_body),
            "classifier": clf.describe(), "tasks": tasks}
    return hook, desc


def run_headless(a: argparse.Namespace) -> int:
    """node_rank ≥ 1：只起本节点的 worker，等首节点的 EngineCore 连过来（同 vllm serve --headless）。"""
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    cfg = build_engine_args(a).create_engine_config(UsageContext.OPENAI_API_SERVER, headless=True)
    print(f"[stage_agent] {a.served_model_name} node_rank {a.node_rank}：headless worker，"
          f"首节点 {a.master_addr}:{a.master_port}", flush=True)
    ex = MultiprocExecutor(cfg, monitor_workers=False)
    ex.start_worker_monitor(inline=True)
    return 0


async def run_head(a: argparse.Namespace) -> int:
    from aiohttp import web

    from vllm.p2p.remote import AgentBridge
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.core_client import EngineCoreClient
    from vllm.v1.executor import Executor

    t0 = time.perf_counter()
    # 与 A 的 vllm serve 同一个 usage context：默认的 max_num_batched_tokens 等才一致
    cfg = build_engine_args(a).create_engine_config(UsageContext.OPENAI_API_SERVER)
    print(f"[stage_agent] {a.served_model_name}：{a.role} 层 {a.layers}，PP={a.pipeline_parallel_size}"
          f"（{a.nnodes} 节点），起 EngineCore …", flush=True)
    client = EngineCoreClient.make_async_mp_client(cfg, Executor.get_class(cfg), log_stats=False)
    mc = cfg.model_config
    info = {"name": a.served_model_name, "role": a.role,
            "layers": [int(x) for x in a.layers.split(",")], "task": a.task,
            "hidden_size": mc.get_hidden_size(), "dtype": str(mc.dtype).removeprefix("torch."),
            "max_num_seqs": cfg.scheduler_config.max_num_seqs,
            "max_model_len": mc.max_model_len, "pp": a.pipeline_parallel_size,
            "kv_blocks": cfg.cache_config.num_gpu_blocks,
            "block_size": cfg.cache_config.block_size, "startup_s": None}
    info["startup_s"] = round(time.perf_counter() - t0, 1)
    print(f"[stage_agent] EngineCore 就绪（{info['startup_s']}s）；KV {info['kv_blocks']} 块 × "
          f"{info['block_size']}；ZMQ tcp://*:{a.port + 1}，HTTP :{a.port}", flush=True)

    hook = None
    lr_cfg = json.loads(a.hf_overrides).get("p2p_lr")
    if lr_cfg:
        if a.role != "front":
            raise SystemExit("p2p_lr 只能给前段（含第 L₀ 层、交出识别统计的那一段）")
        lo, hi = (int(x) for x in a.layers.split(","))
        hook, info["lr"] = make_front_lr(lr_cfg, l0=hi,
                                         n_experts=int(getattr(mc.hf_config, "num_experts")))
        print(f"[stage_agent] 前段本机 LR：{info['lr']['classifier']}；task 与先验 {info['lr']['tasks']}",
              flush=True)

    bridge = AgentBridge(client, f"tcp://*:{a.port + 1}", info, feat_hook=hook)

    async def health(_):
        if bridge.dead:
            return web.json_response({"ok": False, "error": bridge.dead}, status=500)
        return web.json_response({"ok": True, "name": a.served_model_name})

    async def get_info(_):
        return web.json_response({**info, "msgs_in": bridge.n_in, "batches_out": bridge.n_out,
                                  "dead": bridge.dead})

    app = web.Application()
    app.add_routes([web.get("/health", health), web.get("/info", get_info)])
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", a.port).start()
    try:
        await bridge.run()
    finally:
        client.shutdown()
    return 1


def main(argv: list[str] | None = None) -> int:
    a = build_args(argv)
    # 引擎 / worker 进程用 spawn 起（见 tiered.py：父进程里起过 torch 线程池时 fork 会卡住）
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if a.node_rank > 0:
        return run_headless(a)
    return asyncio.run(run_head(a))


if __name__ == "__main__":
    sys.exit(main())
