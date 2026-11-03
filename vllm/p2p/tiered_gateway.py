# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B1 · 分段引擎的网关：OpenAI 兼容的 /v1/completions，后面是 B-1 的协调器。**按文件路径运行。**

    # 控制机上，vllm-p2p 的 venv；分段引擎先用 launch.py --channels stages.json start all 拉起
    python vllm/p2p/tiered_gateway.py --stages p2p_channels/stages.json \\
        --model-dir /home/ubuntu/p2p-framework/weights-q3-30b --port 9100

    # 压测与 A 完全同一个命令，只是 --channels 换成 stages.json
    python vllm/p2p/bench.py --gateway http://127.0.0.1:9100 --channels p2p_channels/stages.json \
        --cases … --model-dir … --out …

与 A 的网关（gateway.py）对外同一套接口，`bench.py` / `deploy_15.sh report` / `compare.py` 不用改：
`POST /v1/completions`（流式 SSE 或一次返回）、`GET /p2p/channels`、`GET /p2p/records`、
`POST /p2p/reset`、`GET /health`。不同的是请求的路径：

    进来 → 挑一个前段（任何 task 都行）→ 前段 prefill（顺带数前 L₀ 层的逐层路由）→ 定 task →
    在这个 task 的后段里挑在飞最少的绑定 → 之后每个 token 在前段与后段之间来回（见 tiered.py），
    前段、后段都不再变。

三种模式（与 A 的网关同名同义）：

* `oracle`：task 用请求自带的 `p2p_task`；
* `lr`（**B-1 的目标形态**）：task 由**前段出口识别**判 —— 前段在 prompt 算完的那一批输出里交出
  前 L₀ 层（只数 body token）的逐层 top-k 计数，网关里的协调器用 p2p-framework 的
  `LRClassifier`（与 A4 识别服务同一个选择器文件、同一套阈值）判出 task。不像 A4 要把前 L₀ 层
  单独再算一遍：前段本来就在算这几层，识别只多一次 LR（亚毫秒）。判给没有后段的类 / 低置信时
  与 p2p-framework 一样退到先验最大的池。请求里带的 `p2p_task` 只用来记对错；
* `shadow`：按 `p2p_task` 绑定，同时记下前段出口识别的结果（量准确率、不影响路由）。

    python vllm/p2p/tiered_gateway.py --stages … --model-dir … --mode lr \
        --lr-model /home/ubuntu/p2p-framework/selector/selector_body_L8_params.npz

  LR 放在前段（含第 L₀ 层的卡）本机跑：前段引擎的 hf_overrides 带 `p2p_lr`（`colorplan.py
  --lr-model …` 生成），网关加 `--lr-at front`，不再加载选择器，直接用前段交来的判定：

    python vllm/p2p/tiered_gateway.py --stages … --model-dir … --mode lr --lr-at front

只支持贪心（temperature 0）与 token id 的 prompt（bench 就是这么发的）。停止条件：模型的
EOS（generation_config 优先，同 vLLM 的默认）+ max_tokens + max_model_len。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Mapping, Sequence

if __package__ in (None, ""):                       # 按文件路径运行：把源码树放上 sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

__all__ = ["IncrementalDetok", "load_stop_ids", "make_lr_classify", "make_app", "main"]

MODES = ("oracle", "lr", "shadow")


def load_stop_ids(model_dir: str | Path) -> tuple[int | None, list[int]]:
    """→ (eos_token_id, 停止 token 列表)。generation_config 优先（Qwen3 对话模型是 <|im_end|>）。"""
    d = Path(model_dir)
    ids: list[int] = []
    first = None
    for name in ("generation_config.json", "config.json"):
        f = d / name
        if not f.exists():
            continue
        try:
            v = json.loads(f.read_text(encoding="utf-8")).get("eos_token_id")
        except (json.JSONDecodeError, OSError):
            continue
        vs = [v] if isinstance(v, int) else [x for x in (v or []) if isinstance(x, int)]
        for x in vs:
            if x not in ids:
                ids.append(int(x))
    if ids:
        first = ids[0]
    return first, ids


class IncrementalDetok:
    """增量解码（vLLM 的做法：只重解一个滑动窗口，不是整段）。半个 UTF-8 字符时先不吐。"""

    def __init__(self, tok, *, skip_special: bool = True):
        self.tok, self.skip = tok, skip_special
        self.ids: list[int] = []
        self.prefix_offset = 0
        self.read_offset = 0
        self.text = ""

    def _dec(self, ids) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=self.skip)

    def push(self, new: Sequence[int]) -> str:
        self.ids.extend(int(x) for x in new)
        prefix = self._dec(self.ids[self.prefix_offset:self.read_offset])
        full = self._dec(self.ids[self.prefix_offset:])
        if len(full) > len(prefix) and not full.endswith("�"):
            delta = full[len(prefix):]
            self.prefix_offset, self.read_offset = self.read_offset, len(self.ids)
            self.text += delta
            return delta
        return ""

    def flush(self) -> str:
        whole = self._dec(self.ids)
        delta = whole[len(self.text):] if whole.startswith(self.text) else ""
        self.text += delta
        return delta


def make_lr_classify(*, lr_model: str, l0: int, n_experts: int, tasks: Mapping[str, float],
                     lr_classes: str | None = None, lr_features: str | None = None,
                     lr_feature_fn: str | None = None, tau_hi: float | None = None,
                     tau_lo: float = 0.40):
    """→ (classify, clf)。`classify(feat, info)`：前段交出的 {层号(1-based): LayerStat 线上格式}
    → 与 A4 识别服务同格式的判定（task / conf / zone / scores / raw / label …）。
    要先把 p2p-framework 放上 sys.path（`identifier.use_p2p_framework`）。"""
    import numpy as np
    from p2pmoe.runtime.lr_classifier import LRClassifier, verdict_label
    from p2pmoe.runtime.model import LayerStat, MoEStats

    from vllm.p2p.identifier import verdict_wire

    clf = LRClassifier.load(lr_model, l0=int(l0), n_experts=int(n_experts), priors=dict(tasks),
                            class_map=lr_classes, features=lr_features,
                            feature_fn=lr_feature_fn, tau_hi=tau_hi, tau_lo=tau_lo)
    E = int(n_experts)

    def classify(feat: Mapping, info: Mapping) -> dict:
        st = MoEStats(hist=np.zeros(E, dtype=np.float64),
                      layers={int(l): LayerStat.from_wire(d) for l, d in feat.items()})
        v = clf.predict_stats(st)
        body = info.get("body")
        return {**verdict_wire(v), "label": verdict_label(v), "n_prompt": info.get("n_prompt"),
                "n_body": sum(int(b) - int(a) for a, b in body) if body else None}

    return classify, clf


def _n_experts(sj: Mapping, model_dir: str | Path) -> int:
    if sj.get("n_experts"):
        return int(sj["n_experts"])
    cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    return int(cfg["num_experts"])


def _finish_reason(f: str | None) -> str | None:
    if f is None:
        return None
    f = str(f).lower()
    return {"finishreason.stop": "stop", "finishreason.length": "length",
            "finishreason.abort": "abort"}.get(f, f)


def make_app(co, *, stages: Mapping[str, Mapping], tokenizer, mode: str = "oracle",
             max_model_len: int = 4096, needs_body: bool = False, identifier: str | None = None,
             stream_timeout: float = 3600.0):
    """co：已经 start() 过的 TieredCoordinator（lr / shadow 模式要带 classify）；
    stages：名字 → stages.json 里的条目。"""
    from aiohttp import web

    if mode not in MODES:
        raise ValueError(f"mode={mode!r}；可选 {MODES}")
    if mode in ("lr", "shadow") and co.classify is None:
        raise ValueError(f"{mode} 模式要协调器带 classify（前段出口识别）")

    records: list[dict] = []

    def bad(status: int, msg: str, rec: dict | None = None) -> web.Response:
        if rec is not None:
            rec.update(status=status, error=msg, t_done=time.time())
        return web.json_response({"error": {"message": msg, "code": status}}, status=status,
                                 dumps=lambda o: json.dumps(o, ensure_ascii=False))

    def chunk(cid, model, text, finish=None, stop_reason=None, usage=None) -> bytes:
        d = {"id": cid, "object": "text_completion", "created": int(time.time()), "model": model,
             "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish,
                          "stop_reason": stop_reason}]}
        if usage is not None:            # stream_options.continuous_usage_stats（与 vLLM 同口径）
            d["usage"] = usage
        return f"data: {json.dumps(d, ensure_ascii=False)}\n\n".encode("utf-8")

    def metrics_of(res: Mapping) -> dict:
        def p50(xs):
            xs = sorted(xs)
            return round(xs[len(xs) // 2], 3) if xs else None
        return {"front": res.get("front"), "back": res.get("back"),
                "front_prefill_ms": res.get("front_prefill_ms"), "bind_ms": res.get("bind_ms"),
                "ttft_ms": res.get("ttft_ms"), "tpot_ms": res.get("tpot_ms"),
                "front_leg_ms_p50": p50(res.get("front_leg_ms") or []),
                "back_leg_ms_p50": p50(res.get("back_leg_ms") or []),
                "front_engine_ms_p50": p50(res.get("front_engine_ms") or []),
                "back_engine_ms_p50": p50(res.get("back_engine_ms") or []),
                "num_generation_tokens": len(res.get("tokens") or [])}

    async def completions(request: web.Request) -> web.StreamResponse:
        t_recv = time.time()
        try:
            body = await request.json()
        except Exception as e:                                    # noqa: BLE001
            return bad(400, f"请求体不是 JSON：{e}")
        pid = str(body.get("p2p_id") or request.headers.get("X-Request-Id")
                  or f"tg-{uuid.uuid4().hex[:12]}")
        task = body.get("p2p_task")
        rec = {"id": pid, "mode": mode, "path": "completions", "true_task": task, "task": task,
               "t_recv": t_recv, "channel": None, "front": None, "front_nodes": [],
               "back_nodes": [], "verdict": None, "shadow": None, "route_ms": 0.0,
               "status": None}
        records.append(rec)
        ids = body.get("prompt")
        if not (isinstance(ids, list) and ids and all(isinstance(x, int) for x in ids)):
            return bad(400, "分段网关只收 token id 的 prompt（bench 就是这么发的）", rec)
        if float(body.get("temperature") or 0.0) != 0.0:
            return bad(400, "分段网关只支持贪心（temperature 0）", rec)
        body_ranges = body.get("p2p_body")
        if mode in ("oracle", "shadow"):
            if not task:
                return bad(400, f"{mode} 模式要在请求里给 p2p_task", rec)
            if not any(b.task == task for b in co.backs):
                # 与 A 的网关一致：task 没有后段 → 立刻 503，不偷偷塞给别的 task 的后段
                have = sorted({b.task for b in co.backs if b.task})
                return bad(503, f"task {task!r} 没有后段（有的是 {have}）", rec)
        if mode in ("lr", "shadow") and needs_body and not body_ranges:
            msg = ("识别器只数 body token —— 请求里要带 p2p_body（bench 用 p2p-framework 的 "
                   "encode_body 构造，与选择器训练时同一口径）")
            if mode == "lr":
                return bad(400, msg, rec)
            rec["shadow"] = {"error": msg}
        rec["task"] = task if mode != "lr" else None
        max_tokens = int(body.get("max_tokens") or 16)
        max_tokens = max(1, min(max_tokens, max_model_len - len(ids)))
        stream = bool(body.get("stream"))
        so = body.get("stream_options") or {}
        cont_usage = bool(so.get("include_usage") and so.get("continuous_usage_stats"))

        def running_usage():
            return {"prompt_tokens": len(ids), "completion_tokens": n_out,
                    "total_tokens": len(ids) + n_out}
        cid = f"cmpl-{pid}"
        det = IncrementalDetok(tokenizer, skip_special=body.get("skip_special_tokens", True))
        resp: web.StreamResponse | None = None
        hdr: dict = {"X-P2P-Id": pid, "X-P2P-Task": task or "", "X-P2P-Route-Ms": "0"}
        req = None
        n_out = 0

        def bound() -> None:
            """第一个 token 出来时绑定早已发生：把前段 / 后段记进路由记录与响应头。"""
            back = req.back.name if req.back else None
            front = req.front.name if req.front else None
            rec.update(channel=back, front=front, t_first_byte=time.time(), task=req.task,
                       front_nodes=list((stages.get(front) or {}).get("nodes", [])),
                       back_nodes=list((stages.get(back) or {}).get("nodes", [])))
            hdr.update({"X-P2P-Channel": back or "", "X-P2P-Front": front or "",
                        "X-P2P-Task": req.task or ""})
            v = req.info.get("verdict")
            if v is not None:
                # bench 的 identify 列按 A4 识别服务的字段名读
                v = dict(v, compute_ms=v.get("lr_ms"), queue_ms=0.0)
                if mode == "lr":
                    rec["verdict"] = v
                    rec["route_ms"] = v.get("lr_ms") or 0.0
                    hdr["X-P2P-LR"] = json.dumps({k: v.get(k) for k in
                                                  ("task", "label", "conf", "zone")},
                                                 ensure_ascii=True, separators=(",", ":"))
                elif rec.get("shadow") is None:
                    rec["shadow"] = v

        try:
            async for ev in co.stream(ids, max_tokens, task=task, rid=pid, timeout=stream_timeout,
                                      body=body_ranges):
                kind = ev[0]
                if kind == "req":
                    req = ev[1]
                    continue
                if kind == "tok":
                    first = n_out == 0
                    n_out += len(ev[1])
                    delta = det.push(ev[1])
                    if first:
                        bound()
                        if stream:
                            resp = web.StreamResponse(status=200, headers={
                                **hdr, "Content-Type": "text/event-stream"})
                            await resp.prepare(request)
                    if stream and delta:
                        await resp.write(chunk(cid, rec["channel"], delta,
                                               usage=running_usage() if cont_usage else None))
                    continue
                # done
                res = ev[1]
                if n_out == 0:
                    bound()
                tail = det.flush()
                fin = _finish_reason(res.get("finish"))
                sr = res.get("stop_reason")
                usage = {"prompt_tokens": len(ids), "completion_tokens": n_out,
                         "total_tokens": len(ids) + n_out}
                met = metrics_of(res)
                rec.update(status=200, t_done=time.time(), finish=fin,
                           front_prefill_ms=met["front_prefill_ms"], bind_ms=met["bind_ms"])
                if not stream:
                    return web.json_response({
                        "id": cid, "object": "text_completion", "created": int(t_recv),
                        "model": rec["channel"],
                        "choices": [{"index": 0, "text": det.text, "logprobs": None,
                                     "finish_reason": fin, "stop_reason": sr}],
                        "usage": usage, "p2p_metrics": met},
                        headers=hdr, dumps=lambda o: json.dumps(o, ensure_ascii=False))
                if resp is None:
                    resp = web.StreamResponse(status=200, headers={
                        **hdr, "Content-Type": "text/event-stream"})
                    await resp.prepare(request)
                await resp.write(chunk(cid, rec["channel"], tail, fin, sr,
                                       usage=usage if cont_usage else None))
                u = {"id": cid, "object": "text_completion", "created": int(time.time()),
                     "model": rec["channel"], "choices": [], "usage": usage, "p2p_metrics": met}
                await resp.write(f"data: {json.dumps(u, ensure_ascii=False)}\n\n".encode())
                await resp.write(b"data: [DONE]\n\n")
                await resp.write_eof()
                return resp
            return bad(500, "流提前结束", rec)
        except (ConnectionResetError, asyncio.CancelledError):
            rec.update(status=499, error="客户端断开", t_done=time.time())
            raise
        except Exception as e:                                    # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            if resp is not None and resp.prepared:
                err = {"error": {"message": msg, "code": 500}}
                await resp.write(f"data: {json.dumps(err, ensure_ascii=False)}\n\n".encode())
                await resp.write_eof()
                rec.update(status=500, error=msg, t_done=time.time())
                return resp
            return bad(502, msg, rec)

    async def get_records(request: web.Request):
        since = int(request.query.get("since", 0))
        return web.json_response({"mode": mode, "n": len(records), "records": records[since:]},
                                 dumps=lambda o: json.dumps(o, ensure_ascii=False))

    async def get_channels(_):
        st = co.state()
        return web.json_response({
            "mode": mode, "kind": "tiered", "identifier": identifier,
            "tasks": sorted({b["task"] for b in st["backs"] if b["task"]}),
            # bench 把这些当「通道」：名字要能在 stages.json 里查到 task / 节点
            "channels": [{"name": b["name"], "task": b["task"], "inflight": b["inflight"],
                          "served": b["bound"]} for b in st["backs"]],
            "fronts": st["fronts"]})

    async def health(_):
        return web.json_response({"ok": True, "mode": mode, "kind": "tiered"})

    async def reset(_):
        records.clear()
        return web.json_response({"ok": True})

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.add_routes([
        web.post("/v1/completions", completions),
        web.get("/p2p/records", get_records),
        web.post("/p2p/reset", reset),
        web.get("/p2p/channels", get_channels),
        web.get("/health", health),
    ])
    return app


async def _amain(a) -> int:
    from aiohttp import web
    from tokenizers import Tokenizer

    from vllm.p2p.remote import RemoteClient
    from vllm.p2p.tiered import Engine, TieredCoordinator

    sj = json.loads(Path(a.stages).read_text(encoding="utf-8"))
    entries = {c["name"]: c for c in sj["channels"]}
    fronts = [entries[n] for n in sj["fronts"]]
    backs = [entries[n] for n in sj["backs"]]
    if a.only:
        want = {x.strip() for x in a.only.split(",") if x.strip()}
        backs = [b for b in backs if b["name"] in want or b["name"].removeprefix("back-") in want]
        if not backs:
            raise SystemExit(f"--only {a.only} 一个后段都没对上（有 {sj['backs']}）")
    if a.fronts:
        want = {x.strip() for x in a.fronts.split(",") if x.strip()}
        fronts = [f for f in fronts if f["name"] in want or f["nodes"][0] in want]
        if not fronts:
            raise SystemExit(f"--fronts {a.fronts} 一个前段都没对上（有 {sj['fronts']}）")

    engines = []
    infos: dict[str, dict] = {}
    for c in (*fronts, *backs):
        cl = RemoteClient(c["zmq"])
        try:
            info = await cl.connect(timeout=a.connect_timeout, direct=not a.via_coordinator)
        except Exception as e:                                    # noqa: BLE001
            raise SystemExit(f"连不上 {c['name']}（{c['zmq']}）：{e}\n先 python3 vllm/p2p/launch.py "
                             f"--channels {a.stages} status {c['name']}") from None
        if info.get("role") != c["role"]:
            raise SystemExit(f"{c['name']} 报的角色是 {info.get('role')}，清单里是 {c['role']}")
        print(f"  {c['name']:<20} {c['role']:<5} task={c.get('task') or '—':<10} "
              f"层 {info.get('layers')}  PP={info.get('pp')}  max_num_seqs {info.get('max_num_seqs')}"
              f"  {c['zmq']}", flush=True)
        infos[c["name"]] = info
        engines.append(Engine(c["name"], c["role"], cl, task=c.get("task")))
    fe = [e for e in engines if e.role == "front"]
    be = [e for e in engines if e.role == "back"]
    eos, stop_ids = load_stop_ids(a.model_dir)
    tok = Tokenizer.from_file(str(Path(a.model_dir) / "tokenizer.json"))
    classify, clf, ident_desc = None, None, None
    front_lr, needs_body = False, False
    if a.mode in ("lr", "shadow") and a.lr_at == "front":
        # LR 在各前段代理本机跑：核对每个前段都配了、配的 task 都有后段
        have = sorted({b.task for b in be if b.task})
        no_lr = [e.name for e in fe if not infos[e.name].get("lr")]
        if no_lr:
            raise SystemExit(f"--lr-at front，但前段 {no_lr} 没有本机 LR（hf_overrides 里没有 p2p_lr；"
                             f"用 colorplan.py --lr-model … 重新生成 stages.json）")
        for e in fe:
            lr = infos[e.name]["lr"]
            extra = sorted(set(lr.get("tasks") or {}) - set(have))
            if extra:
                raise SystemExit(f"{e.name} 的 LR 会判出 {extra}，但这些 task 没有后段（有的是 {have}）")
        needs_body = any(bool(infos[e.name]["lr"].get("needs_body")) for e in fe)
        front_lr = True
        models = sorted({infos[e.name]["lr"].get("model") for e in fe})
        ident_desc = f"前段本机 LR（{', '.join(str(m) for m in models)}）"
        print(f"前段本机识别：{ {e.name: infos[e.name]['lr'].get('classifier') for e in fe} }", flush=True)
    elif a.mode in ("lr", "shadow"):
        from vllm.p2p.identifier import parse_tasks, use_p2p_framework

        use_p2p_framework(a.p2p_framework)
        have = sorted({b.task for b in be if b.task})
        tasks = parse_tasks(a.tasks) if a.tasks else {u: 1.0 for u in have}
        if set(tasks) - set(have):
            raise SystemExit(f"--tasks 里 {sorted(set(tasks) - set(have))} 没有后段（有的是 {have}）")
        classify, clf = make_lr_classify(
            lr_model=a.lr_model, l0=int(sj["l0"]), n_experts=_n_experts(sj, a.model_dir),
            tasks=tasks, lr_classes=a.lr_classes, lr_features=a.lr_features,
            lr_feature_fn=a.lr_feature_fn, tau_hi=a.tau_hi, tau_lo=a.tau_lo)
        ident_desc = f"前段出口 LR（{Path(a.lr_model).name}）"
        print(f"前段出口识别：{clf.describe()}；部署的 task 与先验 {tasks}", flush=True)
    co = TieredCoordinator(fe, be, bind="least", eos=eos, stop_ids=stop_ids, ignore_eos=False,
                           front_pick=a.front_pick, direct=not a.via_coordinator,
                           classify=classify, route="lr" if a.mode == "lr" else "task",
                           front_lr=front_lr)
    co.start()
    app = make_app(co, stages=entries, tokenizer=tok, mode=a.mode,
                   max_model_len=a.max_model_len,
                   needs_body=bool(clf is not None and clf.needs_body) or needs_body,
                   identifier=ident_desc,
                   stream_timeout=a.stream_timeout)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, a.host, a.port).start()
    print(f"分段网关（{a.mode}，前段按 {a.front_pick} 挑，"
          f"{'经协调器转' if a.via_coordinator else '前后段直连'}）：前段 {[e.name for e in fe]}，后段 "
          f"{ {u: [b.name for b in be if b.task == u] for u in sorted({b.task for b in be})} }；"
          f"停止 token {stop_ids}；监听 {a.host}:{a.port}", flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        await co.close()
        await runner.cleanup()
        for e in engines:
            e.client.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stages", default="/home/ubuntu/vllm-p2p/p2p_channels/stages.json")
    ap.add_argument("--model-dir", required=True, help="有 tokenizer.json / generation_config.json 的目录")
    ap.add_argument("--only", default=None, help="只用这些后段（逗号分隔，back-gsm8k-0 或 gsm8k-0）")
    ap.add_argument("--fronts", default=None, help="只用这些前段（front-N01 或 N01）")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9100, help="默认 9100（A 的网关占着 9000）")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--connect-timeout", type=float, default=30.0)
    ap.add_argument("--stream-timeout", type=float, default=3600.0,
                    help="一条请求两次产出之间最多等多久（秒）。离线满载时请求会在后段引擎里排很久的队，"
                         "首 token 之前可能几分钟没有动静")
    ap.add_argument("--via-coordinator", action="store_true",
                    help="每 token 的 hidden / token 都经协调器转（默认前后段代理直连；"
                         "协调器与引擎不在同一网段时每跳多一个往返）")
    ap.add_argument("--front-pick", choices=("speed", "least"), default="speed",
                    help="选前段：speed =（在飞+1）×实测每位置耗时最小（默认）；least = 在飞最少")
    ap.add_argument("--mode", choices=MODES, default="oracle",
                    help="oracle：按请求的 p2p_task 绑；lr：按前段出口识别的结果绑（B-1 目标形态）；"
                         "shadow：按 p2p_task 绑，只记识别结果")
    g = ap.add_argument_group("前段出口识别（--mode lr / shadow）")
    g.add_argument("--lr-at", choices=("gateway", "front"), default="gateway",
                   help="LR 在哪跑：gateway = 网关里（默认，前段交统计）；front = 各前段代理本机"
                        "（含第 L₀ 层的卡，hf_overrides 里要有 p2p_lr），网关只用判定结果、不加载选择器")
    g.add_argument("--p2p-framework", default=None, help="p2p-framework 源码目录（默认自动找）")
    g.add_argument("--lr-model",
                   default="/home/ubuntu/p2p-framework/selector/selector_body_L8_params.npz")
    g.add_argument("--tasks", default=None,
                   help="部署的 task 与先验，如 gsm8k=1,mbpp=1（默认 = 后段的 task，先验各 1）")
    g.add_argument("--lr-classes", default=None)
    g.add_argument("--lr-features", default=None)
    g.add_argument("--lr-feature-fn", default=None)
    g.add_argument("--tau-hi", type=float, default=None)
    g.add_argument("--tau-lo", type=float, default=0.40)
    a = ap.parse_args(argv)
    try:
        return asyncio.run(_amain(a))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
