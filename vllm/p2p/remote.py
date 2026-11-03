# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B1：协调器与**别的机器上**的分段引擎之间的线路（ZMQ DEALER ↔ ROUTER，vLLM 自己的 msgpack）。

每个分段引擎（前段 / 后段）的首节点上跑一个代理 `stage_agent.py`：它在本机起 vLLM 的
EngineCore（`AsyncMPClient`，PP 多机时 worker 在其余节点上以 headless 起），再把 EngineCore 的
四个入口原样搬到网络上。协调器这边的 `RemoteClient` 提供与 `AsyncMPClient` 相同的那几个方法，
所以 `tiered.TieredCoordinator` 不用改就能接远端引擎。

帧格式（每条消息一个 multipart）：

    协调器 → 代理   [H, {direct}]               握手（代理回 [I, info json]）
                    [A, *EngineCoreRequest]     add_request
                    [X, *list[str]]             abort
                    [U, *(method, args)]        utility，不等结果（失败时代理回 [E, …]）
                    [R, route json]             直连路由 {rid, peer, next}（见 AgentBridge）
    代理 → 协调器   [O, meta json, *EngineCoreOutputs]   引擎的每一批输出；meta = {req_id: 引擎内毫秒}
                    （这条请求最近一次「进代理」到这批输出「出代理」—— 排队 + 调度 + 前向，不含网络）
                    [I, info json]  [E, 错误文本]

张量走 vLLM 的零拷贝辅助帧（`MsgpackEncoder` 把大张量放进单独的帧）。
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

__all__ = ["RemoteClient", "AgentBridge", "K_HELLO", "K_ADD", "K_ABORT", "K_UTIL",
           "K_ROUTE", "K_OUT", "K_INFO", "K_ERR"]

K_HELLO, K_ADD, K_ABORT, K_UTIL, K_ROUTE = b"H", b"A", b"X", b"U", b"R"
K_OUT, K_INFO, K_ERR = b"O", b"I", b"E"


class RemoteClient:
    """协调器侧：长得像 `AsyncMPClient` 的远端分段引擎。"""

    def __init__(self, address: str, *, ctx=None):
        import zmq
        import zmq.asyncio

        from vllm.v1.engine import EngineCoreOutputs
        from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

        self.address = address
        self._own_ctx = ctx is None
        self.ctx = ctx or zmq.asyncio.Context()
        self.sock = self.ctx.socket(zmq.DEALER)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(address)
        self.enc = MsgpackEncoder()
        self.dec_out = MsgpackDecoder(EngineCoreOutputs)
        self.dec_any = MsgpackDecoder()
        self.info: dict = {}
        self._pending_info: asyncio.Future | None = None
        self.last_meta: dict = {}           # 最近一批输出里各请求的「引擎内毫秒」（代理量的）

    async def connect(self, timeout: float = 30.0, *, direct: bool = False) -> dict:
        """握手：拿到代理报的本段信息（角色、层、hidden 维度…）。`direct`：协调器会下发
        直连路由（前段代理据此把「路由未到」的 decode hidden 先攒着，而不是发给协调器）。"""
        await self.sock.send_multipart([K_HELLO, json.dumps({"direct": bool(direct)}).encode()])
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            left = deadline - asyncio.get_running_loop().time()
            if left <= 0:
                raise TimeoutError(f"{self.address} {timeout:.0f}s 内没有回握手 —— 代理没起或端口不通")
            frames = await asyncio.wait_for(self.sock.recv_multipart(copy=False), left)
            kind = bytes(frames[0])
            if kind == K_INFO:
                self.info = json.loads(bytes(frames[1]))
                return self.info
            if kind == K_ERR:
                raise RuntimeError(f"{self.address}：{bytes(frames[1]).decode('utf-8', 'replace')}")
            # 握手前到的引擎输出：上一个协调器留下的，丢掉

    async def _send(self, kind: bytes, obj: Any) -> None:
        bufs = self.enc.encode(obj)
        await self.sock.send_multipart([kind, *bufs], copy=False)

    async def add_request_async(self, req) -> None:
        await self._send(K_ADD, req)

    async def abort_requests_async(self, ids: list[str]) -> None:
        if ids:
            await self._send(K_ABORT, list(ids))

    async def p2p_fire(self, method: str, *args):
        await self._send(K_UTIL, (method, list(args)))
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(None)            # 失败由代理以 [E] 异步报回（get_output_async 里抛）
        return fut

    async def send_route(self, rid: str, peer: str, next_pos: int = 0) -> None:
        """直连：让代理把这条请求的 hidden（前段）/ token（后段）直接发给 peer 代理。"""
        await self.sock.send_multipart(
            [K_ROUTE, json.dumps({"rid": rid, "peer": peer, "next": int(next_pos)}).encode()])

    async def get_output_async(self):
        while True:
            frames = await self.sock.recv_multipart(copy=False)
            kind = bytes(frames[0])
            if kind == K_OUT:
                self.last_meta = json.loads(bytes(frames[1])) if len(frames) > 1 else {}
                return self.dec_out.decode(frames[2:])
            if kind == K_ERR:
                raise RuntimeError(f"分段引擎 {self.info.get('name') or self.address} 报错："
                                   f"{bytes(frames[1]).decode('utf-8', 'replace')}")
            if kind == K_INFO:
                self.info = json.loads(bytes(frames[1]))

    def shutdown(self) -> None:
        self.sock.close(0)
        if self._own_ctx:
            self.ctx.term()


class AgentBridge:
    """代理侧：ROUTER socket ↔ 本机 EngineCore（`AsyncMPClient`）。

    协调器：发过 [H] 的那个对端（同一时刻只有一个协调器连着一个分段引擎）。引擎的输出都发给它。

    直连（`K_ROUTE`，协调器在绑定时下发）：每 token 的两样东西不再绕协调器 ——
    * 前段 / 中段代理：这条请求 prompt 之后位置的 hidden 直接 `p2p_feed` 给路线上的下一段代理；
      这些输出不再发给协调器（它只要 prompt 的 hidden 来做绑定 / 逐段 prefill）；路由还没到时先攒着。
    * 后段代理：这条请求新出的 token 直接 `p2p_append` 给它的前段代理；输出照常也发给协调器
      （流式给客户端、判结束）。
    别的代理发来的 [U] 不会被当成协调器。"""

    def __init__(self, client, bind: str, info: dict, feat_hook=None):
        """`feat_hook(feat) -> dict`：前段本机 LR（`stage_agent` 的 `p2p_lr`）—— 把前段交出的
        前 L₀ 层识别统计（`EngineCoreOutput.p2p_feat`）换成本机判出的结果 `{"verdict": …}`
        再发给协调器。不给 = 原样转发统计，由协调器跑 LR。"""
        import zmq
        import zmq.asyncio

        from vllm.v1.engine import EngineCoreRequest
        from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

        self.client, self.info = client, info
        self.feat_hook = feat_hook
        self.role = info.get("role")
        self.ctx = zmq.asyncio.Context()
        self.sock = self.ctx.socket(zmq.ROUTER)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.ROUTER_HANDOVER, 1)      # 协调器重连（同 identity）也认
        self.sock.bind(bind)
        self.enc = MsgpackEncoder()
        self.dec_req = MsgpackDecoder(EngineCoreRequest)
        self.dec_any = MsgpackDecoder()
        self.peer: bytes | None = None
        self.n_in = self.n_out = self.n_direct = 0
        self.dead: str | None = None
        # 每条请求最近一次「有东西进来」的时刻（add / p2p_append / p2p_feed），量引擎内耗时
        self.t_in: dict[str, float] = {}
        # 直连：req_id → [对端地址, 下一个要转的位置]；前段还记 prompt 长度与没路由时攒下的行
        self.routes: dict[str, list] = {}
        self.n_prompt: dict[str, int] = {}
        self.held: dict[str, list] = {}
        self._peers: dict[str, object] = {}
        self.direct = False             # 协调器握手时说的：它会不会下发直连路由

    async def _err(self, msg: str) -> None:
        if self.peer is not None:
            await self.sock.send_multipart([self.peer, K_ERR, msg.encode("utf-8")])

    def _peer_sock(self, addr: str):
        import zmq

        s = self._peers.get(addr)
        if s is None:
            s = self._peers[addr] = self.ctx.socket(zmq.DEALER)
            s.setsockopt(zmq.LINGER, 0)
            s.connect(addr)
        return s

    async def _send_util(self, addr: str, method: str, items: list) -> None:
        self.n_direct += 1
        await self._peer_sock(addr).send_multipart(
            [K_UTIL, *self.enc.encode((method, [items]))], copy=False)

    def _drop(self, rid: str) -> None:
        self.routes.pop(rid, None)
        self.n_prompt.pop(rid, None)
        self.held.pop(rid, None)
        self.t_in.pop(rid, None)

    def _take_rows(self, rid: str, pos: int, rows):
        """前段：这批行里该直连转出去的部分 → (起点, 行) 或 None；顺带推进去重指针。"""
        route = self.routes[rid]
        lo = max(route[1], self.n_prompt.get(rid, 0))
        end = pos + rows.shape[0]
        if end <= lo:
            return None
        route[1] = end
        return lo, rows[lo - pos:]

    async def _inbound(self) -> None:
        from vllm.p2p.stage import pack_rows
        from vllm.p2p.tiered import _fire_utility

        while True:
            frames = await self.sock.recv_multipart(copy=False)
            sender = bytes(frames[0])
            kind = bytes(frames[1])
            body = frames[2:]
            self.n_in += 1
            try:
                if kind == K_HELLO:
                    self.peer = sender
                    hello = json.loads(bytes(body[0])) if body else {}
                    self.direct = bool(hello.get("direct"))
                    await self.sock.send_multipart(
                        [self.peer, K_INFO, json.dumps(self.info).encode("utf-8")])
                elif kind == K_ADD:
                    req = self.dec_req.decode(body)
                    self.t_in[req.request_id] = time.perf_counter()
                    if req.prompt_token_ids is not None:
                        self.n_prompt[req.request_id] = len(req.prompt_token_ids)
                    elif req.prompt_embeds is not None:          # 中段 / 后段
                        self.n_prompt[req.request_id] = int(req.prompt_embeds.shape[0])
                    await self.client.add_request_async(req)
                elif kind == K_ROUTE:
                    d = json.loads(bytes(body[0]))
                    rid = d["rid"]
                    self.routes[rid] = [d["peer"], int(d.get("next", 0))]
                    for pos, rows in self.held.pop(rid, []):     # 路由到之前算出来的 hidden
                        got = self._take_rows(rid, pos, rows)
                        if got is not None:
                            await self._send_util(d["peer"], "p2p_feed",
                                                  [pack_rows(rid, got[0], got[1])])
                elif kind == K_ABORT:
                    ids = list(self.dec_any.decode(body))
                    for rid in ids:
                        self._drop(rid)
                    await self.client.abort_requests_async(ids)
                elif kind == K_UTIL:
                    method, args = self.dec_any.decode(body)
                    if method in ("p2p_append", "p2p_feed") and args:
                        now = time.perf_counter()
                        for it in args[0]:
                            self.t_in[str(it[0])] = now
                    fire = getattr(self.client, "p2p_fire", None)     # 测试用的假引擎
                    fut = await (fire(method, *args) if fire is not None
                                 else _fire_utility(self.client, method, *args))

                    def done(f, method=method):
                        if not f.cancelled() and f.exception() is not None and self.peer:
                            asyncio.ensure_future(self.sock.send_multipart(
                                [self.peer, K_ERR, f"{method}: {f.exception()}".encode("utf-8")]))
                    fut.add_done_callback(done)
                else:
                    await self._err(f"不认识的消息类型 {kind!r}")
            except Exception as e:                           # noqa: BLE001
                await self._err(f"处理 {kind!r} 失败：{type(e).__name__}: {e}")

    async def _outbound(self) -> None:
        from vllm.p2p.stage import pack_rows
        from vllm.v1.engine import EngineCoreOutputs

        while True:
            try:
                outs = await self.client.get_output_async()
            except Exception as e:                           # noqa: BLE001 — 引擎死了
                self.dead = f"{type(e).__name__}: {e}"
                await self._err(f"EngineCore 退出：{self.dead}")
                raise
            if not outs.outputs:
                continue
            now = time.perf_counter()
            meta = {}
            keep = []
            direct: dict[str, list] = {}
            for o in outs.outputs:
                rid = o.request_id
                if self.feat_hook is not None and getattr(o, "p2p_feat", None) is not None:
                    o.p2p_feat = self.feat_hook(o.p2p_feat)      # 前段本机 LR：只交判定结果
                t = self.t_in.pop(rid, None)
                if t is not None:
                    meta[rid] = round((now - t) * 1e3, 3)
                if self.role in ("front", "mid") and o.p2p_hidden is not None:
                    n0 = self.n_prompt.get(rid, 0)
                    end = o.p2p_pos + o.p2p_hidden.shape[0]
                    if rid in self.routes:
                        got = self._take_rows(rid, o.p2p_pos, o.p2p_hidden)
                        if got is not None:
                            direct.setdefault(self.routes[rid][0], []).append(
                                pack_rows(rid, got[0], got[1]))
                        if o.p2p_pos < n0:
                            keep.append(o)       # 抢占后重算回到 prompt 段：协调器会去重
                        continue
                    if self.direct and end > n0 and n0 > 0:
                        # prompt 之后的位置但路由还没到：攒着，路由一到就转
                        self.held.setdefault(rid, []).append((o.p2p_pos, o.p2p_hidden))
                        if o.p2p_pos < n0:
                            keep.append(o)
                        continue
                    keep.append(o)
                    continue
                if (self.role == "back" and o.new_token_ids and o.finish_reason is None
                        and rid in self.routes):
                    direct.setdefault(self.routes[rid][0], []).append(
                        [rid, [int(x) for x in o.new_token_ids]])
                if o.finish_reason is not None:
                    self._drop(rid)
                keep.append(o)
            for addr, items in direct.items():
                await self._send_util(addr, "p2p_append" if self.role == "back" else "p2p_feed",
                                      items)
            if self.peer is None or (not keep and not meta):
                continue
            self.n_out += 1
            # 直连转走的输出不发给协调器，但它们的「引擎内毫秒」照发（协调器按它挑前段）
            if len(keep) != len(outs.outputs):
                outs = EngineCoreOutputs(engine_index=outs.engine_index, outputs=keep,
                                         timestamp=outs.timestamp)
            await self.sock.send_multipart(
                [self.peer, K_OUT, json.dumps(meta).encode(), *self.enc.encode(outs)], copy=False)

    async def run(self) -> None:
        await asyncio.gather(self._inbound(), self._outbound())
