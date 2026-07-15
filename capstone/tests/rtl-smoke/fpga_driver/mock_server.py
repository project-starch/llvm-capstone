#!/usr/bin/env python3
"""Mock Socket.IO FPGA console -- implements the PLACEHOLDER protocol in config.py.

This is NOT the real board. It exists so the driver's flow (connect -> upload ->
load -> reset -> run commands over UART -> harvest RESULT lines -> trace dump) can
be exercised end-to-end offline. It mirrors config.EMIT / config.LISTEN exactly,
so when the real protocol replaces the placeholders you update BOTH sides (or
throw this mock away and point the driver at the board).

It emits UART in small chunks -- including splitting marker lines across chunks --
to prove the driver's cross-chunk marker matching. The RESULT lines carry the
reference numbers (bump 7 / norevoke 60 / revoke 65) so run_rtl_smoke's
--parse-uart step produces the expected breakdown.

Run:  python mock_server.py [--host 127.0.0.1] [--port 8137]
Needs: python-socketio[asgi... no], aiohttp  (server extra).
"""

from __future__ import annotations

import argparse
import asyncio

import socketio
from aiohttp import web

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fpga_driver import config as C  # type: ignore
else:  # pragma: no cover
    from . import config as C

NS = C.CONNECT.namespace

# Reference RESULT lines keyed by the .dom the controller command names.
RESULT_LINES = {
    "revoke_cost_fpga_bump.dom":
        "revoke-cost-fpga: RESULT cycles/op  mode=bump  alloc_free=7",
    "revoke_cost_fpga_norevoke.dom":
        "revoke-cost-fpga: RESULT cycles/op  mode=norevoke  alloc_free=60",
    "revoke_cost_fpga_revoke.dom":
        "revoke-cost-fpga: RESULT cycles/op  mode=revoke  alloc_free=65",
}
REVOKE_DONE = "revoke-cost-fpga: measurement complete"
BORROW_LINES = [
    "borrow-cost-fpga: RAW iters=1024 empty=1 raw=2 borrow=6 copy256=34 copy1024=130",
    "borrow-cost-fpga: RESULT cycles/op  raw=2  borrow=6  copy@256B=34  copy@1024B=130",
    "borrow-cost-fpga: RESULT vs-raw     borrow=3.0x  copy@256B=17.0x  copy@1024B=65.0x",
]


def build_app() -> tuple[socketio.AsyncServer, web.Application]:
    sio = socketio.AsyncServer(async_mode="aiohttp", cors_allowed_origins="*")
    app = web.Application()
    sio.attach(app, socketio_path=C.CONNECT.socketio_path)

    state = {"switch": {}, "capturing": False}

    async def emit_uart(sid: str, text: str, chunk: int = 24) -> None:
        """Send `text` split into small chunks (some marker lines land across a
        chunk boundary on purpose)."""
        for i in range(0, len(text), chunk):
            await sio.emit(
                C.LISTEN["uart_output"], {C.UART_TEXT_KEYS[0]: text[i:i + chunk]},
                namespace=NS, to=sid,
            )
            await asyncio.sleep(0.005)

    async def set_status(sid: str, state_str: str) -> None:
        await sio.emit(C.LISTEN["status"], {C.STATUS_STATE_KEY: state_str},
                       namespace=NS, to=sid)

    @sio.event(namespace=NS)
    async def connect(sid, environ, auth=None):  # noqa: ARG001
        await set_status(sid, "Idle")

    @sio.on(C.EMIT["power"].event, namespace=NS)
    async def _power(sid, data):  # noqa: ARG001
        await set_status(sid, "Idle")

    @sio.on(C.EMIT["boot_image_upload"].event, namespace=NS)
    async def _upload(sid, data):
        # config marks this expects_ack -> return an ack payload.
        name = (data or {}).get("name", "image.bin")
        return {"ok": True, "name": name, "stored": True}

    @sio.on(C.EMIT["boot_image_load"].event, namespace=NS)
    async def _load(sid, data):  # noqa: ARG001
        await set_status(sid, "Loading")
        await asyncio.sleep(0.02)
        await set_status(sid, "Done")

    @sio.on(C.EMIT["reset"].event, namespace=NS)
    async def _reset(sid, data=None):  # noqa: ARG001
        await set_status(sid, "Idle")
        await asyncio.sleep(0.01)
        await emit_uart(
            sid,
            "\n[    0.000000] Linux version 6.x (capstone)\n"
            "[    1.234567] Freeing unused kernel memory\n"
            "Welcome to Buildroot\n"
            "buildroot login: root\n"
            "# ",
        )

    @sio.on(C.EMIT["switch_set"].event, namespace=NS)
    async def _switch(sid, data):
        idx, on = data["index"], data["on"]
        state["switch"][idx] = on
        await sio.emit(C.LISTEN["switches"], {"switches": state["switch"]},
                       namespace=NS, to=sid)
        # switch 1 while capturing triggers the tracer to dump -> end-of-dump.
        if idx == 1 and on and state["capturing"]:
            await sio.emit(C.LISTEN["trace_frame"],
                           {"seq": 0, "event": "debug_print", "data": "0xdeadbeef"},
                           namespace=NS, to=sid)
            await sio.emit(C.LISTEN["trace_complete"],
                           {"frames": 1, "truncated": False}, namespace=NS, to=sid)
            state["capturing"] = False

    @sio.on(C.EMIT["trace_dump"].event, namespace=NS)
    async def _trace(sid, data=None):  # noqa: ARG001
        state["capturing"] = True
        await set_status(sid, "Capturing")

    @sio.on(C.EMIT["terminal_input"].event, namespace=NS)
    async def _terminal(sid, data):
        line = (data or {}).get("data", "")
        cmd = line.strip()
        # Echo the command like a real TTY, then respond based on which .dom ran.
        await emit_uart(sid, cmd + "\n")
        payload = ""
        for dom, result in RESULT_LINES.items():
            if dom in cmd:
                payload = result + "\n" + REVOKE_DONE + "\n# "
                break
        else:
            if "borrow_cost_fpga.dom" in cmd:
                payload = "\n".join(BORROW_LINES) + "\n# "
            elif cmd:
                payload = "# "  # unknown command: just a fresh prompt
        if payload:
            await emit_uart(sid, payload)

    return sio, app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8137)
    args = ap.parse_args()
    _sio, app = build_app()
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
