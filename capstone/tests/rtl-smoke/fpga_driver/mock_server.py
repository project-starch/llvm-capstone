#!/usr/bin/env python3
"""Mock FPGA console -- implements the VERIFIED hybrid protocol in config.py.

This is NOT the real board. It exists so the driver's flow (connect -> power ->
upload -> load -> reset -> run commands over UART -> harvest RESULT lines -> trace
dump, plus the Lock) can be exercised end-to-end offline. It mirrors the real
console's shape: HTTP REST endpoints for the action verbs (``/api/...``) and
Socket.IO events for the live stream + state. When the real protocol changes you
update BOTH sides (or throw this mock away and point the driver at the board).

It emits UART in small chunks -- including splitting marker lines across chunk
boundaries on purpose -- to prove the driver's cross-chunk marker matching. The
RESULT lines carry the reference numbers (bump 7 / norevoke 60 / revoke 65) so
run_rtl_smoke's --parse-uart step produces the expected breakdown.

Run:  python mock_server.py [--host 127.0.0.1] [--port 8137]
Needs: python-socketio + aiohttp (server side).
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

    board = {
        "power": "off",
        "switches": [0] * 8,
        "locked": False,
        "timeout": 600,
        "capturing": False,
        "line": "",       # accumulated UART keystrokes until CR/LF
        "clients": 0,
        "uart_seq": 0,          # monotonic chunk counter (_uart_history_seq)
        "uart_history": [],     # [(seq, text)] ring buffer for request_history
        "gdb": "idle",          # gdb_state: idle|starting|running|error
        "gdb_line": "",         # accumulated gdb PTY keystrokes until newline
    }

    BOOT_LOG = (
        "\n[    0.000000] Linux version 6.x (capstone)\n"
        "[    1.234567] Freeing unused kernel memory\n"
        "Welcome to Buildroot\n"
        "buildroot login: root\n"
        "# "
    )

    async def emit_uart(text: str, chunk: int = 24) -> None:
        """Broadcast `text` split into small chunks (markers may land across a
        chunk boundary on purpose). Each chunk gets a monotonic `seq` and is kept
        in the history buffer, mirroring the real server's uart_data {seq, text}."""
        for i in range(0, len(text), chunk):
            piece = text[i:i + chunk]
            board["uart_seq"] += 1
            seq = board["uart_seq"]
            board["uart_history"].append((seq, piece))
            await sio.emit(C.LISTEN["uart_output"],
                           {C.UART_SEQ_KEY: seq, C.UART_TEXT_KEYS[0]: piece},
                           namespace=NS)
            await asyncio.sleep(0.005)

    async def push_states() -> None:
        await sio.emit(C.POWER_STATE_EVENT, {"state": board["power"]}, namespace=NS)
        await sio.emit(C.SWITCH_STATE_EVENT, {C.SWITCH_STATE_KEY: board["switches"]},
                       namespace=NS)
        await sio.emit(C.LOCK_STATE_EVENT,
                       {C.LOCK_TIMEOUT_KEY: board["timeout"],
                        C.LOCK_LOCKED_KEY: board["locked"]}, namespace=NS)

    # ── Socket.IO events ──────────────────────────────────────────────────
    @sio.event(namespace=NS)
    async def connect(sid, environ, auth=None):  # noqa: ARG001
        board["clients"] += 1
        await sio.emit(C.USER_COUNT_EVENT, {C.USER_COUNT_KEY: board["clients"]},
                       namespace=NS)
        await sio.emit(C.GDB_STATE_EVENT, {"state": board["gdb"]}, namespace=NS)
        await push_states()

    @sio.event(namespace=NS)
    async def disconnect(sid):  # noqa: ARG001
        board["clients"] = max(0, board["clients"] - 1)

    @sio.on(C.EMIT["request_history"].event, namespace=NS)
    async def _history(sid, data=None):
        """Replay UART chunks newer than last_seq, to the requester only, as a
        single uart_data {seq, text} -- exactly like the real server. seq gating
        is what makes threading last_seq on the client observable: pass the real
        seq and nothing (already-seen) is re-sent; pass -1 and the whole buffer
        replays."""
        last = -1
        if isinstance(data, dict):
            try:
                last = int(data.get("last_seq", -1))
            except (TypeError, ValueError):
                last = -1
        chunks = [(s, t) for (s, t) in board["uart_history"] if s > last]
        if chunks:
            text = "".join(t for _, t in chunks)
            await sio.emit(C.LISTEN["uart_output"],
                           {C.UART_SEQ_KEY: chunks[-1][0], C.UART_TEXT_KEYS[0]: text},
                           to=sid, namespace=NS)

    @sio.on(C.EMIT["power_toggle"].event, namespace=NS)
    async def _power(sid, data=None):  # noqa: ARG001
        board["power"] = "on" if board["power"] == "off" else "off"
        await sio.emit(C.POWER_STATE_EVENT, {"state": board["power"]}, namespace=NS)

    @sio.on(C.EMIT["switch_toggle"].event, namespace=NS)
    async def _switch(sid, data):  # noqa: ARG001
        idx = int(data["index"])
        if 0 <= idx < len(board["switches"]):
            board["switches"][idx] ^= 1
        await sio.emit(C.SWITCH_STATE_EVENT, {C.SWITCH_STATE_KEY: board["switches"]},
                       namespace=NS)

    @sio.on(C.EMIT["switch_reset_all"].event, namespace=NS)
    async def _switch_reset(sid, data=None):  # noqa: ARG001
        board["switches"] = [0] * len(board["switches"])
        await sio.emit(C.SWITCH_STATE_EVENT, {C.SWITCH_STATE_KEY: board["switches"]},
                       namespace=NS)

    @sio.on(C.EMIT["set_auto_shutdown"].event, namespace=NS)
    async def _lock(sid, data):  # noqa: ARG001
        board["timeout"] = int(data.get("timeout_seconds", board["timeout"]))
        board["locked"] = bool(data.get("locked", False))
        await sio.emit(C.LOCK_STATE_EVENT,
                       {C.LOCK_TIMEOUT_KEY: board["timeout"],
                        C.LOCK_LOCKED_KEY: board["locked"]}, namespace=NS)

    @sio.on(C.EMIT["uart_clear"].event, namespace=NS)
    async def _uart_clear(sid, data=None):  # noqa: ARG001
        board["line"] = ""
        board["uart_history"].clear()  # doc: clears the server-side history buffer

    @sio.on(C.EMIT["uart_send"].event, namespace=NS)
    async def _uart_send(sid, data):  # noqa: ARG001
        text = (data or {}).get("text", "")
        for ch in text:
            if ch in ("\r", "\n"):
                cmd = board["line"].strip()
                board["line"] = ""
                await _run_line(cmd)
            else:
                board["line"] += ch

    async def _run_line(cmd: str) -> None:
        # Echo the command like a real TTY, then respond based on which .dom ran.
        await emit_uart(cmd + "\n")
        payload = ""
        if cmd.startswith("echo "):
            # Shell echo (e.g. the RDY''OK login-confirm probe); quotes collapse.
            await emit_uart(cmd[len("echo "):].replace("'", "") + "\n# ")
            return
        if "insmod" in cmd and "capstone" in cmd:
            # The .doms need /dev/capstone; a healthy image loads the module.
            await emit_uart("/dev/capstone\nCAPSTONE_MOD_OK\n# ")
            return
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
            await emit_uart(payload)

    # ── GDB session (self-serve boot path) ────────────────────────────────
    async def _gdb_emit(text: str) -> None:
        await sio.emit(C.GDB_OUTPUT_EVENT, {C.GDB_OUTPUT_KEY: text}, namespace=NS)

    @sio.on(C.EMIT["gdb_start"].event, namespace=NS)
    async def _gdb_start(sid, data=None):  # noqa: ARG001
        if board["gdb"] not in ("idle", "error"):
            return
        board["gdb"] = "starting"
        await sio.emit(C.GDB_STATE_EVENT, {"state": "starting"}, namespace=NS)
        await asyncio.sleep(0.01)
        board["gdb"] = "running"
        await sio.emit(C.GDB_STATE_EVENT, {"state": "running"}, namespace=NS)
        await _gdb_emit("GNU gdb (multiarch)\n(gdb) ")

    @sio.on(C.EMIT["gdb_stop"].event, namespace=NS)
    async def _gdb_stop(sid, data=None):  # noqa: ARG001
        board["gdb"] = "idle"
        await sio.emit(C.GDB_STATE_EVENT, {"state": "idle"}, namespace=NS)

    @sio.on(C.EMIT["gdb_input"].event, namespace=NS)
    async def _gdb_input(sid, data):  # noqa: ARG001
        if board["gdb"] != "running":
            return
        text = (data or {}).get("text", "")
        for ch in text:
            if ch in ("\r", "\n"):
                cmd = board["gdb_line"].strip()
                board["gdb_line"] = ""
                await _gdb_emit(cmd + "\n")
                if cmd == "continue" or cmd.startswith("c "):
                    # Resuming our loaded image -> OpenSBI -> Linux -> shell on UART.
                    await emit_uart(BOOT_LOG)
                else:
                    await _gdb_emit("(gdb) ")  # command ack + fresh prompt
            else:
                board["gdb_line"] += ch

    # ── REST action verbs ─────────────────────────────────────────────────
    api = C.CONNECT.api_prefix

    async def _upload(request: web.Request) -> web.Response:
        reader = await request.multipart()
        name = None
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == "name":
                name = (await part.text()).strip()
            elif part.name == "file":
                await part.read()  # drain; the mock doesn't store bytes
                name = name or part.filename
        return web.json_response({"name": name or "image.bin"})

    async def _load_image(request: web.Request) -> web.Response:
        body = await request.json()
        name = body.get("filename")
        await sio.emit(C.LISTEN["load_state"], {"state": "loading"}, namespace=NS)
        await asyncio.sleep(0.02)
        await sio.emit(C.LISTEN["load_state"],
                       {"state": "done", "loaded_image_name": name}, namespace=NS)
        return web.json_response({"state": "done"})

    async def _reset_board(request: web.Request) -> web.Response:  # noqa: ARG001
        async def _boot() -> None:
            await asyncio.sleep(0.01)
            await emit_uart(BOOT_LOG)
        asyncio.create_task(_boot())
        return web.json_response({"ok": True})

    async def _trace_start(request: web.Request) -> web.Response:  # noqa: ARG001
        board["capturing"] = True
        await sio.emit(C.LISTEN["trace_state"], {"state": "capturing"}, namespace=NS)

        async def _dump() -> None:
            await asyncio.sleep(0.02)
            await sio.emit(C.LISTEN["trace_result"],
                           {"text": "seq=0 event=debug_print data=0xdeadbeef\n"
                                    "-- end of dump (1 frame) --"}, namespace=NS)
            await sio.emit(C.LISTEN["trace_state"], {"state": "idle"}, namespace=NS)
            board["capturing"] = False
        asyncio.create_task(_dump())
        return web.json_response({"state": "capturing"})

    async def _trace_cancel(request: web.Request) -> web.Response:  # noqa: ARG001
        board["capturing"] = False
        await sio.emit(C.LISTEN["trace_state"], {"state": "idle"}, namespace=NS)
        return web.json_response({"state": "idle"})

    app.router.add_post(f"/{api}/images/upload", _upload)
    app.router.add_post(f"/{api}/load-image", _load_image)
    app.router.add_post(f"/{api}/reset-board", _reset_board)
    app.router.add_post(f"/{api}/trace-start", _trace_start)
    app.router.add_post(f"/{api}/trace-cancel", _trace_cancel)

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
