"""Headless Socket.IO driver for the FPGA web console.

`FpgaConsole` wraps a `python-socketio` client and exposes the five board actions
the rtl-smoke run needs, each mapped to a wire event through config.py:

    1. upload_boot_image() / load_boot_image()  -> .bin to 0x80000000, await done
    2. reset() + wait_uart(login_prompt)         -> reset, wait for the shell
    3. wait_uart(marker)                          -> read UART until a marker
    4. set_switch(n, on)                          -> toggle a virtual switch
    5. trace_dump()                               -> capture, await end-of-dump

The protocol itself (event names, payloads, completion signals) is entirely in
config.py; this module contains only transport + synchronisation logic. Nothing
here is board-specific beyond the LISTEN/EMIT tables it reads.

Only dependency: python-socketio (with the client extra). Sync client: the
Socket.IO event loop runs in a background thread and invokes our handlers; we
synchronise with the caller through a Condition-guarded event log + a UART text
buffer, so the caller writes plain procedural code.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Callable, List, Optional, Pattern, Tuple, Union

try:
    import socketio
except ModuleNotFoundError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "python-socketio is required: pip install 'python-socketio[client]'"
    ) from exc

from . import config as C


class ProtocolNotVerified(RuntimeError):
    """Raised when asked to drive a real board with a placeholder protocol."""


class ActionTimeout(RuntimeError):
    pass


class ActionError(RuntimeError):
    """The board reported an error status while an action was in flight."""


def _compile(rx: Union[str, Pattern[str]]) -> Pattern[str]:
    return rx if hasattr(rx, "search") else re.compile(rx)


class FpgaConsole:
    def __init__(
        self,
        url: str,
        *,
        token: Optional[str] = None,
        allow_unverified: bool = False,
        logger: Optional[Callable[[str], None]] = None,
        client: Optional["socketio.Client"] = None,
    ) -> None:
        """`url` is the full token'd console URL (kept out of the repo; pass on
        the CLI). `client` lets tests inject a pre-made socketio.Client (e.g.
        pointed at the mock server)."""
        self.url = url
        self.token = token
        self._log = logger or (lambda m: None)

        if C.PROTOCOL_SOURCE != "verified" and not allow_unverified:
            raise ProtocolNotVerified(
                "config.PROTOCOL_SOURCE is 'placeholder' -- the Socket.IO event "
                "names/payloads are guesses from the manual, not the real "
                "protocol. Refusing to drive a board. Wire up config.py from the "
                "client JS and set PROTOCOL_SOURCE='verified', or pass "
                "allow_unverified=True to run against the mock server.\n"
                f"  ({C.PROTOCOL_NOTES})"
            )

        self.sio = client or socketio.Client(
            reconnection=True, logger=False, engineio_logger=False
        )

        # Event log: every server->client event, appended under _cond.
        self._cond = threading.Condition()
        self._events: List[Tuple[float, str, Any]] = []  # (ts, name, data)
        # UART accumulates as text; guarded by the same condition.
        self._uart = ""
        self._last_status: Optional[str] = None

        self._install_handlers()

    # -- connection ---------------------------------------------------------
    def connect(self, timeout: float = 20.0) -> None:
        auth = None
        if C.CONNECT.auth_key and self.token:
            auth = {C.CONNECT.auth_key: self.token}
        self._log(f"connecting to {self.url} (ns={C.CONNECT.namespace})")
        self.sio.connect(
            self.url,
            auth=auth,
            socketio_path=C.CONNECT.socketio_path,
            namespaces=[C.CONNECT.namespace],
            wait=True,
            wait_timeout=timeout,
        )
        self._log("connected")

    def close(self) -> None:
        try:
            self.sio.disconnect()
        except Exception:  # pragma: no cover - best effort
            pass

    def __enter__(self) -> "FpgaConsole":
        self.connect()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -- handlers -----------------------------------------------------------
    def _install_handlers(self) -> None:
        ns = C.CONNECT.namespace

        # Catch-all: record every event so wait_event() can match generically.
        def catch_all(event: str, *args: Any) -> None:
            data = args[0] if len(args) == 1 else (args or None)
            self._record(event, data)

        self.sio.on("*", catch_all, namespace=ns)

    def _record(self, event: str, data: Any) -> None:
        now = time.monotonic()
        with self._cond:
            self._events.append((now, event, data))
            if event == C.LISTEN["uart_output"]:
                self._uart += self._extract_uart_text(data)
            elif event == C.LISTEN["status"]:
                self._last_status = self._extract_status(data)
            self._cond.notify_all()
        if event == C.LISTEN["uart_output"]:
            self._log(f"[uart] +{len(self._extract_uart_text(data))}B")
        else:
            self._log(f"[event] {event}: {data!r}")

    @staticmethod
    def _extract_uart_text(data: Any) -> str:
        if isinstance(data, str):
            return data
        if isinstance(data, (bytes, bytearray)):
            return data.decode("utf-8", "replace")
        if isinstance(data, dict):
            for k in C.UART_TEXT_KEYS:
                if k in data and data[k] is not None:
                    v = data[k]
                    if isinstance(v, (bytes, bytearray)):
                        return v.decode("utf-8", "replace")
                    return str(v)
        return "" if data is None else str(data)

    @staticmethod
    def _extract_status(data: Any) -> Optional[str]:
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            return data.get(C.STATUS_STATE_KEY) or data.get("status")
        return None

    # -- generic waiters ----------------------------------------------------
    def wait_event(
        self,
        name: str,
        predicate: Callable[[Any], bool] = lambda _d: True,
        timeout: float = 60.0,
        *,
        since: Optional[float] = None,
    ) -> Any:
        """Block until a `name` event whose data satisfies `predicate` arrives.
        Only events newer than `since` (monotonic ts) are considered; pass the
        value of `self.now()` captured before the triggering emit to avoid
        matching a stale event."""
        deadline = time.monotonic() + timeout
        cursor = 0
        with self._cond:
            while True:
                while cursor < len(self._events):
                    ts, ev, data = self._events[cursor]
                    cursor += 1
                    if (since is None or ts >= since) and ev == name and predicate(data):
                        return data
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ActionTimeout(f"timed out waiting for event {name!r}")
                self._cond.wait(timeout=remaining)

    def wait_status(self, state: str, timeout: float, error_status: str) -> None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                if self._last_status == state:
                    return
                if self._last_status == error_status:
                    raise ActionError(f"board reported status {error_status!r}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ActionTimeout(
                        f"timed out waiting for status {state!r} "
                        f"(last={self._last_status!r})"
                    )
                self._cond.wait(timeout=remaining)

    def wait_uart(
        self,
        pattern: Union[str, Pattern[str]],
        timeout: float = 120.0,
        *,
        search_from: int = 0,
    ) -> "re.Match[str]":
        """Block until the UART text matches `pattern`. Searched over the buffer
        from index `search_from` onward, so a marker split across chunks still
        matches, but a stale marker printed by an earlier command is skipped
        (pass the buffer length captured before the emit)."""
        rx = _compile(pattern)
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                m = rx.search(self._uart, search_from)
                if m:
                    return m
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    tail = self._uart[-400:]
                    raise ActionTimeout(
                        f"timed out waiting for UART /{rx.pattern}/; "
                        f"last 400B: {tail!r}"
                    )
                self._cond.wait(timeout=remaining)

    def now(self) -> float:
        return time.monotonic()

    @property
    def uart_text(self) -> str:
        with self._cond:
            return self._uart

    def drain_uart(self) -> str:
        with self._cond:
            buf, self._uart = self._uart, ""
            return buf

    # -- action primitives --------------------------------------------------
    def _emit(self, key: str, /, **kwargs: Any) -> Any:
        spec = C.EMIT[key]
        payload = spec.payload(**kwargs)
        ns = C.CONNECT.namespace
        self._log(f"emit {spec.event} <- {payload!r}")
        if spec.expects_ack:
            return self.sio.call(spec.event, payload, namespace=ns, timeout=30)
        if payload is None:
            self.sio.emit(spec.event, namespace=ns)
        else:
            self.sio.emit(spec.event, payload, namespace=ns)
        return None

    def _await_done(self, key: str, mark: float) -> None:
        spec = C.DONE_WHEN.get(key)
        if spec is None:
            return
        if spec.status is not None:
            self.wait_status(spec.status, spec.timeout_s, spec.error_status)
        elif spec.event is not None:
            self.wait_event(spec.event, spec.predicate, spec.timeout_s, since=mark)

    # -- the five actions ---------------------------------------------------
    def power(self, on: bool = True) -> None:
        self._emit("power", on=on)

    def upload_boot_image(self, name: str, data_b64: str, size: int) -> Any:
        """Action 1a: upload a .bin image; returns the ack (if any)."""
        mark = self.now()
        ack = self._emit("boot_image_upload", name=name, data_b64=data_b64, size=size)
        if not C.EMIT["boot_image_upload"].expects_ack:
            self._await_done("boot_image_upload", mark)
        return ack

    def load_boot_image(self, name: str) -> None:
        """Action 1b: load a stored image -> JTAG to 0x80000000; await complete."""
        mark = self.now()
        self._emit("boot_image_load", name=name)
        self._await_done("boot_image_load", mark)

    def reset(self, wait_prompt: bool = True, prompt_timeout: float = 180.0) -> None:
        """Action 2: reset; optionally wait for the Linux prompt on UART."""
        self.drain_uart()
        mark = self.now()
        self._emit("reset")
        self._await_done("reset", mark)
        if wait_prompt:
            self.wait_uart(C.UART["login_prompt"], timeout=prompt_timeout)

    def set_switch(self, index: int, on: bool = True) -> None:
        """Action 4: set virtual switch `index`."""
        self._emit("switch_set", index=index, on=on)

    def trace_dump(self, timeout: Optional[float] = None) -> Any:
        """Action 5: start Trace Dump; block for the end-of-dump frame.

        Per the manual the dump also needs switch 0 (detach UART) then switch 1
        (trigger); the caller sequences those with set_switch() around this call.
        """
        mark = self.now()
        self._emit("trace_dump")
        spec = C.DONE_WHEN["trace_dump"]
        return self.wait_event(
            spec.event, spec.predicate, timeout or spec.timeout_s, since=mark
        )

    # -- UART command helper (runs the .user/.dom pairs) --------------------
    def run_command(
        self,
        command: str,
        done_marker: Union[str, Pattern[str]],
        timeout: float = 180.0,
    ) -> str:
        """Type `command` into the terminal and return the UART text emitted
        until `done_marker` matches."""
        start = len(self.uart_text)
        self._emit("terminal_input", text=command + "\n")
        m = self.wait_uart(done_marker, timeout=timeout, search_from=start)
        with self._cond:
            return self._uart[start:m.end()]
