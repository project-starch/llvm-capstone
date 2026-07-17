"""Headless driver for the FPGA web console (hybrid REST + Socket.IO).

`FpgaConsole` performs the board actions the rtl-smoke run needs. The real
console is a hybrid (see config.py): the action *verbs* are HTTP POST to a REST
API, while the live UART stream, the power/switch controls, terminal keystrokes,
and the auto-shutdown Lock are Socket.IO events; each long action reports
completion through a per-action Socket.IO *state* event.

    1. upload_boot_image() / load_boot_image()  -> HTTP upload + load, await done
    2. reset() + wait_uart(login_prompt)         -> HTTP reset, wait for the shell
    3. wait_uart(marker)                          -> read UART until a marker
    4. set_switch(n, on)                          -> Socket.IO switch toggle
    5. trace_dump()                               -> HTTP trace-start, await result

The protocol (REST paths, event names, payloads, completion signals) is entirely
in config.py; this module is transport + synchronisation only.

Dependencies: python-socketio[client] (which also pulls in `requests`, used here
for the REST calls). Sync client: the Socket.IO event loop runs in a background
thread and invokes our handlers; we synchronise with the caller through a
Condition-guarded event log + per-event latest-state map + a UART text buffer.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Union
from urllib.parse import urlsplit, urlunsplit

try:
    import socketio
except ModuleNotFoundError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "python-socketio is required: pip install 'python-socketio[client]'"
    ) from exc

try:
    import requests
except ModuleNotFoundError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "requests is required (ships with python-socketio[client]): "
        "pip install requests"
    ) from exc

from . import config as C


class ProtocolNotVerified(RuntimeError):
    """Raised when asked to drive a real board with a placeholder protocol."""


class ActionTimeout(RuntimeError):
    pass


class ActionError(RuntimeError):
    """The board reported an error state while an action was in flight."""


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
        the CLI). `client` lets tests inject a pre-made socketio.Client."""
        self.url = url
        self.token = token
        self._log = logger or (lambda m: None)

        if C.PROTOCOL_SOURCE != "verified" and not allow_unverified:
            raise ProtocolNotVerified(
                "config.PROTOCOL_SOURCE is 'placeholder' -- refusing to drive a "
                "board. Wire config.py from the client JS and set "
                "PROTOCOL_SOURCE='verified', or pass allow_unverified=True to run "
                f"against the mock server.\n  ({C.PROTOCOL_NOTES})"
            )

        # Derive the Socket.IO path and REST base from the URL path token, so the
        # token stays only in `url` and never in config.py.  For a token'd URL
        # https://host/<token>/ this yields socketio_path=/<token>/socket.io and
        # api_base=https://host/<token>/api ; for a bare http://host:port (the
        # mock) it yields /socket.io and http://host:port/api .
        parts = urlsplit(url)
        prefix = parts.path.rstrip("/")
        self._socketio_path = f"{prefix}/{C.CONNECT.socketio_path}"
        self._api_base = urlunsplit(
            (parts.scheme, parts.netloc, f"{prefix}/{C.CONNECT.api_prefix}", "", "")
        )

        # Bounded reconnection: a board `reset` briefly drops the socket and we
        # must reconnect to catch the post-reset boot log -- but never in an
        # unbounded loop. Cap the attempts and the backoff so that if the console
        # backend is down we give up quickly instead of retrying forever (which
        # is the one way a client could keep knocking on a wedged backend).
        self.sio = client or socketio.Client(
            reconnection=True,
            reconnection_attempts=12,      # ~finite; give up rather than loop forever
            reconnection_delay=1,
            reconnection_delay_max=5,
            logger=False, engineio_logger=False,
        )
        self._http = requests.Session()

        # Event log + per-event latest payload + UART text buffer, all under _cond.
        self._cond = threading.Condition()
        self._events: List[Tuple[float, str, Any]] = []  # (ts, name, data)
        self._state: Dict[str, Any] = {}                 # event name -> last payload
        self._uart = ""
        # Highest uart_data `seq` seen so far. -1 means "nothing yet" -> a full
        # replay on the first request_history. Threading the real seq on every
        # reconnect is what stops the server re-injecting its whole (<=512 KB)
        # UART history and duplicating the RESULT lines the parser reads.
        self._last_uart_seq = -1

        self._install_handlers()

    # -- connection ---------------------------------------------------------
    def connect(self, timeout: float = 20.0) -> None:
        auth = None
        if C.CONNECT.auth_key and self.token:
            auth = {C.CONNECT.auth_key: self.token}
        self._log(f"connecting (ns={C.CONNECT.namespace}, path={self._socketio_path})")
        self.sio.connect(
            self.url,
            auth=auth,
            socketio_path=self._socketio_path,
            namespaces=[C.CONNECT.namespace],
            wait=True,
            wait_timeout=timeout,
        )
        # The 'connect' handler (installed in _install_handlers) mirrors the
        # browser client by emitting request_history on this and every reconnect.
        self._log("connected")

    def close(self) -> None:
        try:
            self.sio.disconnect()
        except Exception:  # pragma: no cover - best effort
            pass
        try:
            self._http.close()
        except Exception:  # pragma: no cover
            pass

    def __enter__(self) -> "FpgaConsole":
        self.connect()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -- handlers -----------------------------------------------------------
    def _install_handlers(self) -> None:
        ns = C.CONNECT.namespace

        def catch_all(event: str, *args: Any) -> None:
            data = args[0] if len(args) == 1 else (args or None)
            self._record(event, data)

        self.sio.on("*", catch_all, namespace=ns)

        # Re-request history on EVERY (re)connect, not just the first. A board
        # `reset` drops the Socket.IO connection; python-socketio reconnects
        # (reconnection=True), but the reconnected session must ask for the UART
        # ring buffer + current states again or it silently misses the post-reset
        # boot log (this is exactly what stalled the first hardware recon).
        def on_connect() -> None:
            try:
                self._resync_history()
            except Exception:  # pragma: no cover - best effort
                pass

        self.sio.on("connect", on_connect, namespace=ns)

    def _resync_history(self) -> None:
        """Ask the server to replay UART chunks newer than the last we've seen.
        Passing the tracked `seq` (not a hardcoded -1) means a reconnect only
        fills the gap since we dropped, rather than re-delivering the entire
        history buffer -- which on the real board would duplicate/garble the
        RESULT lines. -1 (the initial value) still gives a full replay the first
        time, so a fresh session gets the whole boot log."""
        with self._cond:
            last = self._last_uart_seq
        self._emit("request_history", last_seq=last)

    def _record(self, event: str, data: Any) -> None:
        now = time.monotonic()
        with self._cond:
            self._events.append((now, event, data))
            self._state[event] = data
            if event == C.LISTEN["uart_output"]:
                self._uart += self._extract_uart_text(data)
                seq = self._extract_uart_seq(data)
                if seq is not None and seq > self._last_uart_seq:
                    self._last_uart_seq = seq
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
    def _extract_uart_seq(data: Any) -> Optional[int]:
        """The monotonic `seq` on a uart_data payload, or None if absent. Used to
        thread request_history so a reconnect doesn't replay the whole buffer."""
        if isinstance(data, dict):
            s = data.get(C.UART_SEQ_KEY)
            if isinstance(s, bool):  # bool is an int subclass; not a seq
                return None
            if isinstance(s, int):
                return s
            if isinstance(s, str):
                try:
                    return int(s)
                except ValueError:
                    return None
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
        Only events newer than `since` (monotonic ts) are considered; pass
        `self.now()` captured before the trigger to skip a stale event."""
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

    def wait_state(
        self,
        event: str,
        done_state: str,
        timeout: float,
        *,
        error_state: str = "error",
        state_key: str = C.STATUS_STATE_KEY,
        since: Optional[float] = None,
    ) -> Any:
        """Block until a per-action state `event` reports `done_state` in its
        `state_key` field; raise ActionError if it reports `error_state`."""
        deadline = time.monotonic() + timeout
        cursor = 0
        with self._cond:
            while True:
                while cursor < len(self._events):
                    ts, ev, data = self._events[cursor]
                    cursor += 1
                    if ev != event or (since is not None and ts < since):
                        continue
                    st = data.get(state_key) if isinstance(data, dict) else data
                    if st == done_state:
                        return data
                    if st == error_state:
                        raise ActionError(f"{event}: board reported {error_state!r}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    last = self._state.get(event)
                    raise ActionTimeout(
                        f"timed out waiting for {event}=={done_state!r} (last={last!r})"
                    )
                self._cond.wait(timeout=remaining)

    def wait_uart(
        self,
        pattern: Union[str, Pattern[str]],
        timeout: float = 120.0,
        *,
        search_from: int = 0,
    ) -> "re.Match[str]":
        """Block until the UART text matches `pattern`, searched from index
        `search_from` onward -- so a marker split across chunks still matches, but
        a stale marker from an earlier command is skipped."""
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
                        f"timed out waiting for UART /{rx.pattern}/; last 400B: {tail!r}"
                    )
                self._cond.wait(timeout=remaining)

    def now(self) -> float:
        return time.monotonic()

    def latest(self, event: str) -> Any:
        """Most recent payload seen for `event`, or None."""
        with self._cond:
            return self._state.get(event)

    @property
    def uart_text(self) -> str:
        with self._cond:
            return self._uart

    @property
    def last_uart_seq(self) -> int:
        """Highest uart_data `seq` seen (-1 if none yet)."""
        with self._cond:
            return self._last_uart_seq

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

    def _post(self, key: str, /, **kwargs: Any) -> Any:
        """Fire an HTTP action verb; block for its state-event completion (if the
        action has one). Returns the parsed JSON response of the POST."""
        spec = C.HTTP[key]
        url = f"{self._api_base}/{spec.path}"
        mark = self.now()
        if spec.multipart:
            fields, files = spec.body(**kwargs)
            opened = {f: open(p, "rb") for f, p in files.items()}
            try:
                self._log(f"POST {spec.path} (multipart {list(files)})")
                resp = self._http.request(spec.method, url, data=fields, files=opened)
            finally:
                for fh in opened.values():
                    fh.close()
        else:
            body = spec.body(**kwargs)
            self._log(f"POST {spec.path} <- {body!r}")
            resp = self._http.request(spec.method, url, json=body)
        try:
            data = resp.json()
        except ValueError:
            data = None
        if not resp.ok:
            err = (data or {}).get("error") if isinstance(data, dict) else resp.text
            raise ActionError(f"{spec.path} -> HTTP {resp.status_code}: {err!r}")
        if spec.done_event:
            if spec.done_state is None:
                self.wait_event(spec.done_event, timeout=spec.timeout_s, since=mark)
            else:
                self.wait_state(
                    spec.done_event, spec.done_state, spec.timeout_s,
                    error_state=spec.error_state, state_key=spec.state_key, since=mark,
                )
        return data

    # -- the five actions ---------------------------------------------------
    def power(self, on: bool = True, timeout: float = 30.0) -> None:
        """Ensure board power is on/off. power_toggle is a *toggle*, so read the
        current power_state first and only toggle when it differs."""
        want = "on" if on else "off"
        cur = self._current_state(C.POWER_STATE_EVENT, timeout=5.0)
        if cur == want:
            self._log(f"power already {want}")
            return
        mark = self.now()
        self._emit("power_toggle")
        self.wait_state(C.POWER_STATE_EVENT, want, timeout, since=mark)

    def upload_boot_image(self, name: str, file_path: str) -> Any:
        """Action 1a: upload a boot image (multipart). Returns the JSON response
        (``{name: <stored-name>}``)."""
        return self._post("upload_image", name=name, file_path=file_path)

    def load_boot_image(self, name: str) -> None:
        """Action 1b: load a stored image -> JTAG to 0x80000000; await done."""
        self._post("load_image", filename=name)

    def reset(self, wait_prompt: bool = True, prompt_timeout: float = 180.0) -> None:
        """Action 2: reset the board; optionally wait for the Linux prompt."""
        self.drain_uart()
        self._post("reset")
        if wait_prompt:
            self.wait_uart(C.UART["login_prompt"], timeout=prompt_timeout)

    def set_switch(self, index: int, on: bool = True, timeout: float = 10.0) -> None:
        """Action 4: set virtual switch `index`. switch_toggle is a *toggle*, so
        read switch_state first and only toggle when the bit differs."""
        if self._switch_is(index) == on:
            return
        mark = self.now()
        self._emit("switch_toggle", index=index)
        self.wait_event(
            C.SWITCH_STATE_EVENT,
            lambda d: self._switch_val(d, index) == (1 if on else 0),
            timeout, since=mark,
        )

    def trace_dump(self, timeout: Optional[float] = None) -> str:
        """Action 5: start a trace dump; block for the finished dump text."""
        spec = C.HTTP["trace_start"]
        mark = self.now()
        self._post("trace_start")
        data = self.wait_event(
            C.LISTEN["trace_result"], timeout=timeout or spec.timeout_s, since=mark
        )
        if isinstance(data, dict):
            return str(data.get("text", ""))
        return str(data)

    # -- good-citizen helpers (shared board) --------------------------------
    def lock(self, timeout_seconds: Optional[int] = None, timeout: float = 10.0) -> None:
        """Take the auto-shutdown Lock so the board isn't reaped mid-run."""
        secs = timeout_seconds
        if secs is None:
            st = self.latest(C.LOCK_STATE_EVENT)
            secs = (st or {}).get(C.LOCK_TIMEOUT_KEY, 600) if isinstance(st, dict) else 600
        mark = self.now()
        self._emit("set_auto_shutdown", timeout_seconds=secs, locked=True)
        self.wait_event(
            C.LOCK_STATE_EVENT, lambda d: bool(d.get(C.LOCK_LOCKED_KEY)),
            timeout, since=mark,
        )

    def unlock(self, timeout_seconds: Optional[int] = None, timeout: float = 10.0) -> None:
        secs = timeout_seconds
        if secs is None:
            st = self.latest(C.LOCK_STATE_EVENT)
            secs = (st or {}).get(C.LOCK_TIMEOUT_KEY, 600) if isinstance(st, dict) else 600
        mark = self.now()
        self._emit("set_auto_shutdown", timeout_seconds=secs, locked=False)
        try:
            self.wait_event(
                C.LOCK_STATE_EVENT, lambda d: not d.get(C.LOCK_LOCKED_KEY),
                timeout, since=mark,
            )
        except ActionTimeout:  # pragma: no cover - releasing is best-effort
            pass

    def user_count(self, timeout: float = 5.0) -> Optional[int]:
        """Number of connected clients (including us), or None if unknown."""
        st = self.latest(C.USER_COUNT_EVENT)
        if st is None:
            try:
                st = self.wait_event(C.USER_COUNT_EVENT, timeout=timeout)
            except ActionTimeout:
                return None
        return st.get(C.USER_COUNT_KEY) if isinstance(st, dict) else None

    # -- state helpers ------------------------------------------------------
    def _current_state(self, event: str, timeout: float = 5.0) -> Optional[str]:
        st = self.latest(event)
        if st is None:
            try:
                st = self.wait_event(event, timeout=timeout)
            except ActionTimeout:
                return None
        return st.get(C.STATUS_STATE_KEY) if isinstance(st, dict) else st

    @staticmethod
    def _switch_val(data: Any, index: int) -> Optional[int]:
        if isinstance(data, dict):
            states = data.get(C.SWITCH_STATE_KEY)
            if isinstance(states, (list, tuple)) and index < len(states):
                return states[index]
        return None

    def _switch_is(self, index: int, timeout: float = 5.0) -> Optional[bool]:
        st = self.latest(C.SWITCH_STATE_EVENT)
        if st is None:
            try:
                st = self.wait_event(C.SWITCH_STATE_EVENT, timeout=timeout)
            except ActionTimeout:
                return None
        v = self._switch_val(st, index)
        return None if v is None else bool(v)

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
        self._emit("uart_send", text=command + "\r")
        m = self.wait_uart(done_marker, timeout=timeout, search_from=start)
        with self._cond:
            return self._uart[start:m.end()]
