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

import os
import re
import sys
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


def _redact(msg: str, url: str) -> str:
    """Strip the secret board URL/host/token out of a message before it can be logged."""
    out = str(msg)
    if url:
        out = out.replace(url, "<FPGA-CONSOLE-URL>")
        # host on its own (exception text often carries only host=..., not the full URL)
        try:
            from urllib.parse import urlparse
            h = urlparse(url).hostname
            if h:
                out = out.replace(h, "<FPGA-CONSOLE-HOST>")
        except Exception:
            pass
        # and any path segment, which is where the token lives
        for seg in url.split("/"):
            if len(seg) >= 12 and seg not in ("<FPGA-CONSOLE-URL>",):
                out = out.replace(seg, "<REDACTED>")
    return out


# Boards (2026-10-19, board #2 approved for the board lane, phase 1). A run names its board with FPGA_BOARD (1 or 2;
# unset means 1, so every script written before board #2 keeps meaning board #1). The board selects the console-URL
# secret file and the host-side lock. The constructor checks that the URL it was given IS that board's secret, so a
# wrapper that read the wrong file cannot drive the other board. Neither the URL nor the file content is ever printed.
BOARD_SECRETS = {"1": "~/.claude-kisp/secrets/fpga-console-url", "2": "~/.claude-kisp/secrets/fpga-console-url-board2"}
BOARD_LOCKS = {"1": "/tmp/capstone/.board.lock", "2": "/tmp/capstone/.board2.lock"}

# SERVER-SIDE LEASES (2026-10-19; the console's board-lease, first mode LEASE_ENFORCE=log). A lease is taken where the
# local board lock is taken and released where the console is closed; the local flock stays as the backstop. While held,
# every HTTP call carries X-Lease-Token and every Socket.IO (re)connect emits lease_bind. A thread renews it every
# LEASE_RENEW_S (agent leases last 10 min). A renew answered 409 "yield" (a person joined the queue) or "reserved" (a
# booking) sets lease_yield: the current cycle finishes, and power-ON is refused from then on, so no new cycle starts.
# A console without /api/lease (404) means leases are not available there: the run proceeds under the local lock only.
# FPGA_LEASE=off skips leases entirely; FPGA_LEASE_PURPOSE names the run (default: the driver and its pid).
LEASE_HOLDER = "board-lane"
LEASE_RENEW_S = 240.0


class LeaseYield(RuntimeError):
    """A person is waiting for the board (or a booking starts): no new cycle may start."""


def board_id() -> str:
    b = os.environ.get("FPGA_BOARD", "1").strip() or "1"
    if b not in BOARD_SECRETS:
        raise ValueError(f"FPGA_BOARD={b!r}: the boards are {sorted(BOARD_SECRETS)}")
    return b


def board_url(board: str) -> str:
    """The console URL of `board`, read from its secret file (mode 600). Never print it."""
    with open(os.path.expanduser(BOARD_SECRETS[board])) as fh:
        return fh.read().strip()


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
        # The board check (see BOARD_SECRETS). Mock and test consoles (allow_unverified, or a client injected) skip it.
        self.board = board_id()
        if not allow_unverified and client is None:
            if url.strip() != board_url(self.board):
                raise ValueError(f"FPGA_BOARD={self.board}: the console URL given is not board {self.board}'s "
                                 f"({BOARD_SECRETS[self.board]}); refusing to drive another board")
            print(f"[fpga-board] {self.board}", file=sys.stderr, flush=True)
        # THE SHARED-BOARD WATCH (2026-10-19). Board 2's console is shared with a human colleague, and nothing in the
        # console reserves it. On board 2 every console records the connected-client count (the most it saw is flagged
        # as possible human interference when above 1, this client), and a power or NV-bitstream transition that this
        # client did not make stops the run at once: the board is left exactly as found (no power-off, no release),
        # since a human may be mid-session, and the stop prints [fpga-foreign] so no wrapper retries over it.
        self._watch = (self.board == "2" and not allow_unverified and client is None)
        self._own_until = 0.0
        self._users = {"start": None, "max": None, "last": None, "flagged": False, "samples": []}
        self._last_power = None
        self._last_nv = None

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
        # Raw GDB/OpenOCD terminal text (gdb_output {data}), accumulated for the
        # --boot-method=gdb path. Separate from _uart so a gdb_wait on a prompt
        # doesn't collide with UART marker matching.
        self._gdb = ""
        # Highest uart_data `seq` seen so far. -1 means "nothing yet" -> a full
        # replay on the first request_history. Threading the real seq on every
        # reconnect is what stops the server re-injecting its whole (<=512 KB)
        # UART history and duplicating the RESULT lines the parser reads.
        self._last_uart_seq = -1
        # Server identity and stream integrity, all tolerating a server that does not yet
        # publish them. None means "never seen", which is distinct from any real value.
        self._server_epoch: Optional[int] = None
        self._server_restarted = False
        self._uart_gap_chunks = 0

        # Server-side lease (see LEASE_HOLDER): None until taken; "none" when the console has no lease API.
        self._lease_token: Optional[str] = None
        self._lease_state = "untaken"
        self.lease_yield = False
        self._lease_stop = threading.Event()
        self._lease_thread: Optional[threading.Thread] = None

        self._install_handlers()

    # -- connection ---------------------------------------------------------
    def _acquire_board_lock(self) -> None:
        """Take the exclusive board lock, or refuse to connect.

        THE BOARD IS ONE PHYSICAL RESOURCE AND SERIALIZATION WAS ONLY HALF-ENFORCED.
        `run-board-ladder.sh` has taken a non-blocking flock on this path for a long time,
        with a comment saying serialization "was previously a rule I was supposed to
        remember; now it is enforced". But the Python drivers -- which is what agents and
        subagents actually invoke -- never touched it, so two `run_sqlite_stages_fpga`
        processes ran on the board simultaneously on 2026-08-18 (ages 440 s and 288 s).
        Both runs were discarded: concurrent power-cycles, JTAG loads and switch writes
        interleave, so neither transcript describes what its own domains did.

        Enforcing it HERE rather than in each driver means every path that opens a console
        is covered, including any new one -- a rule in a wrapper script protects only the
        callers who remember the wrapper.

        NON-BLOCKING on purpose: a second session fails immediately and names the holder,
        rather than queueing invisibly behind a run that may itself be stuck. The lock is
        released when the process exits (the fd closes), so a crashed runner does not wedge
        the next one."""
        import fcntl  # local: only board sessions need it

        path = os.environ.get("BOARD_LOCK") or BOARD_LOCKS[getattr(self, "board", None) or board_id()]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Kept on the instance so the fd -- and therefore the lock -- lives as long as the
        # console does. A local would be garbage-collected and silently release it.
        self._board_lock_fd = open(path, "a+")
        try:
            fcntl.flock(self._board_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._board_lock_fd.seek(0)
            holder = self._board_lock_fd.read().strip() or "(unknown)"
            self._board_lock_fd.close()
            raise RuntimeError(
                f"another board session holds {path}\n"
                f"       holder: {holder}\n"
                f"       Wait for it, or stop it with SIGTERM (never -9: that orphans the\n"
                f"       server-side GDB session and every later run then times out).")
        self._board_lock_fd.seek(0)
        self._board_lock_fd.truncate()
        self._board_lock_fd.write(f"pid={os.getpid()} argv={' '.join(sys.argv)}\n")
        self._board_lock_fd.flush()

    def connect(self, timeout: float = 20.0) -> None:
        self._acquire_board_lock()
        try:
            self.lease_take()
        except Exception:
            self._board_lock_fd.close()   # refused: give the local lock back too
            raise
        auth = None
        if C.CONNECT.auth_key and self.token:
            auth = {C.CONNECT.auth_key: self.token}
        # The socket.io path is derived from the secret URL and CONTAINS THE TOKEN, so it
        # must go through _redact like every other logged string -- printing it raw leaked
        # the token into /tmp/capstone/board-*.log, which gets quoted into reports.
        self._log(
            _redact(
                f"connecting (ns={C.CONNECT.namespace}, path={self._socketio_path})",
                self.url,
            )
        )
        # REDACT THE BOARD HOST FROM ANY EXCEPTION. socketio/urllib3 put the full host --
        # and for some failure modes the token-bearing path -- into the exception message,
        # which then lands verbatim in a captured run log. The board URL is secret and a
        # hard project rule says it must never be echoed into a log. Seen for real on
        # 2026-07-29: a connect timeout wrote the hostname into a run log.
        try:
            self.sio.connect(
                self.url,
                auth=auth,
                socketio_path=self._socketio_path,
                namespaces=[C.CONNECT.namespace],
                wait=True,
                wait_timeout=timeout,
            )
        except Exception as e:
            msg = _redact(str(e), self.url)
            # A 5xx from the console is a WEB-UI OUTAGE, not a board fault. Confirmed with the
            # board owner 2026-08-05 after a session spent diagnosing "the board will not boot":
            # the console backend was returning 502 and no run could reach the hardware at all.
            # Say so in the exception, because the default socket.io text ("Unexpected status
            # code 502 in server response") reads like a transport hiccup and invites a retry
            # loop or, worse, a firmware bisection against a board nobody is talking to.
            # Word-boundary match on an HTTP-status-shaped token. A bare substring test
            # ("500" in msg) fires on "15000" or ":5000", and connect errors routinely carry
            # millisecond figures, ports and byte counts -- so a genuinely dead board could be
            # reported as "console is down, do not reflash or bisect", steering the operator
            # away from the real fault. Exactly the misdiagnosis this message exists to prevent.
            if re.search(r"status code (50[0-4])\b", msg):
                raise type(e)(
                    f"CONSOLE IS DOWN (server {msg}). This is a WEB-UI/server outage, NOT a "
                    f"board, bitstream or firmware fault -- do not reflash, rebuild or bisect "
                    f"on it. Nothing can reach the hardware until the console backend is back; "
                    f"it needs a restart by whoever operates it. Re-check with: "
                    f"curl -s -o /dev/null -w '%{{http_code}}' \"$FPGA_URL\""
                ) from None
            raise type(e)(msg) from None
        # The 'connect' handler (installed in _install_handlers) mirrors the
        # browser client by emitting request_history on this and every reconnect.
        self._log("connected")

    def close(self) -> None:
        self.users_report()
        self.lease_release()
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
                self._lease_bind()
            except Exception:  # pragma: no cover - best effort; log mode refuses nothing
                pass
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

    def users_report(self) -> None:
        """The client counts this console saw (board 2's watch): printed at close, so every run log carries them."""
        u = getattr(self, "_users", None)
        if getattr(self, "_watch", False) and u is not None:
            t0 = u["samples"][0][0] if u["samples"] else 0.0
            smp = " ".join(f"{n}@{t - t0:.0f}s" for t, n in u["samples"][-40:])
            print(f"[fpga-users] board {self.board} start {u['start']} max {u['max']} end {u['last']}"
                  f"{' FLAGGED' if u['flagged'] else ''}; samples (count@seconds): {smp}", file=sys.stderr, flush=True)

    def _foreign(self, what: str) -> None:
        """Stop the run: a change on a shared board that this client did not make. Board left as found."""
        print(f"[fpga-foreign] board {self.board}: {what}, not made by this run; stopping as INFRA, board left as found "
              f"(no power-off, no release); do not retry over a human session", file=sys.stderr, flush=True)
        os._exit(75)

    def _watch_event(self, now: float, event: str, data: Any) -> None:
        if event == C.USER_COUNT_EVENT and isinstance(data, dict):
            n = data.get(C.USER_COUNT_KEY)
            if isinstance(n, int):
                u = self._users
                if u["start"] is None:
                    u["start"] = n
                    print(f"[fpga-users] board {self.board} start {n}", file=sys.stderr, flush=True)
                u["last"] = n; u["max"] = n if u["max"] is None else max(u["max"], n)
                prev = u["samples"][-1] if u["samples"] else None
                u["samples"].append((now, n))
                # The baseline is the lane's own client count during a run. FPGA_USERS_BASELINE when measured; otherwise
                # the count this console first saw, so a client JOINING mid-run is still caught. Each board's health
                # check opens a 1-2 s Socket.IO connection every ~30 s (the board-host session, 2026-10-19), which is
                # board 1's 2/3 flicker, so a single sample one above the baseline is NOT a flag: two consecutive
                # samples above it more than 5 s apart are, and so is any sample above baseline + 1.
                base = os.environ.get("FPGA_USERS_BASELINE", "").strip()
                base = int(base) if base.isdigit() else u["start"]
                hit = n > base + 1 or (n > base and prev is not None and prev[1] > base and now - prev[0] > 5.0)
                if hit and not u["flagged"]:
                    u["flagged"] = True
                    print(f"[fpga-users] board {self.board}: {n} connected clients, above the baseline {base}: POSSIBLE "
                          f"HUMAN INTERFERENCE (another client, perhaps a human or an open browser tab, could act on the "
                          f"board); this run is not a clean reading", file=sys.stderr, flush=True)
        elif event == C.POWER_STATE_EVENT and isinstance(data, dict):
            st = data.get("state")
            if self._last_power is not None and st is not None and st != self._last_power and now > self._own_until:
                self._foreign(f"power_state {self._last_power} -> {st}")
            if st is not None:
                self._last_power = st
        elif event == C.LISTEN.get("flash_state") and isinstance(data, dict):
            nv = data.get("nv_bitstream_name")
            if self._last_nv is not None and nv is not None and nv != self._last_nv and now > self._own_until:
                self._foreign(f"nv_bitstream_name {self._last_nv} -> {nv}")
            if data.get("state") not in (None, "done", "idle") and now > self._own_until:
                self._foreign(f"flash_state {data.get('state')}")
            if nv is not None:
                self._last_nv = nv

    def _record(self, event: str, data: Any) -> None:
        now = time.monotonic()
        with self._cond:
            self._events.append((now, event, data))
            self._state[event] = data
            if getattr(self, "_watch", False):
                self._watch_event(now, event, data)
            # SERVER RESTART DETECTION. The server stamps every Socket.IO event with a
            # server_epoch minted at process start. A restart forces the board OFF
            # (init_gpio claims the power pin with initial value 0), so a run that spans one
            # is measuring a board that was power-cycled underneath it -- previously
            # indistinguishable from a board fault, and the single failure of mine that was
            # entirely unattributable.
            #
            # Recorded, never raised from here: this runs on the Socket.IO background thread,
            # where an exception would surface as a socket error somewhere unrelated. The run
            # loop reads `server_restarted` and reports void_reason itself.
            #
            # TOLERATES ABSENCE. The field does not exist on the server yet; until the
            # cutover every event lacks it and this stays inert. `is None` rather than
            # falsy, because epoch 0 is a legal value.
            ep = data.get("server_epoch") if isinstance(data, dict) else None
            if ep is not None:
                if self._server_epoch is None:
                    self._server_epoch = ep
                elif ep != self._server_epoch:
                    if not self._server_restarted:
                        self._log(f"!! SERVER RESTARTED mid-session: epoch "
                                  f"{self._server_epoch} -> {ep}. The board was POWERED OFF "
                                  f"by the restart; any result spanning this point is void.")
                    self._server_restarted = True
                    self._server_epoch = ep
            if event == C.LISTEN["uart_output"]:
                self._uart += self._extract_uart_text(data)
                seq = self._extract_uart_seq(data)
                if seq is not None:
                    # GAP DETECTION, keyed on (epoch, seq) rather than seq alone. Within an
                    # epoch the server increments seq by exactly 1 per chunk and never resets
                    # it -- uart_clear, power cycles and board resets all leave it alone -- so
                    # a jump of more than 1 means chunks were evicted or dropped between us.
                    # It returns to 0 only on process restart, which the epoch above announces,
                    # so a backwards seq is a restart rather than corruption and must not be
                    # counted as a gap.
                    if (self._last_uart_seq >= 0 and seq > self._last_uart_seq + 1
                            and not self._server_restarted):
                        missed = seq - self._last_uart_seq - 1
                        self._uart_gap_chunks += missed
                        self._log(f"!! UART GAP: seq {self._last_uart_seq} -> {seq} "
                                  f"({missed} chunk(s) never seen). Output is INCOMPLETE; a "
                                  f"missing marker is now unattributable.")
                    if seq > self._last_uart_seq:
                        self._last_uart_seq = seq
            elif event == C.GDB_OUTPUT_EVENT:
                self._gdb += self._extract_gdb_text(data)
            self._cond.notify_all()
        if event == C.LISTEN["uart_output"]:
            # Log the DECODED text, not just a byte count. Previously this printed
            # "[uart] +145B" and the text only became visible when run_command()
            # RETURNED, so a run in progress looked identical to a dead one -- which is
            # why board progress was invisible mid-run and why conclusions were twice
            # drawn from a 400-byte truncated tail after the fact. Costs nothing and
            # makes a live run readable.
            _txt = self._extract_uart_text(data)
            self._log("[uart] " + repr(_txt) if _txt else f"[uart] +0B")
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

    @staticmethod
    def _extract_gdb_text(data: Any) -> str:
        """Text from a gdb_output payload ({data: "..."}), or the raw value."""
        if isinstance(data, dict):
            v = data.get(C.GDB_OUTPUT_KEY)
            if v is not None:
                if isinstance(v, (bytes, bytearray)):
                    return v.decode("utf-8", "replace")
                return str(v)
        if isinstance(data, (bytes, bytearray)):
            return data.decode("utf-8", "replace")
        return "" if data is None else str(data)

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
        idle_timeout: Optional[float] = None,
    ) -> "re.Match[str]":
        """Block until the UART text matches `pattern`, searched from index
        `search_from` onward -- so a marker split across chunks still matches, but
        a stale marker from an earlier command is skipped.

        `idle_timeout`, when set, aborts early if NO new UART byte arrives for that
        many seconds. A wedged domain produces total silence, so the overall
        `timeout` is spent waiting for something that will never come: a SQLite run
        sat idle for 10.5 minutes of a 15-minute budget with a frozen log, and every
        such run costs a board session. Progress resets the idle clock, so a slow but
        live workload is unaffected -- only silence trips it.

        Off by default so existing callers keep their exact behaviour."""
        rx = _compile(pattern)
        deadline = time.monotonic() + timeout
        last_len = len(self._uart)
        last_progress = time.monotonic()
        with self._cond:
            while True:
                m = rx.search(self._uart, search_from)
                if m:
                    return m
                now = time.monotonic()
                if len(self._uart) != last_len:
                    last_len = len(self._uart)
                    last_progress = now
                if idle_timeout is not None and now - last_progress >= idle_timeout:
                    tail = self._uart[-400:]
                    raise ActionTimeout(
                        f"UART idle {idle_timeout:.0f}s while waiting for "
                        f"/{rx.pattern}/ (wedged, not slow); last 400B: {tail!r}"
                    )
                remaining = deadline - now
                if remaining <= 0:
                    tail = self._uart[-400:]
                    raise ActionTimeout(
                        f"timed out waiting for UART /{rx.pattern}/; last 400B: {tail!r}"
                    )
                if idle_timeout is not None:
                    remaining = min(remaining, idle_timeout - (now - last_progress))
                self._cond.wait(timeout=max(0.05, remaining))

    def wait_gdb(
        self,
        pattern: Union[str, Pattern[str]],
        timeout: float = 60.0,
        *,
        search_from: int = 0,
    ) -> "re.Match[str]":
        """Block until the accumulated GDB terminal text matches `pattern`,
        searched from `search_from` onward (so a fresh prompt after a command is
        matched, not a stale one)."""
        rx = _compile(pattern)
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                m = rx.search(self._gdb, search_from)
                if m:
                    return m
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    tail = self._gdb[-400:]
                    raise ActionTimeout(
                        f"timed out waiting for GDB /{rx.pattern}/; last 400B: {tail!r}"
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
    def gdb_text(self) -> str:
        with self._cond:
            return self._gdb

    def get_state(self, timeout: float = 15.0) -> Dict[str, Any]:
        """One GET /api/state instead of assembling six socket events.

        Also the ONLY way to see three things no event carries: whether UART chunks were
        dropped server-side, the oldest sequence still retained, and whether the serial port
        is open at all. Returns {} if the server does not implement it yet, so callers can
        adopt this before the cutover -- an empty dict is distinguishable from a real answer,
        which a default-filled dict would not be.
        """
        try:
            r = self._http.get(f"{self._api_base}/state", timeout=timeout)
            if r.status_code != 200:
                return {}
            st = r.json()
        except Exception as exc:
            self._log(_redact(f"get_state failed: {exc}", self.url))
            return {}
        if not isinstance(st, dict):
            return {}
        ep = st.get("server_epoch")
        if ep is not None:
            if self._server_epoch is None:
                self._server_epoch = ep
            elif ep != self._server_epoch:
                self._server_restarted = True
                self._server_epoch = ep
        return st

    def integrity(self) -> Dict[str, Any]:
        """Everything that decides whether this run's output can be believed.

        Call it before recording any verdict. A run that spans a server restart measured a
        board that was power-cycled underneath it; a run with a UART gap or a server-side
        drop is missing output, and a MISSING MARKER IS A RESULT in this project's
        classification -- absence of `SQ: G/enter` is how an entry stall is told from a
        wedge. So an unnoticed drop does not degrade a verdict, it INVERTS one.
        """
        st = self.get_state()
        dropped = (st.get("uart_dropped_chunks") or 0) + (st.get("uart_dropped_bytes") or 0)
        first = st.get("uart_first_seq")
        evicted = (first is not None and self._last_uart_seq >= 0
                   and first > self._last_uart_seq)
        # `ok` is "no positive evidence of a problem". It is NOT "verified clean", and the
        # difference is the whole point: with no /api/state the server-side drop counters are
        # unreadable, so a silent truncation is indistinguishable from a quiet board. Reporting
        # ok=True there would be an unearned pass -- the exact class this pair of counters
        # exists to remove -- so `checks_complete` says whether the question could be asked at
        # all, and a caller that reads `ok` without it is reading a half-answer.
        ok = not (self._server_restarted or self._uart_gap_chunks or dropped or evicted)
        return {
            "ok": ok,
            "checks_complete": bool(st),
            "server_restarted": self._server_restarted,
            "server_epoch": self._server_epoch,
            "uart_gap_chunks": self._uart_gap_chunks,
            "uart_dropped_chunks": st.get("uart_dropped_chunks"),
            "uart_dropped_bytes": st.get("uart_dropped_bytes"),
            "uart_first_seq": first,
            "uart_last_seq_seen": self._last_uart_seq,
            "evicted_before_read": evicted,
            "void_reason": ("server_restarted" if self._server_restarted else
                            "uart_gap" if self._uart_gap_chunks else
                            "uart_dropped_serverside" if dropped else
                            "uart_evicted_unread" if evicted else None),
            "state_endpoint_present": bool(st),
        }

    @property
    def server_restarted(self) -> bool:
        """True if server_epoch changed since the first event seen this session."""
        return self._server_restarted

    @property
    def uart_gap_chunks(self) -> int:
        """Count of uart_data chunks whose seq we never observed."""
        return self._uart_gap_chunks

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

    # -- server-side lease ---------------------------------------------------
    def _lease_url(self, sub: str = "") -> str:
        return f"{self._api_base}/lease{sub}"

    def _lease_say(self, msg: str) -> None:
        print(f"[fpga-lease] board {self.board}: {msg}", file=sys.stderr, flush=True)

    def lease_info(self, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        """GET /api/lease (open): holder, until, queue, offer, reservations, enforce. None without a lease API."""
        r = self._http.get(self._lease_url(), timeout=timeout)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def lease_take(self, purpose: Optional[str] = None, timeout: float = 15.0) -> None:
        """Take the board's lease, or refuse to proceed when someone else holds, is offered or has booked it."""
        if os.environ.get("FPGA_LEASE", "").strip().lower() == "off":
            self._lease_state = "off"
            self._lease_say("FPGA_LEASE=off: no lease; local lock only")
            return
        purpose = purpose or os.environ.get("FPGA_LEASE_PURPOSE") or f"{os.path.basename(sys.argv[0])} pid {os.getpid()}"
        try:
            r = self._http.post(self._lease_url(), json={"holder": LEASE_HOLDER, "purpose": purpose, "kind": "agent"},
                                timeout=timeout)
        except Exception as e:
            raise RuntimeError("[fpga-lease] lease request failed: " + _redact(str(e), self.url)) from None
        if r.status_code == 404:
            self._lease_state = "none"
            self._lease_say("no lease API (404): leases not available; local lock only")
            return
        try:
            d = r.json()
        except ValueError:
            d = {}
        if r.status_code == 409:
            raise RuntimeError(f"[fpga-lease] board {self.board} is not free: {d.get('code')} by {d.get('holder')!r} "
                               f"until {d.get('until')}; not starting")
        if not r.ok or not d.get("token"):
            raise RuntimeError(f"[fpga-lease] lease take -> HTTP {r.status_code}: {d.get('error', r.text[:200])!r}")
        self._lease_token = d["token"]
        self._lease_state = "held"
        self._http.headers["X-Lease-Token"] = self._lease_token
        self._lease_say(f"held, until {d.get('until')}, purpose {purpose!r}")
        self._lease_stop.clear()
        self._lease_thread = threading.Thread(target=self._lease_renew_loop, name="lease-renew", daemon=True)
        self._lease_thread.start()

    def _lease_bind(self) -> None:
        if self._lease_token:
            self.sio.emit("lease_bind", {"lease_token": self._lease_token}, namespace=C.CONNECT.namespace)

    def lease_renew(self, timeout: float = 15.0) -> Optional[str]:
        """Renew once. Returns None on success, else the 409 code (yield/reserved set lease_yield) or an error text."""
        if not self._lease_token:
            return None
        try:
            r = self._http.post(self._lease_url("/renew"), timeout=timeout)
        except Exception as e:
            self._lease_say("renew failed (kept): " + _redact(str(e), self.url))
            return "error"
        try:
            d = r.json()
        except ValueError:
            d = {}
        if r.ok:
            return None
        code = d.get("code") or f"http {r.status_code}"
        if r.status_code == 409 and code in ("yield", "reserved"):
            if not self.lease_yield:
                self._lease_say(f"YIELD ({code}, waiting {d.get('waiting')}): finishing the current cycle; "
                                f"no new cycle will start")
            self.lease_yield = True
        else:
            self._lease_say(f"renew refused: {code}")
        return code

    def _lease_renew_loop(self) -> None:
        while not self._lease_stop.wait(LEASE_RENEW_S):
            self.lease_renew()

    def lease_release(self, timeout: float = 15.0) -> None:
        """DELETE /api/lease with the token. Best effort: an unreleased lease expires on the server."""
        self._lease_stop.set()
        tok, self._lease_token = self._lease_token, None
        if not tok:
            return
        try:
            r = self._http.delete(self._lease_url(), headers={"X-Lease-Token": tok}, timeout=timeout)
            self._lease_say(f"released (HTTP {r.status_code})")
        except Exception as e:
            self._lease_say("release failed (the lease expires on the server): " + _redact(str(e), self.url))
        self._http.headers.pop("X-Lease-Token", None)
        self._lease_state = "released"

    # -- the five actions ---------------------------------------------------
    def power(self, on: bool = True, timeout: float = 30.0) -> None:
        """Ensure board power is on/off. power_toggle is a *toggle*, so read the
        current power_state first and only toggle when it differs."""
        want = "on" if on else "off"
        if on and self.lease_yield:
            raise LeaseYield("[fpga-lease] a person is waiting for the board (or a booking starts): "
                             "no new cycle may start; finish and release")
        cur = self._current_state(C.POWER_STATE_EVENT, timeout=5.0)
        if cur == want:
            self._log(f"power already {want}")
            return
        mark = self.now()
        self._own_until = mark + timeout + 30.0   # this client's own power transition (the shared-board watch)
        self._emit("power_toggle")
        self.wait_state(C.POWER_STATE_EVENT, want, timeout, since=mark)

    def upload_boot_image(self, name: str, file_path: str) -> Any:
        """Action 1a: upload a boot image (multipart). Returns the JSON response
        (``{name: <stored-name>}``)."""
        return self._post("upload_image", name=name, file_path=file_path)

    def flash_bitstream(self, name: str) -> Any:
        """Flash a stored bitstream to the FPGA (PERSISTENT non-volatile write --
        reprograms the shared board). Blocks until flash_state=done."""
        self._own_until = self.now() + 900.0      # this client's own flash (the shared-board watch)
        return self._post("flash_bitstream", filename=name)

    def load_boot_image(self, name: str, *, attempts: int = 3,
                        settle: float = 6.0) -> None:
        """Action 1b: load a stored image -> JTAG to 0x80000000; await done.

        A load issued right after power-on can transiently fail: OpenOCD/GDB
        can't attach to the JTAG TAP until the core has settled, and the console
        reports `load_state: error`. That is recoverable, so retry a few times
        with a short settle between attempts (each attempt re-issues the POST;
        the prior attempt has reached the terminal `error` state, so there is no
        `Already loading` 409). Only a persistent error propagates."""
        last: Optional[ActionError] = None
        for i in range(1, attempts + 1):
            try:
                self._post("load_image", filename=name)
                return
            except ActionError as exc:
                last = exc
                self._log(f"load attempt {i}/{attempts} failed ({exc}); "
                          f"settling {settle}s before retry")
                if i < attempts:
                    time.sleep(settle)
        assert last is not None
        raise last

    def reset(self, wait_prompt: bool = True, prompt_timeout: float = 180.0) -> None:
        """Action 2: reset the board; optionally wait for the Linux prompt."""
        self.drain_uart()
        self._post("reset")
        if wait_prompt:
            self.wait_uart(C.UART["login_prompt"], timeout=prompt_timeout)

    # -- GDB-driven boot (self-serve: no reset-board, flashes nothing) -------
    def gdb_start(self, timeout: float = 60.0) -> None:
        """Open the OpenOCD + gdb-multiarch session; wait for gdb_state=running.
        No-ops on the server unless gdb_state is idle|error."""
        mark = self.now()
        self._emit("gdb_start")
        self.wait_event(
            C.GDB_STATE_EVENT,
            lambda d: isinstance(d, dict) and d.get(C.STATUS_STATE_KEY) == "running",
            timeout, since=mark,
        )

    def gdb_stop(self, timeout: float = 20.0) -> None:
        """End the GDB session (resumes the CPU, tears down OpenOCD). Best-effort
        wait for gdb_state=idle."""
        mark = self.now()
        self._emit("gdb_stop")
        try:
            self.wait_event(
                C.GDB_STATE_EVENT,
                lambda d: isinstance(d, dict) and d.get(C.STATUS_STATE_KEY) == "idle",
                timeout, since=mark,
            )
        except ActionTimeout:  # pragma: no cover - stopping is best-effort
            pass

    def gdb_cmd(
        self,
        command: str,
        done_marker: Union[str, Pattern[str]],
        timeout: float = 120.0,
    ) -> str:
        """Send one GDB command (as PTY keystrokes) and return the gdb_output text
        up to `done_marker` (typically the `(gdb) ` prompt)."""
        start = len(self.gdb_text)
        self._emit("gdb_input", text=command if command.endswith("\n") else command + "\n")
        m = self.wait_gdb(done_marker, timeout=timeout, search_from=start)
        with self._cond:
            return self._gdb[start:m.end()]

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
    # -- run-scoped UART access -------------------------------------------
    # The UART buffer ACCUMULATES across every command and every boot in a session, and
    # the captured log file accumulates across SESSIONS. Grepping that file and attributing
    # what you find to the current run is how a whole day's analysis went wrong on
    # 2026-07-30: an ILLX tag and an MEPC value from a firmware three rebuilds earlier were
    # read as current output, and a monitor tag that never fired at all was assumed present.
    # Any diagnostic read MUST be bounded by a mark taken before the command ran.
    def uart_mark(self) -> int:
        """Buffer position to read from. Take this BEFORE issuing a command."""
        with self._cond:
            return len(self._uart)

    def uart_since(self, mark: int) -> str:
        """UART text produced since `mark` -- the only provenance-safe read."""
        with self._cond:
            return self._uart[mark:]

    # 16 = the ns16550a RX FIFO depth, so one burst fills it at most once.
    UART_BURST = int(os.environ.get("FPGA_UART_BURST", "16") or 0)
    UART_BURST_DELAY = float(os.environ.get("FPGA_UART_BURST_DELAY") or 0.02)
    UART_ECHO_TIMEOUT = float(os.environ.get("FPGA_UART_ECHO_TIMEOUT") or 1.5)

    @staticmethod
    def _echo_normalise(text: str) -> str:
        """Echoed input as the shell RECEIVED it.

        A long command wraps in the terminal, which injects CR/LF (and sometimes colour
        escapes) INTO THE MIDDLE OF A WORD -- an observed echo split
        `/lib/ld-linu` + newline + `x-riscv64-lp64d.so.1`. Comparing raw text would call
        that a corruption and fall back every time, which would silently undo the speedup
        while looking like it worked. Spaces are preserved so a DROPPED space still fails
        the check."""
        return re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text).replace("\r", "").replace("\n", "")

    def _type_burst(self, command: str, start: int) -> bool:
        """Type `command` (no Enter) in bursts. True iff the shell echoed it back intact."""
        if self.UART_BURST <= 1:
            return False                      # kill switch: take the char-by-char path
        for i in range(0, len(command), self.UART_BURST):
            self._emit("uart_send", text=command[i:i + self.UART_BURST])
            time.sleep(self.UART_BURST_DELAY)
        deadline = time.time() + self.UART_ECHO_TIMEOUT
        while time.time() < deadline:
            if command in self._echo_normalise(self.uart_text[start:]):
                return True
            time.sleep(0.05)
        return False

    def run_command(
        self,
        command: str,
        done_marker: Union[str, Pattern[str]],
        timeout: float = 180.0,
        idle_timeout: Optional[float] = None,
    ) -> str:
        """Type `command` into the terminal and return the UART text emitted
        until `done_marker` matches.

        The board's UART RX FIFO overruns on a bulk write and silently drops
        characters (a long command like the .user/.dom path arrives corrupted
        and the shell errors on a garbled path -- `borrow_cost` -> `row_cost`).
        A leading Ctrl-U clears any partial line a prior dropped char may have
        left in the input buffer.

        The char-by-char throttle that mitigated this cost 50 ms PER CHARACTER, which made
        typing a command dramatically more expensive per byte than SENDING A WHOLE PROGRAM:
        fast_xfer moves 16 chars per emit at 0.02 s (1.25 ms/char), so file transfer was 40x
        faster per byte than typing `echo ...`. Measured on a real SQLite session, its four
        commands are ~266 typed characters = ~13 s of pure sleep.

        So type in bursts and VERIFY, rather than typing slowly and hoping. 16 is the
        ns16550a FIFO depth, so a burst fills it at most exactly once and drains in ~1.4 ms
        at 115200 baud -- an order of magnitude under the inter-burst delay. The bound comes
        from the hardware, not from a guess.

        The verification is the part that makes this safe, and it is why the burst could not
        simply be copied from fast_xfer: fast_xfer re-reads each chunk's sha and retries, so
        it can afford to be wrong, whereas run_command has NO integrity check -- a corrupted
        command just runs and produces a misleading failure. Here the shell's own echo is
        the check: if the typed line does not come back intact before Enter is pressed, the
        line is cleared and retyped on the proven char-by-char tier. Worst case is the old
        behaviour plus one short echo wait; a silently garbled command is not possible.

        FPGA_UART_BURST=0 forces the old path outright."""
        self._emit("uart_send", text="\x15")  # Ctrl-U: clear the input line
        time.sleep(0.2)
        start = len(self.uart_text)
        if not self._type_burst(command, start):
            # Echo did not come back intact (or the shell is not echoing at all): clear the
            # line and fall back to the throttle that is known to work.
            self._emit("uart_send", text="\x15")
            time.sleep(0.2)
            start = len(self.uart_text)
            for ch in command:
                self._emit("uart_send", text=ch)
                time.sleep(0.05)
        self._emit("uart_send", text="\r")
        m = self.wait_uart(done_marker, timeout=timeout, search_from=start,
                           idle_timeout=idle_timeout)
        with self._cond:
            return self._uart[start:m.end()]
