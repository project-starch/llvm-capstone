"""Protocol map for the FPGA web console (fpga.corank.info).

THIS FILE IS THE SINGLE WIRE-UP POINT. The rest of the driver is written against
the logical action names below; only this file names the *wire* protocol -- the
REST endpoints, the Socket.IO event names, and the payload shapes.

Source of the map: the console's own client JS (``static/app.js``), fetched from
the live site on 2026-07-16 and read directly (it is unminified). See PROTOCOL.md
for the observed event tables and the call sites each entry came from.

Key finding: the real console is a **hybrid**, NOT pure Socket.IO (the earlier
placeholder assumed pure Socket.IO). The action verbs -- upload / load / reset /
trace -- are **HTTP POST to a REST API** under ``<url>/api/...``; the live UART
stream and a few controls (power, switches, terminal keystrokes, the auto-shutdown
Lock) are **Socket.IO events**; and every long action reports completion through a
per-action Socket.IO *state* event (``load_state``, ``trace_result`` ...). So:

  * action verbs               -> HTTP  (see ``HTTP``)
  * live controls / keystrokes -> EMIT  (Socket.IO client->server)
  * stream + completion signals -> LISTEN (Socket.IO server->client)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Provenance guard.  The driver REFUSES to talk to the real board while this is
# "placeholder" (see fpga_console.py).  Verified against the live client JS.
# ---------------------------------------------------------------------------
PROTOCOL_SOURCE = "verified"  # "placeholder" | "verified"
PROTOCOL_NOTES = (
    "Observed from the console client JS (static/app.js), fetched from the live "
    "site 2026-07-16. Hybrid REST (action verbs) + Socket.IO (stream + state). "
    "Socket.IO client on the board is v4.7.5 (Engine.IO 4); python-socketio 5.x "
    "is compatible."
)


@dataclass(frozen=True)
class Connect:
    """How the Socket.IO client attaches to the console.

    The access token lives only in the URL *path* (``https://host/<token>/``).
    The board serves Socket.IO under ``/<token>/socket.io`` and the REST API under
    ``/<token>/api`` -- both derived at runtime from the token'd URL, so no token
    ever lives in this file. ``socketio_path`` below is the *suffix* the console
    appends to the URL path; ``FpgaConsole`` builds the full ``/<token>/socket.io``.
    """
    namespace: str = "/"
    socketio_path: str = "socket.io"   # suffix -> "/<url-path>/socket.io"
    api_prefix: str = "api"            # REST base -> "<url>/api"
    # The board authenticates purely by the path token; the Socket.IO handshake
    # carries no extra auth payload. Kept for parametrisation only.
    auth_key: Optional[str] = None


@dataclass(frozen=True)
class Emit:
    """A Socket.IO client->server action: emit `event` with a built payload."""
    event: str
    payload: Callable[..., Any] = field(default=lambda **kw: kw or None)
    expects_ack: bool = False


@dataclass(frozen=True)
class Http:
    """An HTTP REST action verb (the console fires these with fetch()).

    `body` returns the JSON dict to POST (or None for a bare POST). For a
    multipart upload, set `multipart=True` and have `body` return
    ``(fields_dict, files_dict)`` where files_dict maps a form field -> a path.

    Completion of a long action is reported by a Socket.IO *state* event, not by
    the HTTP response: set `done_event` to that event, and either `done_state`
    (wait until ``payload[state_key] == done_state``) or leave `done_state` None
    to treat the *first* `done_event` as completion (e.g. ``trace_result`` which
    only fires once, when the dump is ready).
    """
    method: str
    path: str                                   # relative to the api base
    body: Callable[..., Any] = field(default=lambda: None)
    multipart: bool = False
    done_event: Optional[str] = None
    done_state: Optional[str] = None
    error_state: str = "error"
    state_key: str = "state"
    timeout_s: float = 180.0


# ---------------------------------------------------------------------------
# LISTEN: server -> client Socket.IO events the driver subscribes to.
# (Names verified from app.js `socket.on(...)`.)
# ---------------------------------------------------------------------------
LISTEN = {
    # Live UART bytes: uart_data -> {seq, text}.
    "uart_output": "uart_data",
    # Per-action state streams (each carries {state: ...}).
    "load_state": "load_state",         # {state, loaded_image_name}
    "flash_state": "flash_state",       # {state, nv_bitstream_name}
    "trace_state": "trace_state",       # {state}: idle|capturing|done
    "power_state": "power_state",       # {state}: on|off
    "gdb_state": "gdb_state",           # {state}: idle|starting|running|error
    # One-shot / informational broadcasts.
    "trace_result": "trace_result",     # {text}: the finished trace dump
    "switch_state": "switch_state",     # {states: [0/1, ...]}
    "led_state": "led_state",           # {states: [0/1, ...]}
    "auto_shutdown_state": "auto_shutdown_state",  # {timeout, locked}
    "user_count": "user_count",         # {count}
}

# Key that carries the UART text inside a `uart_data` payload (verified: "text").
UART_TEXT_KEYS: List[str] = ["text"]

# Key carrying the monotonic sequence number on a `uart_data` payload
# ({seq, text}). The driver tracks the max seq seen and passes it as
# request_history{last_seq} on every reconnect, so the server replays only
# chunks newer than we've seen (seq > last_seq) instead of the whole 512 KB
# ring buffer -- which would otherwise duplicate the RESULT lines. Verified
# against socketio-api.md ("uart_data ... {seq, text}", request_history replays
# seq > last_seq; -1 = full replay).
UART_SEQ_KEY = "seq"

# Field inside a per-action state payload holding the state string.
STATUS_STATE_KEY = "state"

# Named state events + their value fields, for the toggle/lock helpers.
POWER_STATE_EVENT = "power_state"       # {state: "on"|"off"}
SWITCH_STATE_EVENT = "switch_state"     # {states: [...]}
SWITCH_STATE_KEY = "states"
LOCK_STATE_EVENT = "auto_shutdown_state"  # {timeout, locked}
LOCK_LOCKED_KEY = "locked"
LOCK_TIMEOUT_KEY = "timeout"
USER_COUNT_EVENT = "user_count"
USER_COUNT_KEY = "count"

# GDB debug session (used by the --boot-method=gdb path, which starts our image
# without the bootrom-reloading reset-board -- see PROTOCOL.md "GDB-driven boot").
GDB_STATE_EVENT = "gdb_state"           # {state}: idle|starting|running|error
GDB_OUTPUT_EVENT = "gdb_output"         # {data}: raw terminal text from the PTY
GDB_OUTPUT_KEY = "data"


# ---------------------------------------------------------------------------
# HTTP: the action verbs (POST). Paths are relative to "<url>/api".
# ---------------------------------------------------------------------------
HTTP = {
    # 1a. Upload a boot image to the console's image store (multipart form).
    #     app.js FileManager.upload(): POST /api/images/upload, FormData{name,file}
    #     -> 200 {name} | non-200 {error}. No async state; the response is final.
    "upload_image": Http(
        method="POST",
        path="images/upload",
        multipart=True,
        body=lambda name, file_path: ({"name": name}, {"file": file_path}),
    ),
    # 1b. Load a stored image -> JTAG to 0x80000000 (~2 min for a big image).
    #     app.js onAction: POST /api/load-image {filename} -> {state}. Completion
    #     via load_state {state}: loading -> done | error.
    "load_image": Http(
        method="POST",
        path="load-image",
        body=lambda filename: {"filename": filename},
        done_event=LISTEN["load_state"],
        done_state="done",
        error_state="error",
        timeout_s=300.0,
    ),
    # 2. Board reset. app.js resetBoard(): POST /api/reset-board. No state event;
    #    the caller waits for the Linux prompt on the UART stream instead.
    "reset": Http(method="POST", path="reset-board"),
    # 5. Trace dump. app.js toggleTrace(): POST /api/trace-start -> {state}. The
    #    finished dump arrives once as trace_result {text}; trace_state also goes
    #    capturing -> idle. trace_dump() owns the wait for trace_result (so no
    #    done_event here -- that would make the driver wait for it twice).
    "trace_start": Http(method="POST", path="trace-start", timeout_s=120.0),
    "trace_cancel": Http(method="POST", path="trace-cancel"),
}


# ---------------------------------------------------------------------------
# EMIT: Socket.IO client->server events (verified from app.js `socket.emit`).
# ---------------------------------------------------------------------------
EMIT = {
    # Power is a TOGGLE (no payload); the driver reads power_state and only
    # toggles when the current state differs from the requested one.
    "power_toggle": Emit(event="power_toggle", payload=lambda: None),
    # Terminal keystrokes -- how we type the .user/.dom commands over UART.
    "uart_send": Emit(event="uart_send", payload=lambda text: {"text": text}),
    "uart_clear": Emit(event="uart_clear", payload=lambda: None),
    # Virtual switches. switch_toggle is a TOGGLE by index; the driver reads
    # switch_state and only toggles when the bit differs from the target.
    "switch_toggle": Emit(event="switch_toggle", payload=lambda index: {"index": index}),
    "switch_reset_all": Emit(event="switch_reset_all", payload=lambda: None),
    # Auto-shutdown "Lock" (the good-citizen hold on the shared board).
    "set_auto_shutdown": Emit(
        event="set_auto_shutdown",
        payload=lambda timeout_seconds, locked: {
            "timeout_seconds": timeout_seconds, "locked": locked,
        },
    ),
    # Replay of the UART ring buffer since a sequence number (client sends this
    # on connect). Handy to resync; not required for a fresh post-reset run.
    "request_history": Emit(
        event="request_history", payload=lambda last_seq=-1: {"last_seq": last_seq}
    ),
    # GDB session control (no payload) + raw PTY keystrokes. gdb_input is how we
    # type `monitor reset halt` / `restore` / `set $pc` / `continue` to boot our
    # image WITHOUT the console reset-board button (which reloads SPI firmware).
    "gdb_start": Emit(event="gdb_start", payload=lambda: None),
    "gdb_stop": Emit(event="gdb_stop", payload=lambda: None),
    "gdb_input": Emit(event="gdb_input", payload=lambda text: {"text": text}),
}

CONNECT = Connect()

# ---------------------------------------------------------------------------
# UART markers used by the end-to-end rtl-smoke run (run_rtl_smoke.py). These are
# REAL and verified -- they come from the guest software, not the web protocol:
#   - the shell prompt to wait for after reset;
#   - the revoke-cost measurement-complete marker (build-revoke-cost-fpga.sh);
#   - the RESULT line format the parser in run-revoke-cost-fpga-qemu.sh consumes.
# ---------------------------------------------------------------------------
UART = {
    # Regex for the Linux shell prompt after boot. Buildroot default is
    # "# " at line start; widen if the board's prompt differs.
    "login_prompt": r"(?m)^(?:buildroot login:|.+ login:|/ #|~ #|# )\s*$",
    # Per-run completion marker printed by the domain payload.
    "measurement_complete": r"revoke-cost-fpga: measurement complete",
    "borrow_complete": r"borrow-cost-fpga: RESULT vs-raw",
    # The RESULT lines we harvest and hand to --parse-uart (kept loose; the
    # bundled parser does the strict extraction).
    "result_line": r"(?m)^.*RESULT.*$",
    # Sentinel echoed by the insmod step (the .doms need /dev/capstone, created
    # by capstone.ko -- the boot does NOT auto-load it). See run_rtl_smoke.
    "module_loaded": r"CAPSTONE_MOD_(?:OK|FAIL)",
}

# The GDB/OpenOCD interactive prompt, matched after each gdb command.
GDB_PROMPT = r"\(gdb\)\s*$"
