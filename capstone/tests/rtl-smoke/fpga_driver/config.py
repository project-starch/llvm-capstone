"""Socket.IO protocol map for the FPGA web console (fpga.corank.info).

THIS FILE IS THE SINGLE WIRE-UP POINT. The rest of the driver is written against
the logical action names below; only this file names the *wire* Socket.IO events
and payload shapes. When the console's client JS arrives (grep it for
`socket.emit(`, `.on(`, `io(`), edit the values here and flip PROTOCOL_SOURCE to
"verified" -- nothing else in the driver should need to change.

Every wire event / payload key here is a PLACEHOLDER inferred from the user
manual (FPGA_Remote_Manual.md), NOT from the real protocol. The mock server
(mock_server.py) implements exactly these placeholders so the flow can be
exercised offline. Placeholders are named after the manual's UI vocabulary so the
mapping to the real events is mechanical.

Fill-in checklist when the JS lands (see PROTOCOL.md for the full procedure):
  1. namespace / connection URL suffix + how the access token is passed
     (query string? auth payload? path?)  -> CONNECT
  2. the emit event name + payload for each of: upload image, load image, reset,
     set switch, trace dump, terminal input                    -> EMIT
  3. the server->client event(s) that carry: UART bytes, status changes,
     trace frames / end-of-dump                                 -> LISTEN
  4. the completion signal for each long action (status string? dedicated
     event? ack callback?)                                      -> DONE_WHEN
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Provenance guard.  The driver REFUSES to talk to the real board while this is
# "placeholder" (see fpga_console.py). Flip to "verified" only after the wire
# events below have been confirmed against the real client JS or a live capture.
# ---------------------------------------------------------------------------
PROTOCOL_SOURCE = "placeholder"  # "placeholder" | "verified"
PROTOCOL_NOTES = (
    "Inferred from FPGA_Remote_Manual.md; event names/payloads are guesses. "
    "Replace from the console client JS or a DevTools WS capture, then set "
    "PROTOCOL_SOURCE='verified'."
)


@dataclass(frozen=True)
class Connect:
    """How the Socket.IO client attaches to the console."""
    # Path component after the origin, e.g. https://fpga.corank.info/<token>/ .
    # Left empty here: pass the token'd URL to the driver on the command line so
    # the secret never lives in the repo.
    namespace: str = "/"
    # socketio path (the HTTP endpoint Socket.IO polls/upgrades on). Default is
    # "socket.io"; some deployments customise it.
    socketio_path: str = "socket.io"
    # How the access token in the URL path is presented to the server. Most
    # token-in-path deployments still complete the Socket.IO handshake fine with
    # the token only in the connect URL; if the server wants it echoed, set
    # auth_key and the driver puts {auth_key: <token>} in the connect auth dict.
    auth_key: Optional[str] = None  # e.g. "token"; None = token only in URL


@dataclass(frozen=True)
class Emit:
    """A client->server action: emit `event` with a payload built from kwargs."""
    event: str
    # Build the payload dict from call kwargs. Default: pass kwargs through.
    payload: Callable[..., Any] = field(default=lambda **kw: kw or None)
    # If the server replies via a Socket.IO ack callback, set True and the driver
    # waits on the ack instead of (or in addition to) a DONE_WHEN event.
    expects_ack: bool = False


@dataclass(frozen=True)
class DoneWhen:
    """How the driver knows a long-running action finished.

    Exactly one of (status, event) is the trigger; `predicate` further filters
    the payload. status compares against the STATUS event's state field.
    """
    status: Optional[str] = None          # e.g. "Done" (from the status stream)
    event: Optional[str] = None           # e.g. a dedicated "load_complete" event
    predicate: Callable[[Any], bool] = field(default=lambda _data: True)
    error_status: str = "Error"           # status value that means the action failed
    timeout_s: float = 180.0


# ---------------------------------------------------------------------------
# LISTEN: server -> client events the driver subscribes to.
# ---------------------------------------------------------------------------
LISTEN = {
    # Stream of UART bytes from the board. Payload assumed to carry the bytes in
    # one of the keys in `uart_text_keys` (first present wins), or to be a bare
    # str. Adjust keys to match the real payload.
    "uart_output": "terminal_output",
    # Board status changes: Idle / Loading / Flashing / Capturing / Done / Error.
    "status": "status",
    # Trace frames + end-of-dump (tracer path; not needed for the cycle-count run).
    "trace_frame": "trace_frame",
    "trace_complete": "trace_complete",
    # Live LED / switch state broadcasts (informational).
    "leds": "leds",
    "switches": "switches",
}

# Candidate keys that may hold the UART text inside a `uart_output` payload.
# The driver tries these in order, then falls back to str(payload).
UART_TEXT_KEYS: List[str] = ["data", "text", "bytes", "chunk", "output"]

# Field inside a `status` payload holding the state string.
STATUS_STATE_KEY = "state"


# ---------------------------------------------------------------------------
# EMIT + DONE_WHEN: the five board actions (+ helpers) the task calls for.
# ---------------------------------------------------------------------------
EMIT = {
    # 1a. Upload a .bin boot image to the console's image store.
    #     Payload placeholder: {name, size, data(base64)} -- the real console may
    #     chunk large uploads over multiple events; see PROTOCOL.md.
    "boot_image_upload": Emit(
        event="boot_image_upload",
        payload=lambda name, data_b64, size: {
            "name": name, "size": size, "data": data_b64,
        },
        expects_ack=True,
    ),
    # 1b. Load a stored image -> JTAG to 0x80000000 (~2 min for ~15 MB).
    "boot_image_load": Emit(
        event="boot_image_load",
        payload=lambda name: {"name": name},
    ),
    # 2. Board reset.
    "reset": Emit(event="reset", payload=lambda: None),
    # 4. Set a virtual switch (Trace Dump prep: switch 0 detaches UART, switch 1
    #    triggers the dump; switch 2 = replacement policy).
    "switch_set": Emit(
        event="switch_set",
        payload=lambda index, on: {"index": index, "on": on},
    ),
    # 5. Start a trace capture (button: "Trace Dump" -> status "Capturing").
    "trace_dump": Emit(event="trace_dump", payload=lambda: None),
    # Terminal keystrokes -- how we run the .user/.dom commands over UART.
    "terminal_input": Emit(
        event="terminal_input",
        payload=lambda text: {"data": text},
    ),
    # Power on (most actions require power). Optional in the run if already on.
    "power": Emit(event="power", payload=lambda on: {"on": on}),
}

DONE_WHEN = {
    # Upload: assume an ack callback (expects_ack=True) OR a Done status.
    "boot_image_upload": DoneWhen(status="Done", timeout_s=120.0),
    # Load: Loading... -> Done (JTAG transfer).
    "boot_image_load": DoneWhen(status="Done", timeout_s=300.0),
    # Reset: no dedicated completion; the caller instead waits for the Linux
    # prompt on the UART stream (wait_uart). Kept short as a safety net.
    "reset": DoneWhen(status="Idle", timeout_s=30.0),
    # Trace dump end-of-dump frame.
    "trace_dump": DoneWhen(event=LISTEN["trace_complete"], timeout_s=120.0),
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
}
