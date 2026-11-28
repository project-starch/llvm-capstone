#!/usr/bin/env python3
"""test_socket_drop.py DRIVER_RS [healthy|drop|lostafter|nofresh] : the 2026-11-26 board-1 failure on the driver's mock console. Connect; kill the engineio write
loop while its state stays 'connected' (what 'packet queue is empty, aborting' leaves); then power(True). Prints the outcome."""
import sys, time
rs = sys.argv[1]; MODE = sys.argv[2] if len(sys.argv) > 2 else "drop"; sys.path.insert(0, rs)
from fpga_driver.fpga_console import FpgaConsole  # noqa: E402
from fpga_driver.test_dryrun import _MockServer  # noqa: E402  (the dry test's own mock-console helper)
mock = _MockServer().__enter__()
try:
    c = FpgaConsole(mock.url, allow_unverified=True); c.connect(timeout=10)
    time.sleep(1.0)
    print("before:", "power", c._current_state("power_state"), "| connected", c.sio.connected, "| eio state", c.sio.eio.state)
    toggles = []
    orig_emit = c._emit
    def spy(key, **kw):
        if key == "power_toggle":
            toggles.append(time.time())
            if MODE == "lostafter" and len(toggles) == 1:
                c.sio.eio.queue.put(None); time.sleep(0.5)   # the loop dies between the live check and this toggle
        return orig_emit(key, **kw)
    c._emit = spy
    if MODE == "nofresh":
        orig_rec = c._record
        c._record = lambda ev, d: None if (ev == "power_state" and getattr(c, "_own_reconnects", 0) > 0) else orig_rec(ev, d)
    if MODE in ("drop", "nofresh"):
        c.sio.eio.queue.put(None)      # the write loop reads None -> empty packet list -> exits; eio.state stays 'connected'
    wl = c.sio.eio.write_loop_task
    for _ in range(50):
        if not wl.is_alive(): break
        time.sleep(0.1)
    print("after the drop: write loop alive", wl.is_alive(), "| eio state", c.sio.eio.state, "| sio.connected", c.sio.connected)
    t0 = time.time()
    try:
        c.power(True, timeout=8.0)
        print(f"RESULT power(True) RETURNED after {time.time()-t0:.1f} s: power_state now {c.latest('power_state')!r}; own reconnects {getattr(c, '_own_reconnects', 'n/a')}; toggles emitted {len(toggles)}")
    except Exception as e:
        print(f"RESULT power(True) RAISED after {time.time()-t0:.1f} s: {type(e).__name__}: {str(e)[:200]}; toggles emitted {len(toggles)}")
    try: c.close()
    except Exception: pass
finally:
    mock.proc.kill(); mock.proc.wait(timeout=5)
