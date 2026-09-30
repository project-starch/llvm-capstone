#!/usr/bin/env python3
"""Two-sided test of the wedge dump's refusal-record reads (run_sqlite_stages_fpga.py), against a fake console.

The wedge loop sets the switches DIRECTLY for every non-record aperture, then reads the record bytes (204-209) with
settled_halted_read, which walks via set_switch_value and its switch model (_SW_CURRENT). The loop's direct sets
never update that model, so without a reset before each record read the walk lands on the wrong aperture. On
2026-09-29 (Part A boot m1v2-2) "208" was read at 210 and "205" at 207.

  python3 test_refusal_wedge_walk.py    exit 0 = the reset makes every read land, and without it the bug reproduces
"""
import ast, itertools, pathlib, sys, time

SRC = pathlib.Path(__file__).with_name("run_sqlite_stages_fpga.py").read_text()

def load():
    ns = {"time": time, "itertools": itertools, "LED_SETTLE_S": 0.0, "LED_FRESH_TIMEOUT_S": 0.0,
          "DESTRUCTIVE_SWITCHES": frozenset({220}), "_SW_CURRENT": None}
    for node in ast.parse(SRC).body:
        if isinstance(node, ast.FunctionDef) and node.name in ("settled_halted_read", "set_switch_value",
                                                                "safe_switch_bit_order"):
            exec(compile(ast.Module([node], []), "run_sqlite_stages_fpga.py", "exec"), ns)
    return ns

class C:
    LISTEN = {"led_state": "led_state"}

VAL = {a: (a * 37 + 11) & 0xFF for a in range(256)}
VAL.update({0: 0x00, 224: 0x9f, 204: 0x11, 205: 0x5f, 206: 0x00, 207: 0x00, 208: 0x00, 209: 0x00})

class Fake:
    """Pushes led_state on change only, like the console."""
    def __init__(s): s.sw = 0; s.t = 0.0; s.events = []; s.last = {"states": [0] * 8}
    def now(s): s.t += 0.001; return s.t
    def set_switch(s, bit, on):
        s.sw = (s.sw | (1 << bit)) if on else (s.sw & ~(1 << bit))
        v = VAL[s.sw]; st = [(v >> i) & 1 for i in range(8)]
        if st != s.last["states"]:
            s.last = {"states": st}; s.events.append((s.now(), v))
    def wait_event(s, ev, timeout, since):
        if any(t >= since for t, _ in s.events):
            return s.last
        raise TimeoutError()
    def latest(s, ev): return s.last

# the wedge dump's order: trap log, 204, the mepc/tval bytes (direct), then 208, 205, 206, 207, 209
ORDER = [255, 204] + list(range(196, 204)) + [210, 211, 213, 214, 215, 216, 217, 218] + [208, 205, 206, 207, 209]

def wedge_loop(reset):
    ns = load(); f = Fake(); bad = []
    ns["_SW_CURRENT"] = 0
    for sw in ORDER:
        if 204 <= sw <= 209:
            if reset:
                ns["_SW_CURRENT"] = None       # what the runner does before each record read
            v = ns["settled_halted_read"](f, C, sw, settle=0, fresh_timeout=0)
            if f.sw != sw or v != VAL[sw]:
                bad.append((sw, f.sw, v))
        else:
            for bit in range(8):
                f.set_switch(bit, bool(sw & (1 << bit)))
    return bad

def main():
    if "globals()[\"_SW_CURRENT\"] = None" not in SRC:
        print("FAIL: the runner no longer resets the switch model before a record read"); return 1
    fixed, bug = wedge_loop(True), wedge_loop(False)
    print("with the reset:   wrong reads", fixed)
    print("without it:       wrong reads", bug)
    if fixed:
        print("FAIL: a record read lands on the wrong aperture even with the reset"); return 1
    if not bug:
        print("FAIL: the test cannot reproduce the bug, so it proves nothing about the fix"); return 1
    print("PASS"); return 0

if __name__ == "__main__":
    sys.exit(main())
