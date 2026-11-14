#!/usr/bin/env python3
"""Offline check of switch_park.park_switches (no board): a fake console on a virtual clock.

switch_toggle is a TOGGLE and its confirmation can be late, so a blind retry can set a bit again. Arms:
 1. 217 at connect, switch_reset_all works: parked by it, no toggle emitted;
 2. 0 at connect: no emit at all (the gate is silent on a clean board);
 3. switch_reset_all unsupported (no event): one toggle each for bits 0,3,4,6,7 only, parked;
 4. as 3, bit 3's confirmation 17 s late (past the 15 s wait, inside the 20 s late wait): ONE toggle for bit 3, parked;
 5. as 3, bit 3 never confirmed: SwitchParkFailed with the must-fire line, ONE toggle for bit 3, no blind second;
 6. no switch_state ever pushed: SwitchParkFailed (no data is a failure, never zero);
 7. a toggle that lands but whose fresh event shows the bit still set: SwitchParkFailed, no second toggle.
Exit 0 when all hold."""
import contextlib
import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from fpga_driver import config as C  # noqa: E402
from fpga_driver import switch_park as SP  # noqa: E402
from fpga_driver.fpga_console import ActionTimeout  # noqa: E402

CLOCK = [0.0]
SP.time.monotonic = lambda: CLOCK[0]


class Fake:
    def __init__(self, states, reset_all=True, delay=None, never=(), stale=(), initial=True):
        self.sw = list(states)
        self.reset_all = reset_all
        self.delay = delay or {}
        self.never = set(never)
        self.stale = set(stale)
        self.emits = []
        self.events = []          # (deliver_ts, data)
        if initial:
            self.events.append((CLOCK[0], {C.SWITCH_STATE_KEY: list(self.sw)}))

    def now(self):
        return CLOCK[0]

    def latest(self, ev):
        got = [d for ts, d in self.events if ts <= CLOCK[0]]
        return got[-1] if got else None

    def _emit(self, key, **kw):
        self.emits.append((key, kw.get("index")))
        if key == "switch_reset_all":
            if self.reset_all:
                self.sw = [0] * len(self.sw)
                self.events.append((CLOCK[0] + 1, {C.SWITCH_STATE_KEY: list(self.sw)}))
        elif key == "switch_toggle":
            b = kw["index"]
            if b in self.stale:   # the server reports the old value as a fresh event (the ambiguous case)
                self.events.append((CLOCK[0] + 1, {C.SWITCH_STATE_KEY: list(self.sw)}))
                return
            self.sw[b] ^= 1
            if b not in self.never:
                self.events.append((CLOCK[0] + self.delay.get(b, 1), {C.SWITCH_STATE_KEY: list(self.sw)}))

    def wait_event(self, ev, predicate=lambda d: True, timeout=60.0, *, since=None):
        end = CLOCK[0] + timeout
        for ts, d in sorted(self.events, key=lambda e: e[0]):
            if ts > end:
                break
            if (since is None or ts >= since) and predicate(d):
                CLOCK[0] = max(CLOCK[0], ts) + 1e-6   # as the real console: now() after a return is strictly later
                return d
        CLOCK[0] = end
        raise ActionTimeout(f"timed out waiting for event {ev!r}")


def run(fake):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            return SP.park_switches(fake), err.getvalue(), None
        except SP.SwitchParkFailed as e:
            return None, err.getvalue(), e


B217 = [1, 0, 0, 1, 1, 0, 1, 1]
fails = []


def check(name, cond, detail):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f": {detail}"))
    if not cond:
        fails.append(name)


f = Fake(B217)
st, out, e = run(f)
check("1 reset_all parks 217", st == [0] * 8 and f.emits == [("switch_reset_all", None)], (st, f.emits, out))

f = Fake([0] * 8)
st, out, e = run(f)
check("2 clean board: no emit", st == [0] * 8 and f.emits == [], (st, f.emits))

f = Fake(B217, reset_all=False)
st, out, e = run(f)
tog = [i for k, i in f.emits if k == "switch_toggle"]
check("3 toggles only the set bits", st == [0] * 8 and tog == [0, 3, 4, 6, 7] and f.sw == [0] * 8, (st, f.emits))

f = Fake(B217, reset_all=False, delay={3: 17})
st, out, e = run(f)
tog = [i for k, i in f.emits if k == "switch_toggle"]
check("4 late confirmation: one toggle for bit 3", st == [0] * 8 and tog.count(3) == 1 and f.sw == [0] * 8,
      (st, f.emits, out))

f = Fake(B217, reset_all=False, never={3})
t0 = CLOCK[0]
st, out, e = run(f)
tog = [i for k, i in f.emits if k == "switch_toggle"]
check("5 never confirmed: fails, no blind second toggle",
      e is not None and "PARK FAILED" in out and tog.count(3) == 1 and CLOCK[0] - t0 <= 60.0 + 1e-6,
      (e, tog, out, CLOCK[0] - t0))

f = Fake(B217, initial=False)
st, out, e = run(f)
check("6 no switch_state: fails, not zero", e is not None and "PARK FAILED states=None" in out and f.emits == [],
      (e, out, f.emits))

f = Fake(B217, reset_all=False, stale={3})
st, out, e = run(f)
tog = [i for k, i in f.emits if k == "switch_toggle"]
check("7 fresh event still set: fails, no second toggle", e is not None and tog.count(3) == 1, (e, tog, out))

sys.exit(1 if fails else 0)
