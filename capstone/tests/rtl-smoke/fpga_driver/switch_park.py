"""Verified, bounded park of the virtual switches at 0 (2026-11-14).

WHY. sw[0] hands the console TX pin to the tracer and sw[1] arms a trace dump (board-run skill), so a board left at an
odd switch value boots normally and transmits NOTHING: every later boot reads as "no UART byte", whatever its image.
On 2026-11-14 board 2 sat at 217 from 21:06 to 22:00 and cost four cold boots across two images. The baked runner's
resting-aperture read had died on a switch_state timeout, and its `finally: setsw(0)` timed out the same way, so the
park never happened and nothing said so.

WHAT. park_switches(console) brings every switch to 0 and accepts only a FRESH switch_state (newer than its own emit)
reading all zero:
  1. switch_reset_all, which is idempotent, so it may be retried;
  2. if that is not confirmed, one switch_toggle per set bit. A toggle whose confirmation does not arrive is NEVER
     re-toggled blind: if the toggle landed and only its event was late, a second toggle would set the bit again. The
     park waits for any fresh switch_state, and if none arrives, or the fresh one still shows the bit set, it FAILS.
Bounded by `budget` seconds in all. A failure prints a must-fire "[fpga-switch] PARK FAILED states=[...]" line and
raises SwitchParkFailed. A board already at 0 costs no emit at all. No switch_state ever seen is a FAILURE, never zero.
"""
import sys
import time

from . import config as C
from .fpga_console import ActionTimeout


class SwitchParkFailed(RuntimeError):
    """The switches could not be confirmed at 0: a boot now would transmit nothing. INFRA, not a reading."""


def _say(msg):
    print(msg, file=sys.stderr, flush=True)


def _states(data):
    s = (data or {}).get(C.SWITCH_STATE_KEY) if isinstance(data, dict) else None
    return list(s) if isinstance(s, list) else None


def _all_zero(data):
    s = _states(data)
    return bool(s) and not any(s)


def park_switches(console, budget: float = 60.0, first_wait: float = 20.0):
    """Park every switch at 0 and confirm it; return the confirmed states. Raise SwitchParkFailed otherwise."""
    deadline = time.monotonic() + budget
    left = lambda cap: max(0.1, min(cap, deadline - time.monotonic()))
    ev = C.SWITCH_STATE_EVENT

    cur = _states(console.latest(ev))
    if cur is None:   # nothing pushed yet: wait for the connect burst, never assume zero
        try:
            console.wait_event(ev, timeout=left(first_wait))
        except ActionTimeout:
            pass
        cur = _states(console.latest(ev))
    if cur is None:
        _say("[fpga-switch] PARK FAILED states=None: no switch_state event received")
        raise SwitchParkFailed("no switch_state event received")
    if not any(cur):
        return cur
    _say(f"[fpga-switch] found states={cur}; parking at 0")

    for attempt in (1, 2):
        if time.monotonic() >= deadline:
            break
        mark = console.now()
        console._emit("switch_reset_all")
        try:
            got = console.wait_event(ev, _all_zero, timeout=left(15.0), since=mark)
            _say(f"[fpga-switch] parked by switch_reset_all (attempt {attempt}): states={_states(got)}")
            return _states(got)
        except ActionTimeout:
            _say(f"[fpga-switch] switch_reset_all attempt {attempt} not confirmed; states={_states(console.latest(ev))}")

    for bit in range(len(cur)):
        s = _states(console.latest(ev)) or cur
        if not s[bit]:
            continue
        if time.monotonic() >= deadline:
            break
        mark = console.now()
        console._emit("switch_toggle", index=bit)
        try:
            console.wait_event(ev, lambda d, b=bit: (_states(d) or [1] * (b + 1))[b] == 0, timeout=left(15.0),
                               since=mark)
            continue
        except ActionTimeout:
            pass
        try:   # late confirmation: accept any fresh state, never toggle again on a guess
            late = console.wait_event(ev, timeout=left(20.0), since=mark)
        except ActionTimeout:
            late = None
        ls = _states(late)
        if ls is None or ls[bit]:
            _say(f"[fpga-switch] PARK FAILED states={_states(console.latest(ev))}: bit {bit} toggle not confirmed "
                 f"(fresh state {ls}); not re-toggling blind")
            raise SwitchParkFailed(f"bit {bit} toggle not confirmed")

    final = _states(console.latest(ev))
    if final and not any(final):
        _say(f"[fpga-switch] parked by per-bit toggles: states={final}")
        return final
    _say(f"[fpga-switch] PARK FAILED states={final}: budget {budget:.0f} s spent")
    raise SwitchParkFailed(f"switches not at 0: {final}")
