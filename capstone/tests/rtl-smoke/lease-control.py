#!/usr/bin/env python3
"""The lease client's yield control (2026-10-19; the console lane's recipe, harmless in LEASE_ENFORCE=log).
Takes board N's lease through FpgaConsole (with the local board lock, as a run does), then:
  1. POST /api/lease/queue as "yield-control"  -> a ticket at position 1;
  2. renew                                      -> must be 409 "yield" (the client sets lease_yield);
  3. power(on=True)                             -> must raise LeaseYield, touching nothing (the no-new-cycle rule);
  4. DELETE /api/lease/queue with the ticket    -> left;
  5. renew                                      -> must be 200 (the quiet control);
  6. release.
Leaving the queue BEFORE the release matters: a release with someone queued offers the board to them for 5 minutes.
Exit 0 only when 2, 3 and 5 read as required. No socket connection, no power, load or flash.
Usage: FPGA_BOARD=N python3 lease-control.py"""
import os, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fpga_driver.fpga_console import FpgaConsole, LeaseYield, board_id, board_url

b = board_id()
c = FpgaConsole(board_url(b), logger=lambda m: None)
c._acquire_board_lock()
ok = True
def check(cond, msg):
    global ok
    print(("OK   " if cond else "FAIL ") + msg, flush=True); ok &= bool(cond)
info = c.lease_info()
if info is None:
    print(f"NO DATA: board {b} has no lease API (404)"); sys.exit(2)
print(f"before: holder {(info.get('lease') or {}).get('holder')!r}, queue {len(info.get('queue') or [])}, enforce {info.get('enforce')}")
c.lease_take(purpose="lease-control (yield control)")
check(c._lease_state == "held", f"take: {c._lease_state}")
ticket = None
try:
    r = c._http.post(c._lease_url("/queue"), json={"holder": "yield-control", "purpose": "control"}, timeout=15)
    d = r.json(); ticket = d.get("ticket")
    check(r.ok and ticket and d.get("position") == 1, f"queue join: HTTP {r.status_code}, position {d.get('position')}")
    code = c.lease_renew()
    check(code == "yield" and c.lease_yield, f"renew with a person queued: {code!r}, lease_yield {c.lease_yield}")
    try:
        c.power(True); check(False, "power-on after a yield was NOT refused")
    except LeaseYield:
        check(True, "power-on after a yield refused (LeaseYield), nothing touched")
finally:
    if ticket:
        r = c._http.delete(c._lease_url("/queue"), json={"ticket": ticket}, timeout=15)
        d = r.json() if r.content else {}
        check(r.ok and d.get("left") is True, f"queue leave: HTTP {r.status_code}, left {d.get('left')}")
    c.lease_yield = False
    code = c.lease_renew()
    check(code is None, f"renew with nobody queued: {code!r} (want success)")
    c.lease_release()
info = c.lease_info()
check(not (info.get("lease") or {}).get("holder") and not info.get("offer"), f"after: holder {(info.get('lease') or {}).get('holder')!r}, offer {info.get('offer')!r}, queue {len(info.get('queue') or [])}")
print("LEASE CONTROL PASS" if ok else "LEASE CONTROL FAIL"); sys.exit(0 if ok else 1)
