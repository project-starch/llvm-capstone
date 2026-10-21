#!/usr/bin/env python3
"""order.py [QEMU_LOCK_PY]: the lock-ORDER control for qemu_lock.py, without QEMU (2026-10-21).

A domain session that QUEUES for its domain must not hold the global lock while it waits: otherwise every queued session
is a reader "already inside", and an exclusive caller is granted only after the whole queue has run (an un-migrated
exclusive, which bypasses the turnstile, can starve outright). Scenario, on private lock files:
  D1  domain A, holds the lock 6 s;
  Q   domain A, arrives 1 s later and queues behind D1 for the domain;
  X   exclusive (through the turnstile), arrives 2 s after D1.
Required: X is granted when D1 ends and BEFORE Q. The pre-fix module (reader takes global SH, then waits for its domain)
grants Q first and must FAIL here. Two regressions are checked alongside, so the reorder cannot buy that at their cost:
  parallel:  domain A and domain B sessions, both 3 s, must overlap;
  turnstile: D1 (A, 6 s) runs, X arrives at 2 s, D2 (B) at 3 s: X must be granted when D1 ends and before D2.
Exit 0 = pass, 1 = the order is wrong, 2 = no data (a process did not report)."""
import importlib.util
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

MOD = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / 'runtime/virtual/qemu_lock.py')


def worker(tag, domain, delay, hold, t0, out):
    spec = importlib.util.spec_from_file_location('qemu_lock_under_test', MOD)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    if domain: os.environ['CAPSTONE_QEMU_DOMAIN_LOCK'] = domain
    else: os.environ.pop('CAPSTONE_QEMU_DOMAIN_LOCK', None)
    time.sleep(max(0, t0 + delay - time.time()))
    with open(os.devnull, 'w') as null:
        sys.stdout = null
        with m.qemu_lock():
            g = time.time(); time.sleep(hold); e = time.time()
    out.put((tag, g - t0, e - t0))


def scenario(name, specs):
    """specs: (tag, domain letter or '', delay, hold). Returns {tag: (granted, ended)} relative to the start, or None."""
    with tempfile.TemporaryDirectory() as d:
        os.environ['CAPSTONE_QEMU_LOCK'] = f'{d}/global.lock'
        t0 = time.time() + 0.5; q = mp.Queue()
        ps = [mp.Process(target=worker, args=(tag, f'{d}/domain-{dom}.lock' if dom else '', delay, hold, t0, q))
              for tag, dom, delay, hold in specs]
        for p in ps: p.start()
        for p in ps: p.join(60)
        r = {}
        while not q.empty():
            tag, g, e = q.get(); r[tag] = (g, e)
    print(f"{name}:")
    if set(r) != {s[0] for s in specs}:
        print(f"  NO DATA: reported {sorted(r)} of {[s[0] for s in specs]}"); return None
    for tag, *_ in specs: print(f"  {tag}: granted {r[tag][0]:.2f} s, ended {r[tag][1]:.2f} s")
    return r


def main():
    print(f"module {MOD}")
    res = {}
    r = scenario('queued (the defect)', (('D1', 'A', 0.0, 6.0), ('Q', 'A', 1.0, 1.0), ('X', '', 2.0, 1.0)))
    res['queued'] = r and r['X'][0] >= r['D1'][1] - 0.2 and r['X'][0] < r['Q'][0]
    r = scenario('parallel', (('A1', 'A', 0.0, 3.0), ('B1', 'B', 0.0, 3.0)))
    res['parallel'] = r and r['A1'][0] < r['B1'][1] and r['B1'][0] < r['A1'][1]
    r = scenario('turnstile', (('D1', 'A', 0.0, 6.0), ('X', '', 2.0, 1.0), ('D2', 'B', 3.0, 1.0)))
    res['turnstile'] = r and r['X'][0] >= r['D1'][1] - 0.2 and r['X'][0] < r['D2'][0]
    for k, v in res.items(): print(f"{k}: {'NO DATA' if v is None else 'PASS' if v else 'FAIL'}")
    if any(v is None for v in res.values()): return 2
    return 0 if all(res.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
