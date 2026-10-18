#!/usr/bin/env python3
# derive-b5-prediction.py <QEMU serial.log> <out>: boot 5's suite is run.py's without the sparse pair, so its QEMU prediction is
# the compiler lane's run.py serial under arena_pin=1 with those two steps' lines cut (each from the previous step's VM_EXIT to its
# own). board-check3.py then scores the cut serial; the uncut serial is the control (it must read the record's 33/41).
import sys, re
src, out = sys.argv[1], sys.argv[2]
t = open(src, errors="replace").read().replace("\r", "")
L = t.split("\n")
def idx(line):
    k = [i for i, s in enumerate(L) if s.strip() == line]
    assert len(k) == 1, (line, k); return k[0]
# sparse-churn: the lines after heap-threads' exit up to and including its own exit; sparse-stale likewise after vm-padding
a1, b1 = idx("VM_EXIT:heap-threads:1"), idx("VM_EXIT:sparse-churn:0")
a2, b2 = idx("VM_EXIT:vm-padding:139"), idx("VM_EXIT:sparse-stale:139")
assert a1 < b1 < a2 < b2
cut = set(range(a1 + 1, b1 + 1)) | set(range(a2 + 1, b2 + 1))
kept = [s for i, s in enumerate(L) if i not in cut]
removed = [L[i] for i in sorted(cut)]
open(out, "w").write("\n".join(kept))
print(f"removed {len(removed)} lines; stats lines removed {sum(s.startswith('CAPSTONE_VM_STATS ') for s in removed)}; "
      f"fault lines removed {sum('domain fault cause=' in s for s in removed)}: "
      f"{[re.search(r'cause=(\d+)', s).group(1) for s in removed if 'domain fault cause=' in s]}")
