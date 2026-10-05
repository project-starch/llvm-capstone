#!/usr/bin/env python3
"""Check clovsim against a hand-built trace whose every metadata access is counted below.

    selftest_clover.py <clovsim binary>

Record order is capstone-qemu's. Lines: node records 32 B (2 per line), alias
records 16 B (4 per line), sidecar cells 4 B (16 per line), radix entries 8 B.
"""
import json
import struct
import subprocess
import sys
import tempfile

READ, WRITE, ALLOC, FREE, RESET, REPEAT, INC, DEC, REG, SLOT = range(10)
LDST, LDC, MREV, SPLIT, REVOKE, DELIN, CREATE, SUP, GC, CAPSTORE, UNTAG, CLEAR, DROP = range(13)
NONE = 0xFFFFFFFF
G, G2, G3, G4 = 0x1000, 0x1001, 0x2000, 0x3000

TRACE = ([(NONE, RESET, CREATE), (0, ALLOC, CREATE), (0, WRITE, CREATE)]   # lone 0: tree 1
         + [(0, READ, MREV), (1, ALLOC, MREV), (1, WRITE, MREV), (0, WRITE, MREV)]   # tree 2
         + [(G, SLOT, CAPSTORE), (0, INC, CAPSTORE)]                       # fresh: 1 + 6
         + [(G, SLOT, CAPSTORE), (0, DEC, CAPSTORE), (0, INC, CAPSTORE)]   # same node: 0
         + [(G, SLOT, CAPSTORE), (0, DEC, CAPSTORE), (1, INC, CAPSTORE)]   # other: 1 + 7 + 6
         + [(G2, SLOT, CAPSTORE), (0, INC, CAPSTORE)]                      # fresh: 1 + 6
         + [(G2, SLOT, UNTAG), (0, DEC, UNTAG)]                            # data over it: 7
         + [(G3, SLOT, CAPSTORE), (0, INC, CAPSTORE)]                      # fresh: 1 + 6
         + [(0, READ, LDST), (5, REPEAT, 0)]                               # 6 checks
         + [(1, REG, REVOKE), (2, REPEAT, 0), (1, READ, REVOKE), (0, READ, REVOKE), (0, WRITE, REVOKE),
            (1, WRITE, REVOKE)]                                            # revoke 1: 7
         + [(G4, SLOT, CAPSTORE), (0, INC, CAPSTORE)]                      # node 0 revoked: stale
         + [(G, SLOT, UNTAG), (1, DEC, UNTAG)])                            # data over G: 7

failures = []


def check(cond, what):
    if not cond:
        failures.append(what)
        print("FAIL:", what)


def run(sim, recs):
    with tempfile.NamedTemporaryFile(suffix=".bin") as t:
        with open(t.name, "wb") as f:
            f.write(b"CRNTRC02" + b"".join(struct.pack("<IBBH", *x, 0) for x in recs))
            f.write(struct.pack("<IBBH", 0xFFFFFFFF, 10, 0, 0))
        p = subprocess.run([sim, "--max-node", "1000", t.name], capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout) if p.stdout else None), p.stderr


def main():
    sim = sys.argv[1]
    rc, r, err = run(sim, TRACE)
    check(rc == 0, f"exit {rc} {err}")
    c, k = r["clover"], r["capstone"]
    ops = {o: v["accesses"] for o, v in c["by_operation"].items()}
    check(ops == {"store_register": 28, "store_unregister": 7, "data_store_unregister": 14,
                  "revoke": 7, "tree": 3}, f"clover operations {ops}")
    check(c["accesses"] == 59, f"clover accesses {c['accesses']}")
    check(c["misses"] == [7] * 7, f"clover misses {c['misses']}")
    check(c["capability_stores"] == {"index_unchanged_same_node": 1, "index_updated": 4,
                                     "of_a_revoked_capability": 1}, f"stores {c['capability_stores']}")
    check(c["data_stores_unregistering"] == 2, "data stores unregistering")
    check(c["revokes"]["count"] == 1 and c["revokes"]["tag_clears"] == 1 and c["revokes"]["accesses_max"] == 7,
          f"revokes {c['revokes']}")
    check((c["alias_records_peak"], c["alias_records_end"], c["sidecar_frames_peak"]) == (2, 0, 2),
          f"pool/frames {c['alias_records_peak']} {c['alias_records_end']} {c['sidecar_frames_peak']}")
    check(r["lifetime_checks"] == 6, f"checks {r['lifetime_checks']}")
    check(k["accesses"] == 16 and k["misses"] == [1] * 7, f"capstone {k['accesses']} {k['misses']}")

    rc, _, _ = run(sim, TRACE + [(0, FREE, GC), (0, ALLOC, CREATE)])
    check(rc == 0, "reusing a revoked node's id with no records left is fine")
    rc, _, _ = run(sim, TRACE[:-2] + [(1, FREE, GC), (1, ALLOC, CREATE)])
    check(rc == 3, "reusing an id whose records are still indexed must exit 3")
    # a run that ends at a node: [10 d0, 9 d1, 2 d0, 1 d1] after mrev(1,2), split(2->9),
    # mrev(9->10); revoking 10 invalidates 9 (R, W) and relinks 2: R root, R 9, W 9, W 2, W root
    t4 = ([(NONE, RESET, CREATE), (0, ALLOC, CREATE), (0, WRITE, CREATE), (1, ALLOC, CREATE), (1, WRITE, CREATE),
           (1, READ, MREV), (2, ALLOC, MREV), (2, WRITE, MREV), (1, WRITE, MREV),
           (2, READ, SPLIT), (9, ALLOC, SPLIT), (9, WRITE, SPLIT), (2, WRITE, SPLIT),
           (9, READ, MREV), (10, ALLOC, MREV), (10, WRITE, MREV), (9, WRITE, MREV),
           (10, READ, REVOKE), (9, READ, REVOKE), (9, WRITE, REVOKE), (2, READ, REVOKE), (10, WRITE, REVOKE),
           (2, WRITE, REVOKE), (0, READ, LDST)])
    rc, r, err = run(sim, t4)
    check(rc == 0 and r["clover"]["by_operation"]["revoke"]["accesses"] == 5
          and r["clover"]["by_operation"]["tree"]["accesses"] == 8,
          f"end-of-run relink: {r and r['clover']['by_operation']} {err}")

    print("selftest_clover:", "PASS" if not failures else f"FAIL ({len(failures)})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
