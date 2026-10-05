#!/usr/bin/env python3
"""Check bucketsim against hand-built traces whose every metadata access is counted below.

    selftest_bucket.py <bucketsim binary>

Record order is capstone-qemu's. Costs: a read-modify-write of a line is one access;
the revoke's root is touched at its start and its end (the second is a hit).
"""
import json
import struct
import subprocess
import sys
import tempfile

READ, WRITE, ALLOC, FREE, RESET, REPEAT, INC, DEC, REG, SLOT, END = range(11)
LDST, LDC, MREV, SPLIT, REVOKE, DELIN, CREATE, SUP, GC, CAPSTORE, UNTAG, CLEAR, DROP = range(13)
NONE = 0xFFFFFFFF
G1, G2, G3, G4 = 0x8000100, 0x8000101, 0x8000200, 0x8000300

failures = []


def check(cond, what):
    if not cond:
        failures.append(what)
        print("FAIL:", what)


def run(sim, recs, variants):
    with tempfile.NamedTemporaryFile(suffix=".bin") as t:
        with open(t.name, "wb") as f:
            f.write(b"CRNTRC02" + b"".join(struct.pack("<IBBH", *x, 0) for x in recs))
            f.write(struct.pack("<IBBH", NONE, END, 0, 0))
        p = subprocess.run([sim, "--max-node", "1000", "--variants", variants, t.name],
                           capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout) if p.stdout else None), p.stderr


def lone(n):
    return [(n, ALLOC, CREATE), (n, WRITE, CREATE)]


def mrev(src, new):          # R src, ALLOC new, W new, W src: new becomes src's parent
    return [(src, READ, MREV), (new, ALLOC, MREV), (new, WRITE, MREV), (src, WRITE, MREV)]


def revoke(root, run):       # R root, (R n, W n)*, W root
    recs = [(root, READ, REVOKE)]
    for n in run:
        recs += [(n, READ, REVOKE), (n, WRITE, REVOKE)]
    return recs + [(root, WRITE, REVOKE)]


def ops(v):
    return {o: x["accesses"] for o, x in v["by_operation"].items() if x["accesses"]}


def main():
    sim = sys.argv[1]

    # Trace 1: inline capacity 2, with and without a 2-entry slot cache.
    t1 = ([(NONE, RESET, CREATE)] + lone(0) + lone(1) + mrev(1, 2)
          + [(G1, SLOT, CAPSTORE), (0, INC, CAPSTORE)]                      # a fresh
          + [(G1, SLOT, CAPSTORE), (0, DEC, CAPSTORE), (0, INC, CAPSTORE)]  # b same node
          + [(G1, SLOT, CAPSTORE), (0, DEC, CAPSTORE), (1, INC, CAPSTORE)]  # c other node
          + [(G2, SLOT, CAPSTORE), (1, INC, CAPSTORE)]                      # d fresh: node 1 inline full
          + [(G3, SLOT, CAPSTORE), (1, INC, CAPSTORE)]                      # e fresh: overflow table
          + [(G2, SLOT, UNTAG), (1, DEC, UNTAG)]                            # f data store
          + [(G3, SLOT, CAPSTORE), (1, DEC, CAPSTORE), (0, INC, CAPSTORE)]  # g other: from the table
          + [(1, REG, REVOKE)] + revoke(2, [1])                             # i revoke 2 -> {1}
          + [(G4, SLOT, CAPSTORE), (1, INC, CAPSTORE)]                      # j store of a revoked cap
          + [(G3, SLOT, CAPSTORE), (0, DEC, CAPSTORE), (1, INC, CAPSTORE)]  # k same, over node 0
          # the emulator's lazy view of G1, which Clover cleared at the revoke: the
          # collector's untag is nothing, and a capability store over it is a fresh store
          + [(G1, SLOT, GC), (1, DEC, GC)]
          + [(G1, SLOT, CAPSTORE), (1, DEC, CAPSTORE), (0, INC, CAPSTORE)]  # l fresh under Clover
          + [(0, READ, LDST)])                                              # one check, the denominator
    rc, r, err = run(sim, t1, "2,0,64;2,2,64")
    check(rc == 0, f"trace 1 exit {rc} {err}")
    a, b = r["variants"]
    check(ops(a) == {"insert": 7, "delete": 5, "tree": 4, "revoke": 5}, f"no cache: {ops(a)}")
    check(a["capability_stores"] == {"same_node": 1, "fresh": 4, "other_node": 2, "of_revoked": 2}
          and a["data_stores_over_capability"] == 2 and a["index_changes"] == 6, f"no cache stores {a['capability_stores']}")
    check(a["collector_untags"] == 0, "an untag of a slot Clover cleared is nothing")
    check(a["overflow"]["inserts"] == 1 and a["overflow"]["nodes"] == 1 and a["overflow"]["max_lines"] == 2,
          f"overflow {a['overflow']}")
    check(a["revokes"]["count"] == 1 and a["revokes"]["tag_clears"] == 1, f"no cache revoke {a['revokes']}")
    check(a["memory_entries_peak"] == 3 and a["errors"] == 0, "no cache peak/errors")
    check(ops(b) == {"flush": 1, "tree": 4, "revoke": 3}, f"cache 2: {ops(b)}")
    check(b["absorbed_by_slot_cache"] == 2 and b["deletes_absorbed"] == 2 and b["evictions"] == 2,
          f"cache 2 absorption {b['absorbed_by_slot_cache']} {b['deletes_absorbed']} {b['evictions']}")
    check(b["revokes"]["tag_clears"] == 1 and b["overflow"]["inserts"] == 0 and b["memory_entries_peak"] == 1
          and b["errors"] == 0, f"cache 2 revoke {b['revokes']} {b['overflow']}")
    check(r["lifetime_checks"] == 1, "trace 1 has one check")

    # Trace 2: the CAM at revoke. A slot listed in memory under node 1 but holding
    # node 0 must not be tag-cleared when 1 is revoked (skip), and a cached slot
    # holding node 0 must be cleared when 0 is revoked (purge).
    t2 = ([(NONE, RESET, CREATE)] + lone(0) + lone(1) + lone(2) + mrev(1, 3)
          + [(G1, SLOT, CAPSTORE), (1, INC, CAPSTORE)]                      # cached G1: 1
          + [(G2, SLOT, CAPSTORE), (2, INC, CAPSTORE)]                      # evicts G1 -> memory (1,G1)
          + [(G1, SLOT, CAPSTORE), (1, DEC, CAPSTORE), (0, INC, CAPSTORE)]  # evicts G2 -> (2,G2); G1 cached c=0 f=1
          + revoke(3, [1])                                                  # memory lists G1 under 1: skip
          + mrev(0, 4) + revoke(4, [0])                                     # G1 holds 0 in the cache: purge
          + [(0, READ, LDST), (9, REPEAT, 0)])                              # 10 checks
    rc, r, err = run(sim, t2, "2,1,64")
    check(rc == 0, f"trace 2 exit {rc} {err}")
    v = r["variants"][0]
    check(ops(v) == {"flush": 2, "tree": 7, "revoke": 6}, f"cam trace ops {ops(v)}")
    check(v["revokes"]["cam_skips"] == 1 and v["revokes"]["cam_purges"] == 1 and v["revokes"]["tag_clears"] == 1
          and v["revokes"]["count"] == 2, f"cam {v['revokes']}")
    check(r["lifetime_checks"] == 10 and v["errors"] == 0, "checks counted, no errors")

    # a 32-byte node record: two nodes per line, so the two tree writes of an mrev of
    # node 0 into node 1 touch one line
    rc, r, _ = run(sim, [(NONE, RESET, CREATE)] + lone(0) + mrev(0, 1) + [(0, READ, LDST)], "2,0,32")
    v = r["variants"][0]
    check(ops(v) == {"tree": 3} and v["misses"][0] == 1, f"32-byte nodes: {ops(v)} misses {v['misses'][0]}")

    # a run that ends at a node: after mrev(1,2), split(2 -> 9), mrev(9 -> 10) the list is
    # [10 d0, 9 d1, 2 d0, 1 d1]; revoking 10 invalidates 9 and relinks 2 (one more access)
    t4 = ([(NONE, RESET, CREATE)] + lone(0) + lone(1) + mrev(1, 2)
          + [(2, READ, SPLIT), (9, ALLOC, SPLIT), (9, WRITE, SPLIT), (2, WRITE, SPLIT)]
          + mrev(9, 10) + revoke(10, [9]) + [(0, READ, LDST)])
    rc, r, err = run(sim, t4, "2,0,64")
    v = r["variants"][0]
    check(rc == 0 and ops(v) == {"tree": 8, "revoke": 4}, f"end-of-run relink: {ops(v)} {err}")

    # id policies: after revoke(2) frees node 1, the next lone node (trace id 9) gets node 1's
    # id under lifo, bitmap and hybrid (its record line was just written: a hit), and its own
    # under traced (a new line: a miss). Policy ids: 0,1,2 handed out; 1 returned; 9 -> 1.
    t5 = ([(NONE, RESET, CREATE)] + lone(0) + lone(1) + mrev(1, 2) + revoke(2, [1]) + lone(9)
          + [(0, READ, LDST)])
    rc, r, err = run(sim, t5, "2,0,64,traced;2,0,64,lifo;2,0,64,bitmap;2,0,64,hybrid")
    check(rc == 0, f"policies exit {rc} {err}")
    for v in r["variants"]:
        ids = v["ids"]
        want_fp = 10 if ids["policy"] == "traced" else 3
        want_reused = 0 if ids["policy"] == "traced" else 1
        check(ids["allocations"] == 4 and ids["frees"] == 1 and ids["footprint"] == want_fp
              and ids["reused"] == want_reused and ids["peak_live"] == 3,
              f"{ids['policy']}: {ids}")
        # the tree write of node 9: traced misses (new line), the others hit node 1's line
        tree_miss = v["by_operation"]["tree"]["misses_64_lines"]
        check(tree_miss == (4 if ids["policy"] == "traced" else 3),
              f"{ids['policy']}: tree misses {tree_miss}")
    # bitmap hands out the lowest free id: free 2 of {0,1,2,3} live, next allocation is 2
    t6 = ([(NONE, RESET, CREATE)] + lone(0) + lone(1) + lone(2) + lone(3) + mrev(2, 4) + revoke(4, [2])
          + lone(9) + [(0, READ, LDST)])
    rc, r, err = run(sim, t6, "2,0,64,lifo;2,0,64,bitmap")
    for v in r["variants"]:
        ids = v["ids"]
        # both reuse id 2 here (lifo: most recent free; bitmap: lowest free); footprint stays 5
        check(ids["footprint"] == 5 and ids["reused"] == 1, f"{ids['policy']}: {ids}")

    print("selftest_bucket:", "PASS" if not failures else f"FAIL ({len(failures)})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
