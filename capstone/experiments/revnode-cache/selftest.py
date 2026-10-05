#!/usr/bin/env python3
"""Check cachesim against an independent reference model and hand-derived cases.

    selftest.py <cachesim binary>

1. Hand-derived: a cyclic sweep over K nodes under LRU misses K times and then
   hits every access when K <= entries, and misses EVERY access when K > entries
   (fully associative). A direct-mapped cache with two nodes that collide
   misses every access. A RESET empties the caches.
2. Reference: random traces (several locality shapes, all kinds and sites,
   nodes-per-line 1 and 4, an excluded site) replayed through a plain-Python
   LRU model; every cache's hits and misses per site must match exactly.
3. Walk detector: a synthetic revoke with three walked nodes is reported as one
   walk of length 3.

Exits 0 only if every check passes; prints each failure.
"""
import json
import random
import struct
import subprocess
import sys
import tempfile
from collections import OrderedDict

SITES = ["ldst", "ldc", "mrev", "split", "revoke", "delin", "create", "supervisor", "gc"]
READ, WRITE, ALLOC, FREE, RESET = range(5)
NONE = 0xFFFFFFFF


REPEAT = 5


def write_trace(path, recs, rle=False):
    """rle=True writes CRNTRC02 the way capstone-qemu does: a run of identical
    records once, then a REPEAT carrying the number of further copies."""
    with open(path, "wb") as f:
        f.write(b"CRNTRC02" if rle else b"CRNTRC01")
        last, reps = None, 0
        for rec in recs:
            if rle and rec == last and rec[1] != RESET:
                reps += 1
                continue
            if reps:
                f.write(struct.pack("<IBBH", reps, REPEAT, 0, 0))
                reps = 0
            f.write(struct.pack("<IBBH", *rec, 0))
            last = rec
        if reps:
            f.write(struct.pack("<IBBH", reps, REPEAT, 0, 0))


def run(sim, recs, *args, rle=False):
    with tempfile.NamedTemporaryFile(suffix=".bin") as t:
        write_trace(t.name, recs, rle)
        out = subprocess.run([sim, *args, t.name], check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def cache(result, entries, ways):
    if ways != "full" and ways >= entries:
        ways = "full"
    for c in result["caches"]:
        if c["entries"] == entries and (c["ways"] == ways or (ways == "full" and c["ways"] == "full")):
            return c
    raise KeyError((entries, ways))


def reference(recs, entries, ways, npl, excluded):
    sets = 1 if ways == "full" else entries // ways
    nways = entries if ways == "full" else ways
    lru = [OrderedDict() for _ in range(sets)]
    by_site = {s: [0, 0] for s in SITES}
    for node, kind, site in recs:
        if kind == RESET:
            lru = [OrderedDict() for _ in range(sets)]
            continue
        if SITES[site] in excluded or node == NONE:
            continue
        line = node // npl
        s = lru[line % sets]
        if line in s:
            s.move_to_end(line)
            by_site[SITES[site]][0] += 1
        else:
            if len(s) >= nways:
                s.popitem(last=False)
            s[line] = True
            by_site[SITES[site]][1] += 1
    return by_site


failures = []


def check(cond, what):
    if not cond:
        failures.append(what)
        print("FAIL:", what)


def main():
    sim = sys.argv[1]

    # 1. hand-derived
    for k, entries in [(8, 8), (9, 8), (64, 64), (65, 64)]:
        recs = [(i % k, READ, 0) for i in range(k * 10)]
        c = cache(run(sim, recs), entries, "full")
        want_miss = k if k <= entries else k * 10
        check(c["misses"] == want_miss and c["hits"] == k * 10 - want_miss,
              f"cyclic K={k} full/{entries}: got {c['hits']} hits {c['misses']} misses, want {k*10-want_miss}/{want_miss}")
    recs = [(n, READ, 0) for n in [0, 8, 0, 8, 0, 8]]
    c = cache(run(sim, recs), 8, 1)
    check(c["misses"] == 6, f"direct-mapped conflict 0/8 in 8 entries: {c['misses']} misses, want 6")
    c = cache(run(sim, recs), 8, 2)
    check(c["misses"] == 2, f"2-way 0/8 in 8 entries: {c['misses']} misses, want 2")
    recs = [(1, READ, 0), (1, READ, 0), (NONE, RESET, 6), (1, READ, 0)]
    c = cache(run(sim, recs), 8, "full")
    check(c["misses"] == 2 and c["hits"] == 1, f"reset: {c['hits']}/{c['misses']}, want 1/2")
    r = run(sim, [(NONE, READ, 0), (3, READ, 1)])
    check(r["checks_without_node"]["ldst"] == 1 and r["accesses"]["read"]["ldc"] == 1,
          "no-node check counted separately")

    # 2. random traces against the reference
    rng = random.Random(1)
    shapes = {
        "uniform-100": lambda: rng.randrange(100),
        "uniform-5000": lambda: rng.randrange(5000),
        "hot-cold": lambda: rng.randrange(10) if rng.random() < 0.8 else rng.randrange(3000),
    }
    for name, gen in shapes.items():
        recs = []
        for _ in range(20000):
            r = rng.random()
            if r < 0.002:
                recs.append((NONE, RESET, 6))
            elif r < 0.01:
                recs.append((NONE, READ, 0))
            elif r < 0.3 and recs:
                recs.extend([recs[-1]] * rng.randrange(1, 5))     # runs, for the RLE format
            else:
                recs.append((gen(), rng.choice([READ, READ, READ, WRITE, ALLOC, FREE]), rng.randrange(9)))
        for npl, excl in [(1, []), (4, []), (1, ["gc", "supervisor"])]:
            args = ["--nodes-per-line", str(npl)] + (["--exclude", ",".join(excl)] if excl else [])
            res = run(sim, recs, *args)
            res_rle = run(sim, recs, *args, rle=True)
            check(res_rle["records"] < res["records"] and res_rle["logical_records"] == len(recs),
                  f"{name}: RLE trace has {res_rle['records']} records for {len(recs)} accesses")
            check(res_rle["caches"] == res["caches"] and res_rle["accesses"] == res["accesses"]
                  and res_rle["checks_without_node"] == res["checks_without_node"],
                  f"{name} npl={npl} excl={excl}: RLE result differs from the plain trace")
            for entries in [8, 64, 512, 4096]:
                for ways in [1, 2, 4, 8, "full"]:
                    want = reference(recs, entries, ways, npl, excl)
                    got = cache(res, entries, ways)["by_site"]
                    check(all(list(got[s]) == want[s] for s in SITES),
                          f"{name} npl={npl} excl={excl} {entries}/{ways}: {got} != {want}")

    # compulsory misses: first touch per line since a reset
    r = run(sim, [(1, READ, 0), (2, READ, 1), (1, READ, 0), (NONE, RESET, 6), (1, READ, 0), (5, WRITE, 2)])
    check(r["compulsory_misses"]["ldst"] == 2 and r["compulsory_misses"]["ldc"] == 1
          and r["compulsory_misses"]["mrev"] == 1, f"compulsory: {r['compulsory_misses']}")
    r = run(sim, [(0, READ, 0), (1, READ, 0), (4, READ, 0)], "--nodes-per-line", "4")
    check(r["compulsory_misses"]["ldst"] == 2, f"compulsory, 4 nodes/line: {r['compulsory_misses']}")
    big = [(rng.randrange(70000), READ, 0) for _ in range(30000)]
    rb = run(sim, big)
    check(cache(rb, 65536, "full")["misses"] == sum(rb["compulsory_misses"].values()),
          "a 65536-entry cache larger than the working set misses only compulsorily")
    for entries in [16384, 65536]:
        for ways in [8, "full"]:
            want = reference(big, entries, ways, 1, [])
            check(cache(rb, entries, ways)["by_site"]["ldst"] == want["ldst"],
                  f"large {entries}/{ways} vs reference")

    # 3. revoke walk
    # In capstone-qemu's order: R root, (R n, W n)*, [R end], W root, [W end].
    walk3 = [(5, READ, 4), (6, READ, 4), (6, WRITE, 4), (7, READ, 4), (7, WRITE, 4),
             (8, READ, 4), (8, WRITE, 4), (9, READ, 4), (5, WRITE, 4), (9, WRITE, 4)]
    walk0_at_tail = [(11, READ, 4), (11, WRITE, 4)]          # nothing below, no end node
    walk1_no_end = [(12, READ, 4), (13, READ, 4), (13, WRITE, 4), (12, WRITE, 4)]
    w = run(sim, walk3 + walk0_at_tail + walk1_no_end + walk3)["revoke_walks"]
    check(w["count"] == 4 and w["nodes_walked"] == 7 and w["max"] == 3
          and w["hist"][:4] == [1, 1, 0, 2],
          f"revoke walks: {w}, want 4 walks of 3, 0, 1, 3")

    print("selftest:", "PASS" if not failures else f"FAIL ({len(failures)})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
