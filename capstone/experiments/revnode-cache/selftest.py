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

SITES = ["ldst", "ldc", "mrev", "split", "revoke", "delin", "create", "supervisor", "gc",
         "mem_capstore", "mem_untag", "mem_clear", "drop",
         "rc_reg_inc", "rc_reg_dec", "rc_mem_inc", "rc_mem_dec", "rc_same", "rc_free",
         "rc_ld_inc", "rc_ld_dec", "rc_sweep_dec", "rc_sweep_free", "rc_move", "rc_call"]
READ, WRITE, ALLOC, FREE, RESET = range(5)
NONE = 0xFFFFFFFF


REPEAT = 5


def write_trace(path, recs, rle=False, end=True):
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
        if end:
            f.write(struct.pack("<IBBH", 0xFFFFFFFF, 10, 0, 0))


def run(sim, recs, *args, rle=False):
    with tempfile.NamedTemporaryFile(suffix=".bin") as t:
        write_trace(t.name, recs, rle)
        out = subprocess.run([sim, *args, t.name], check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def cache(result, entries, ways):
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
    recs = [(n, READ, 0) for n in [0, 16, 0, 16, 0, 16]]
    c = cache(run(sim, recs), 16, 1)
    check(c["misses"] == 6, f"direct-mapped conflict 0/16 in 16 entries: {c['misses']} misses, want 6")
    c = cache(run(sim, recs), 16, 4)
    check(c["misses"] == 2, f"4-way 0/16 in 16 entries: {c['misses']} misses, want 2")
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
            for entries, ways in ([(e, "full") for e in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]]
                                  + [(e, w) for e in [16, 64, 256, 1024, 4096] for w in [1, 4, 8]]):
                if True:
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
    for entries in [8192, 16384, 32768, 65536]:
        for ways in ["full"]:
            want = reference(big, entries, ways, 1, [])
            check(cache(rb, entries, ways)["by_site"]["ldst"] == want["ldst"],
                  f"large {entries}/{ways} vs reference")

    # past the largest size: evictions from the 65536-entry list, every boundary crossed
    huge = []
    for _ in range(250000):
        r = rng.random()
        huge.append((rng.randrange(40) if r < 0.5 else rng.randrange(3000) if r < 0.7
                     else rng.randrange(90000), READ, rng.randrange(2)))
    huge += [(i % 70000, READ, 0) for i in range(140000)]      # cyclic sweep larger than 65536
    rh = run(sim, huge)
    for entries in [8, 64, 1024, 8192, 32768, 65536]:
        want = reference(huge, entries, "full", 1, [])
        got = cache(rh, entries, "full")["by_site"]
        check(all(list(got[s]) == want[s] for s in SITES), f"huge {entries}/full: {got['ldst']} != {want['ldst']}")

    # alias records are not node accesses: a trace with them interleaved (and repeated)
    # gives the same cache results as without them
    base = [(rng.randrange(300), READ, 0) for _ in range(5000)]
    mixed = []
    for rec in base:
        mixed.append(rec)
        if rng.random() < 0.3:
            mixed += [(rng.randrange(300), 6 + rng.randrange(3), 9 + rng.randrange(3))] * rng.randrange(1, 4)
    ra, rm = run(sim, base), run(sim, mixed, rle=True)
    check(ra["caches"] == rm["caches"] and rm["alias_records"] == len(mixed) - len(base),
          f"alias records: {rm['alias_records']} counted, caches equal {ra['caches'] == rm['caches']}")

    # a trace without END is truncated: an error, unless --no-end
    with tempfile.NamedTemporaryFile(suffix=".bin") as t:
        write_trace(t.name, [(1, READ, 0)], end=False)
        check(subprocess.run([sim, t.name], capture_output=True).returncode != 0, "no END must fail")
        check(subprocess.run([sim, "--no-end", t.name], capture_output=True).returncode == 0, "--no-end accepts it")
    with tempfile.NamedTemporaryFile(suffix=".bin") as t:
        write_trace(t.name, [(1, READ, 0)])
        with open(t.name, "ab") as f:
            f.write(struct.pack("<IBBH", 1, READ, 0, 0))
        check(subprocess.run([sim, t.name], capture_output=True).returncode != 0, "a record after END must fail")

    # id policies: lone 0, lone 1, mrev(1 -> 2), revoke 2 (frees 1), lone 9. Under lifo the
    # new node takes id 1 and its creation write hits the line the revoke just wrote; under
    # traced it is a new line. Node accesses in capstone-qemu's order.
    def lone(n):
        return [(n, ALLOC, 6), (n, WRITE, 6)]
    t = ([(NONE, RESET, 6)] + lone(0) + lone(1)
         + [(1, READ, 2), (2, ALLOC, 2), (2, WRITE, 2), (1, WRITE, 2)]
         + [(2, READ, 4), (1, READ, 4), (1, WRITE, 4), (2, WRITE, 4)]
         + lone(9) + [(0, READ, 0)])
    for policy, want_fp, want_reused in [("traced", 10, 0), ("lifo", 3, 1), ("bitmap", 3, 1), ("hybrid", 3, 1), ("chunk", 3, 1)]:
        r = run(sim, t, "--nodes-per-line", "1", "--ids", policy)
        ids = r["ids"]
        check(ids["policy"] == policy and ids["allocations"] == 4 and ids["frees"] == 1
              and ids["footprint"] == want_fp and ids["reused"] == want_reused, f"{policy}: {ids}")
        # the fourth allocation (lone 9) is the one that reuses; live/footprint at the four
        # allocations: traced 1/1, 2/2, 3/3, 3/10; a reuse policy 1/1, 2/2, 3/3, 3/3
        check(ids["first_reuse_allocation"] == (0 if policy == "traced" else 4)
              and ids["density_samples"] == 4
              and abs(ids["density_mean"] - ((3 + 0.3) / 4 if policy == "traced" else 1.0)) < 1e-6
              and abs(ids["density_min"] - (0.3 if policy == "traced" else 1.0)) < 1e-6, f"{policy}: {ids}")
        create = cache(r, 16, "full")["by_site"]["create"]      # [hits, misses] of the lone nodes
        check(create == ([3, 3] if policy == "traced" else [4, 2]), f"{policy}: create hits/misses {create}")
    # the emulator itself reusing id 1 after the revoke: traced counts it (positive control
    # of the reuse detector), the policies treat it as any new lifetime
    t_emu = t[:-3] + lone(1) + [(0, READ, 0)]
    for policy in ("traced", "lifo", "bitmap", "hybrid", "chunk"):
        ids = run(sim, t_emu, "--nodes-per-line", "1", "--ids", policy)["ids"]
        check(ids["allocations"] == 4 and ids["reused"] == 1 and ids["first_reuse_allocation"] == 4
              and ids["footprint"] == 3, f"emulator reuse, {policy}: {ids}")

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

    # 4. reference-count records (format CRNTRC03): one WRITE per update at the rc_* sites,
    # an rc_same pair as two records, a FREE at rc_free (the free-list write, an access of
    # the record the decrement just wrote); --exclude leaves them out.
    rc = [(1, WRITE, 13), (1, WRITE, 17), (1, WRITE, 17), (2, WRITE, 15), (2, FREE, 18), (1, READ, 0),
          (3, WRITE, 19), (3, WRITE, 21), (3, FREE, 22)]
    with tempfile.NamedTemporaryFile(suffix=".bin") as t:
        with open(t.name, "wb") as f:
            # format 04: the INSN pair (low, high 32 bits of the instruction count) before END
            f.write(b"CRNTRC04" + b"".join(struct.pack("<IBBH", *x, 0) for x in rc)
                    + struct.pack("<IBBH", 0x89ABCDEF, 11, 0, 0) + struct.pack("<IBBH", 0x12, 11, 1, 0)
                    + struct.pack("<IBBH", 0xFFFFFFFF, 10, 0, 0))
        r = json.loads(subprocess.run([sim, t.name], check=True, capture_output=True, text=True).stdout)
        x = json.loads(subprocess.run([sim, "--exclude", "rc_reg_inc,rc_mem_inc,rc_same,rc_free", t.name],
                                      check=True, capture_output=True, text=True).stdout)
    w = r["accesses"]["write"]
    check(w["rc_reg_inc"] == 1 and w["rc_same"] == 2 and w["rc_mem_inc"] == 1 and w["rc_ld_inc"] == 1
          and w["rc_sweep_dec"] == 1 and r["accesses"]["free"]["rc_free"] == 1
          and r["accesses"]["free"]["rc_sweep_free"] == 1, f"rc sites counted {w}")
    check(r["domain_instructions"] == 0x1289ABCDEF, f"instruction count {r['domain_instructions']:#x}")
    check(cache(r, 16, "full")["misses"] == 3 and cache(x, 16, "full")["misses"] == 2
          and x["excluded_sites"] == ["rc_reg_inc", "rc_mem_inc", "rc_same", "rc_free"],
          f"rc exclusion: {cache(r, 16, 'full')['misses']} / {cache(x, 16, 'full')['misses']} misses")
    # the other readers accept the INSN record and report it
    check(r["logical_records"] == len(rc), f"INSN records are not accesses: {r['logical_records']}")
    # 5. the teardown: everything from the last ALLOC record on is reported apart (two
    #    allocations, then two decrements and a free that miss a 16-line cache)
    td = ([(NONE, RESET, 6), (1, ALLOC, 6), (1, WRITE, 6), (2, ALLOC, 6), (2, WRITE, 6), (1, READ, 0), (2, READ, 0),
           (5, WRITE, 16), (6, WRITE, 16), (2, FREE, 18)])
    a = run(sim, td)["after_last_allocation"]
    fc = [c for c in a["full_caches"] if c["entries"] == 16][0]
    check(a["logical_records"] == 7 and a["accesses"]["write"]["rc_mem_dec"] == 2 and a["accesses"]["free"]["rc_free"] == 1
          and fc["misses"] == 3 and fc["misses_by_site"]["rc_mem_dec"] == 2,
          f"teardown split: {a['logical_records']} records, {fc['misses']} misses, {fc['misses_by_site']['rc_mem_dec']} at rc_mem_dec")

    # 6. write-backs. Nine lines written: the 8-line cache dropped line 1 dirty (counted at
    #    its re-read), then holds 2..9 dirty and drops line 2 dirty at the end; the 16-line
    #    cache keeps all nine dirty. Reads alone dirty nothing. Direct-mapped 16 lines:
    #    lines 0 and 16 collide, each eviction writes back a dirty line.
    wb = [(i, WRITE, 0) for i in range(1, 10)] + [(1, READ, 0)]
    r = run(sim, wb, "--nodes-per-line", "1")
    c8, c16 = cache(r, 8, "full"), cache(r, 16, "full")
    check((c8["writebacks"], c8["dirty_at_end"], c16["writebacks"], c16["dirty_at_end"]) == (2, 7, 0, 9),
          f"write-backs full: {c8['writebacks']}/{c8['dirty_at_end']} and {c16['writebacks']}/{c16['dirty_at_end']}, want 2/7 and 0/9")
    r = run(sim, [(i, READ, 0) for i in range(1, 10)], "--nodes-per-line", "1")
    check(cache(r, 8, "full")["writebacks"] == 0 and cache(r, 8, "full")["dirty_at_end"] == 0, "reads dirty nothing")
    r = run(sim, [(0, WRITE, 0), (16, WRITE, 0), (0, READ, 0)], "--nodes-per-line", "1")
    dm = [c for c in r["caches"] if c["entries"] == 16 and c["ways"] == 1][0]
    check(dm["writebacks"] == 2 and dm["misses"] == 3, f"write-backs direct-mapped: {dm['writebacks']}, want 2")
    check(any(c["entries"] == 128 and c["ways"] == 2 for c in r["caches"]), "the paper's 8 KB 2-way cache is simulated")

    print("selftest:", "PASS" if not failures else f"FAIL ({len(failures)})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
