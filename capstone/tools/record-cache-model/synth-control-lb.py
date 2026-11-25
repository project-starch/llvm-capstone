#!/usr/bin/env python3
"""synth-control-lb.py RCM_LB_BINARY [--perturb]: a synthetic v2 trace whose every LB / NC / RC / GC count was computed by
hand, run through rcm-lb, each count asserted. --perturb changes one expectation, and the script must then FAIL
(its positive control). Exit 0 only when every count matches."""
import os, struct, subprocess, sys, tempfile

K = dict(c=0, a=3, r=4, E=10)

def ev(kind, op, slot, gen, live, cnt=1):
    return struct.pack("<QQ", ((K[kind] << 2 | op) << 56) | (cnt << 32) | slot, (live << 32) | gen)

def trace():
    t = []
    t.append(ev("c", 0, 5, 1, 1))                            # 1 miss, probe, install s5 in way 0 (lowest free)
    t.append(ev("c", 0, 5, 1, 1, 3))                         # 2 three hits
    for s in (69, 133, 197, 261):                            # 3 set 5: ways 1, 2, 3 fill, then s261 evicts way 0 (rr 0 -> 1)
        t.append(ev("c", 0, s, 1, 1))
    t.append(ev("c", 0, 5, 1, 1))                            # 4 s5 evicted: miss, victim rr 1 (s69), rr -> 2
    t.append(ev("r", 1, 133, 1, 0))                          # 5 release write: s133 dead in place (way 2 kept)
    t.append(ev("c", 0, 133, 1, 1, 2))                       # 6 two hits on a dead entry: deny_hit_dead 2, no probe
    t.append(ev("a", 1, 133, 2, 1))                          # 7 reissue at generation 2: same index, way 2 reused, live
    t.append(ev("c", 0, 133, 2, 1))                          # 8 hit
    t.append(struct.pack("<QQ", (K["E"] << 2) << 56, 0))      # 9 a namespace edge: LB empties
    t.append(ev("c", 0, 133, 2, 1))                          # 10 miss
    t.append(ev("c", 0, 300, 9, 0, 2))                       # 11 a dead id: one probe, no install, one dead-record deny
    return b"".join(t)

# hand counts
EXPECT = {
    ("LB", "check"): dict(accesses=15, hits=6, misses=8, mem_reads=8, mem_writes=0),
    ("LB", "op"):    dict(accesses=2, hits=0, misses=2, mem_reads=0, mem_writes=2),
    ("NC", "check"): dict(accesses=15, hits=0, misses=15, mem_reads=15, mem_writes=0),
    ("NC", "op"):    dict(accesses=2, hits=0, misses=2, mem_reads=0, mem_writes=2),
    # RC: lines = slot >> 2, set = line & 63. s5, s69, s133, s197, s261 are lines 1, 17, 33, 49, 65 -> sets 1, 17, 33,
    # 49, 1: s261 and s5 share set 1 with 2 ways, so every check after the first of each line hits; s300 is line 75,
    # set 11. Checks: s5 miss, 3 hits, s69 s133 s197 s261 misses, s5 hit (2 ways), s133 x2 hit, s133 hit, s133 hit (RC
    # does not flush on E), s300 miss + 1 hit = 15 accesses, 6 misses, 9 hits. Ops: the release write (hit, dirty)
    # and the reissue write (hit): 2 accesses, 2 hits.
    ("RC", "check"): dict(accesses=15, hits=9, misses=6, mem_reads=6, mem_writes=0),
    ("RC", "op"):    dict(accesses=2, hits=2, misses=0, mem_reads=0, mem_writes=0),
    # GC: set = slot & 127, tag = slot >> 7. s5 (set 5), s69 (set 69), s133 (set 5, tag 1), s197 (set 69, tag 1),
    # s261 (set 5, tag 2), s300 (set 44): 4 ways, no eviction. Checks: misses on first sight of each of s5, s69, s133,
    # s197, s261, s300 = 6; the rest hit = 9. Ops always go to memory: 2 accesses, 2 misses, 2 writes.
    ("GC", "check"): dict(accesses=15, hits=9, misses=6, mem_reads=6, mem_writes=0),
    ("GC", "op"):    dict(accesses=2, hits=0, misses=2, mem_reads=0, mem_writes=2),
}
LBX = dict(probes=8, deny_hit_dead=2, dead_record=1, installs=8, invalidations=1, flushes=1)

def main():
    binary = sys.argv[1]; perturb = "--perturb" in sys.argv
    exp = {k: dict(v) for k, v in EXPECT.items()}
    if perturb: exp[("LB", "check")]["hits"] += 1
    with tempfile.NamedTemporaryFile(suffix=".rt2", delete=False) as f:
        f.write(trace()); path = f.name
    out = subprocess.run([binary, path], capture_output=True, text=True)
    os.unlink(path)
    if out.returncode != 0 or "END" not in out.stdout:
        print("RUN FAILED", out.returncode, out.stderr); return 1
    bad = 0; cols = ["accesses", "hits", "misses", "mem_reads", "mem_writes"]
    for line in out.stdout.splitlines():
        f = line.split()
        if len(f) >= 7 and (f[0], f[1]) in exp:
            got = dict(zip(cols, map(int, f[2:7])))
            for c, v in exp[(f[0], f[1])].items():
                if got[c] != v: print(f"MISMATCH {f[0]} {f[1]} {c}: got {got[c]} expected {v}"); bad += 1
            exp.pop((f[0], f[1]))
        if f and f[0] == "LB" and f[1] == "probes":
            got = {f[i]: int(f[i + 1]) for i in range(1, len(f), 2)}
            for c, v in LBX.items():
                if got.get(c) != v: print(f"MISMATCH LB {c}: got {got.get(c)} expected {v}"); bad += 1
    if exp: print("ROWS NOT PRINTED", sorted(exp)); bad += 1
    print("CONTROL", "PASS" if not bad else f"FAIL ({bad})"); return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
