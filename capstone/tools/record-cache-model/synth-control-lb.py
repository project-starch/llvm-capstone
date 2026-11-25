#!/usr/bin/env python3
"""synth-control-lb.py RCM_LB_BINARY [--perturb]: a synthetic v2 trace whose every LB / NC / RC / GC count was computed by
hand, run through rcm-lb, each count asserted. --perturb changes one expectation, and the script must then FAIL
(its positive control). Exit 0 only when every count matches."""
import os, struct, subprocess, sys, tempfile

K = dict(c=0, a=3, r=4, T=7, I=8, PI=9, E=10)

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
    t.append(struct.pack("<QQ", (K["E"] << 2) << 56, 0))      # 7b an edge with the key unchanged (word 1 bit 0 = 0): no flush
    t.append(ev("c", 0, 133, 2, 1))                          # 8 hit (a flush at 7b would make this a miss)
    t.append(struct.pack("<QQ", (K["E"] << 2 | 1) << 56, 1))  # 9 a namespace edge, key changed (srevroot write): LB empties
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
EOPS = [1, 1, 0, 0]   # E by op: 7b is op 0, 9 is op 1
LBX = dict(probes=8, deny_hit_dead=2, dead_record=1, installs=8, invalidations=1, flushes=1)

# malformed traces rcm-lb must refuse with exit 2 (each after one valid record event, so only the bad word decides)
BAD = {
    "E word1 2":   struct.pack("<QQ", (K["E"] << 2 | 1) << 56, 2),
    "T word1 1":   struct.pack("<QQ", (K["T"] << 2) << 56, 1),
    "I word1 1":   struct.pack("<QQ", (K["I"] << 2) << 56 | 5, 1),
    "PI word1 1":  struct.pack("<QQ", (K["PI"] << 2) << 56 | 5, 1),
    "c word1 b33": struct.pack("<QQ", ((K["c"] << 2) << 56) | (1 << 32) | 7, 1 << 33),
    "kind 11":     struct.pack("<QQ", (11 << 2) << 56, 0),
}
OK = {"E word1 1": struct.pack("<QQ", (K["E"] << 2 | 1) << 56, 1)}   # the accepting side of the same check

# self_evict: one stretch installing five slots of set 1 displaces its own first (1); four slots, or five split by a
# check, displace nothing of their own (0)
def wr(s): return ev("a", 1, s, 1, 1)
SELF = {
    "five in one stretch": (b"".join(wr(s) for s in (1, 65, 129, 193, 257)), 1),
    "four in one stretch": (b"".join(wr(s) for s in (1, 65, 129, 193)), 0),
    "five split by a check": (b"".join(wr(s) for s in (1, 65, 129, 193)) + ev("c", 0, 1, 1, 1) + wr(257), 0),
}

# the open phase (b42c8c644: every lifetime op reads the first two free entries, and the read tap fills them live for
# their stored NEXT generation, ex_stage.sv:2628-2629; measured as reads in capstone-ariane slu-walk 4bbdbc6ed):
# allocate s20 {0}, check (hit), release (dead in place), check (deny_hit_dead), the next op's open reads s20 at gen
# 1, then a check of the stale id {0, 20}: with the open read live it MISSES and probes (a dead id: no install); the
# same trace with the open read at live 0 (the twin's shape before nested a1b52f58ab) denies it without a probe.
def openph(live): return (ev("a", 1, 20, 0, 1) + ev("c", 0, 20, 0, 1) + ev("r", 1, 20, 0, 0) + ev("c", 0, 20, 0, 1)
                          + ev("a", 0, 20, 1, live) + ev("c", 0, 20, 0, 0))
OPENPH = {"open read live 1": (openph(1), dict(probes=1, deny_hit_dead=1)),
          "open read live 0": (openph(0), dict(probes=0, deny_hit_dead=2))}

def run_out(binary, body):
    with tempfile.NamedTemporaryFile(suffix=".rt2", delete=False) as f:
        f.write(body); path = f.name
    out = subprocess.run([binary, path], capture_output=True, text=True).stdout
    os.unlink(path); return out

def run_one(binary, body):
    with tempfile.NamedTemporaryFile(suffix=".rt2", delete=False) as f:
        f.write(ev("c", 0, 5, 1, 1) + body); path = f.name
    rc = subprocess.run([binary, path], capture_output=True, text=True).returncode
    os.unlink(path); return rc

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
    bad = 0; seen_e = False; cols = ["accesses", "hits", "misses", "mem_reads", "mem_writes"]
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
        if f[:3] == ["E", "by", "op"]:
            got = list(map(int, f[3:7])); seen_e = True
            if got != EOPS: print(f"MISMATCH E by op: got {got} expected {EOPS}"); bad += 1
    if exp: print("ROWS NOT PRINTED", sorted(exp)); bad += 1
    if not seen_e: print("E BY OP NOT PRINTED"); bad += 1
    for name, body in BAD.items():
        rc = run_one(binary, body)
        if rc != 2: print(f"MALFORMED ACCEPTED {name}: rc {rc}, expected 2"); bad += 1
    for name, (body, want) in SELF.items():
        got = [l.split() for l in run_out(binary, ev("c", 0, 9, 1, 1) + body).splitlines() if l.startswith("LB op_stretches")]
        se = int(got[0][4]) if got else None
        if se != want: print(f"MISMATCH self_evict {name}: got {se} expected {want}"); bad += 1
    for name, (body, want) in OPENPH.items():
        lines = [l.split() for l in run_out(binary, body).splitlines() if l.startswith("LB probes")]
        got = {lines[0][i]: int(lines[0][i + 1]) for i in range(1, len(lines[0]), 2)} if lines else {}
        for c, v in want.items():
            if got.get(c) != v: print(f"MISMATCH open phase {name} {c}: got {got.get(c)} expected {v}"); bad += 1
    for name, body in OK.items():
        rc = run_one(binary, body)
        if rc != 0: print(f"WELL-FORMED REFUSED {name}: rc {rc}, expected 0"); bad += 1
    print("CONTROL", "PASS" if not bad else f"FAIL ({bad})"); return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
