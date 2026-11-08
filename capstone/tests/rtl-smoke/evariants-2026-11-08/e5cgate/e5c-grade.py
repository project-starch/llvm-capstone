#!/usr/bin/env python3
"""e0-grade.py <E0 run dir> [e0-vectors.txt] : E0's 32-image first-boot gate (rtl lane, land/rtl-evariants (e5-native section)).

For each image: E0's silicon vector (<run>/<img>/uart.txt: SR count, SV values) against
  E0sim, C5sim  - e0-vectors.txt (the E0 and C.5a image-sim columns),
  C5sil         - C.5a's own silicon run of the same image (perf/c5a-b2/{r67,hpm,objb,slot-c5a}, perf/r32/run-b2).
Rule, per field k (pre-registered with this file):
  VALUE field  (C5sil[k] == C5sim[k]: silicon reads the simulator's value): E0sil[k] must equal E0sim[k] exactly.
  TIMING field (C5sil[k] != C5sim[k]):
    |E0sim[k] - C5sim[k]| < max(1 %, 8) (unchanged, or a sim delta below the cross-board noise floor):
                                                                |E0sil[k] - C5sil[k]| <= max(3 %, 40);
    E0sim[k] <  C5sim[k] beyond that (the DYN trim speeds it up): E0sil[k] <  C5sil[k];
    E0sim[k] >  C5sim[k] beyond that:                             E0sil[k] >  C5sil[k].
  The count (SR) must equal E0sim's count. An image whose sims are both empty (objb-pcc: it never reaches its
  recorder) must read what C.5a's silicon read (count and values).
Exit 0 = PASS (every image), 1 = FAIL (any image), 2 = NO DATA (a missing image run or reference)."""
import sys, re, pathlib, os
H = pathlib.Path(os.environ.get('CAPSTONE_WORK', f"/mnt/hdd/{os.environ['USER']}/capstone")) / 'perf'
run = pathlib.Path(sys.argv[1])
vf = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(__file__).with_name('e0-vectors.txt')
C5DIR = {'t1': H/'c5a-b2/r67', 'hpm': H/'c5a-b2/hpm', 'objb': H/'c5a-b2/objb', 'slot': H/'c5a-b2/slot-c5a'}
def c5dir(n):
    if n == 'shrinkto': return H/'r32/run-b2/run'
    return C5DIR[n.split('-')[0]] / n
def parse_vec(s): return [int(x, 16) for x in s.split()] if s.strip() and '<empty>' not in s else []
def uart(p):
    p = pathlib.Path(p) / 'uart.txt'
    if not p.exists(): return None
    t = p.read_text(errors='replace'); m = re.findall(r'^SR ([0-9A-Fa-f]+)', t, re.M)
    if not m: return []
    cnt = int(m[-1], 16); sv = re.findall(r'^SV ([0-9A-Fa-f]+)', t, re.M)
    return [int(x, 16) for x in sv[-cnt:]] if cnt else []
rows = {}
for l in open(vf):
    if l.startswith('#') or not l.strip(): continue
    n = l.split()[0]; e0 = l.split('E5N:')[1].split('|')[0]; c5 = l.split('C5a:')[1]
    rows[n] = (parse_vec(e0), parse_vec(c5))
nodata = fail = 0
for n, (e0s, c5s) in rows.items():
    e0 = uart(run / n); c5 = uart(c5dir(n))
    if e0 is None or c5 is None:
        print(f"NO DATA {n}: E0 run {'missing' if e0 is None else 'ok'}, C.5a silicon {'missing' if c5 is None else 'ok'}"); nodata += 1; continue
    bad = []
    if not e0s and not c5s:
        if e0 != c5: bad.append(f"empty-sim image: E0 {e0} vs C.5a silicon {c5}")
    elif len(e0) != len(e0s): bad.append(f"count {len(e0)} vs E0 sim {len(e0s)}")
    else:
        for k in range(len(e0s)):
            if k >= len(c5) or k >= len(c5s): bad.append(f"field {k}: no C.5a reference"); continue
            if c5[k] == c5s[k]:
                if e0[k] != e0s[k]: bad.append(f"VALUE field {k}: {e0[k]:#x} != E0 sim {e0s[k]:#x}")
            elif abs(e0s[k] - c5s[k]) < max(0.01 * c5s[k], 8):
                if abs(e0[k] - c5[k]) > max(0.03 * c5[k], 40): bad.append(f"TIMING field {k}: {e0[k]} vs C.5a silicon {c5[k]} (> max(3 %, 40))")
            elif e0s[k] < c5s[k]:
                if not e0[k] < c5[k]: bad.append(f"TIMING field {k}: {e0[k]} not below C.5a silicon {c5[k]} (sim {e0s[k]} < {c5s[k]})")
            else:
                if not e0[k] > c5[k]: bad.append(f"TIMING field {k}: {e0[k]} not above C.5a silicon {c5[k]}")
    if bad: fail += 1; print(f"FAIL {n}: " + "; ".join(bad))
    else: print(f"PASS {n}" + (f"  (timing: " + ", ".join(f"f{k} {e0[k]}/{c5[k]}" for k in range(min(len(e0), len(c5), len(c5s))) if c5[k] != c5s[k]) + ")" if e0 and c5s else ""))
print(f"E0 GATE: {len(rows) - fail - nodata} PASS, {fail} FAIL, {nodata} NO DATA of {len(rows)}")
sys.exit(2 if nodata else (1 if fail else 0))
