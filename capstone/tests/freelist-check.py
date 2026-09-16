#!/usr/bin/env python3
# freelist-check.py READINGS.txt [--control] -- layer 2 of r12-recl-freelist: from the retirement-aligned
# readings (capprint-retired.py output), check the invariants the free list must keep across NROUNDS:
#  * the sliver index (a3 first, then every a1) is the SAME index at STRICTLY INCREASING generation;
#  * bump ids (s1) are strictly increasing bare indices at generation 0, never 0..2;
#  * a dropped handle's index (s4) NEVER appears again in any later print;
#  * no index 0..2 at generation >= 1 anywhere;
#  * the final prints: LCC(sliver) then LCC(base) then traps. Under --control (gen-blind A34) the stale
#    sliver reads 1; under A5 it must read 0. Base 1, traps 0 either way.
# Exits 1 on any violation, 2 if the readings file is not an aligned one.
import re, sys
path = sys.argv[1]; control = '--control' in sys.argv
rows = []
for ln in open(path):
    m = re.match(r'\s+0x[0-9a-f]+\s+cyc\s+(\d+)\s+(\S+)\s+(.*)', ln)
    if m: rows.append((int(m.group(1)), m.group(2), m.group(3).strip()))
if not rows or 'ALIGNMENT FAILED' in open(path).read() or 'UNCONSUMED' in open(path).read():
    print('NOT AN ALIGNED READINGS FILE'); sys.exit(2)
def cap(v):
    m = re.match(r'revnode_id=(\d+)', v); return int(m.group(1)) if m else None
bad = []
sliver = [(c, cap(v), v) for c, r, v in rows if r in ('a3', 'a1') and (cap(v) is not None or 'UNREADABLE' in v)]
sliver = [s for s in sliver if not (s[1] is None and 'UNREADABLE' not in s[2])]
idx = {s[1] & 0xffff for s in sliver if s[1] is not None}
if len(idx) != 1: bad.append('sliver index not unique: %s' % sorted(idx))
gens = [s[1] >> 16 for s in sliver if s[1] is not None]
if any(b <= a for a, b in zip(gens, gens[1:])): bad.append('sliver generation not strictly increasing: %s' % gens[:12])
unread = sum(1 for s in sliver if s[1] is None)
bumps = [cap(v) for c, r, v in rows if r == 's1' and cap(v) is not None]
if any(b >> 16 for b in bumps): bad.append('a bump id at generation != 0')
if any(b <= a for a, b in zip(bumps, bumps[1:])): bad.append('bump ids not strictly increasing')
if any((b & 0xffff) <= 2 for b in bumps): bad.append('bump id at a sentinel index')
dropped = [(c, cap(v)) for c, r, v in rows if r == 's4' and cap(v) is not None]
allcaps = [(c, cap(v)) for c, r, v in rows if cap(v) is not None]
for dc, d in dropped:
    later = [v for c, v in allcaps if c > dc and (v & 0xffff) == (d & 0xffff)]
    if later: bad.append('dropped index %d recurs later as %s' % (d & 0xffff, later[:3]))
if any((v & 0xffff) <= 2 and (v >> 16) >= 1 for c, v in allcaps): bad.append('a sentinel index at generation >= 1')
tail = [(r, v) for c, r, v in rows[-4:]]
exp_sliver = '1' if control else '0'
# the closing pop leaves index 3 ALIVE at generation NROUNDS, so LCC on the retained gen-0 sliver separates
# the columns: 1 gen-blind (valid alone), 0 under A5 (generation mismatch). A dead node would read 0 anywhere.
if not (len(tail) == 4 and tail[0][0] == 'a1' and tail[1][0] == 'a4' and tail[2][0] == 'a4' and tail[3][0] == 's11'): bad.append('tail is not a1,a4,a4,s11: %s' % tail)
else:
    if cap(tail[0][1]) is None or (cap(tail[0][1]) & 0xffff) != 3 or (cap(tail[0][1]) >> 16) != max(gens): bad.append('closing pop is not index 3 at the top generation: %s' % tail[0][1])
    if tail[1][1] != exp_sliver: bad.append('LCC(sliver) = %s, expected %s (%s)' % (tail[1][1], exp_sliver, 'control' if control else 'A5'))
    if tail[2][1] != '1': bad.append('LCC(base) = %s, expected 1' % tail[2][1])
    if tail[3][1] != '0': bad.append('traps = %s, expected 0 (a CORRUPT pop would show here)' % tail[3][1])
print('%s: %d rounds of sliver at index %s, generations %d..%d (%d unreadable), %d bump ids %d..%d, %d dropped handles, tail %s' % (
    path.split('/')[-1], len(sliver), sorted(idx), min(gens), max(gens), unread, len(bumps), bumps[0], bumps[-1], len(dropped), tail))
for b in bad: print('  VIOLATION:', b)
print('  ' + ('PASS' if not bad else 'FAIL') + (' (control column: stale sliver expected to read 1)' if control else ''))
sys.exit(1 if bad else 0)
