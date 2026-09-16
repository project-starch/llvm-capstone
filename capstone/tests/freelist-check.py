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
bound = int(sys.argv[sys.argv.index('--bound') + 1]) if '--bound' in sys.argv else 16383  # saturating retirement: a node AT this generation is never pushed
rows = []
for ln in open(path):
    m = re.match(r'\s+0x[0-9a-f]+\s+cyc\s+(\d+)\s+(\S+)\s+(.*)', ln)
    if m: rows.append((int(m.group(1)), m.group(2), m.group(3).strip()))
if not rows or 'ALIGNMENT FAILED' in open(path).read() or 'UNCONSUMED' in open(path).read():
    print('NOT AN ALIGNED READINGS FILE'); sys.exit(2)
def cap(v):
    m = re.match(r'revnode_id=(\d+)', v); return int(m.group(1)) if m else None
bad = []
allrows = [(c, cap(v), v) for c, r, v in rows if r in ('a3', 'a1') and (cap(v) is not None or 'UNREADABLE' in v)]
# RUNS. A reclaimed index is popped at generation 0, 1, 2, ... until it reaches `bound`; a node AT the
# bound is never pushed again, so the index RETIRES and the next allocation BUMPS a fresh index, which
# then enters the same cycle. So the pop sequence is a series of runs, each on one index, each starting
# at generation 0 (the bump) or at the first generation seen, each strictly increasing, and a retired
# index must never reappear in a later run. An earlier version of this checker assumed every
# post-retirement pop was a bump and reported the CORRECT 16,386-round run as a violation (2026-09-16).
runs, cur = [], []
for c, v, txt in allrows:
    if v is None: cur.append((c, v)); continue           # torn line: stays in the current run
    if cur and any(x[1] is not None for x in cur) and (v & 0xffff) != next(x[1] for x in reversed(cur) if x[1] is not None) & 0xffff:
        runs.append(cur); cur = []
    cur.append((c, v))
if cur: runs.append(cur)
run_idx = [next(x[1] for x in r if x[1] is not None) & 0xffff for r in runs]
if len(set(run_idx)) != len(run_idx): bad.append('an index is reclaimed again in a LATER run after retiring: %s' % run_idx)
first_idx = run_idx[0]
if first_idx != 3: bad.append('the first reclaimed index is not 3: %s' % first_idx)
sliver = [(c, v) for c, v in runs[0]]
post = [x for r in runs[1:] for x in r]
retired = len(runs) > 1
for k, r in enumerate(runs):
    g = [v >> 16 for c, v in r if v is not None]
    if any(bb <= aa for aa, bb in zip(g, g[1:])): bad.append('run %d (index %d): generation not strictly increasing: %s' % (k, run_idx[k], g[:12]))
    if any(gg > bound for gg in g): bad.append('run %d (index %d): generation EXCEEDS the bound %d' % (k, run_idx[k], bound))
    if k > 0 and g and g[0] != 0: bad.append('run %d (index %d) does not start at a bump (generation %d)' % (k, run_idx[k], g[0]))
    if k < len(runs) - 1 and g and max(g) != bound: bad.append('run %d (index %d) ended at generation %d, not the bound %d -- an index stopped being reclaimed early' % (k, run_idx[k], max(g), bound))
gens = [v >> 16 for c, v in sliver if v is not None]
unread = sum(1 for c, v in sliver if v is None)
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
    cp = cap(tail[0][1])
    if cp is None: bad.append('closing pop is not a capability print: %s' % tail[0][1])
    elif retired:
        # index 3 retired: it is dead under any RTL, so LCC on the retained gen-0 sliver reads 0 in BOTH
        # columns and separates nothing. The closing pop must belong to the LAST run, still being reclaimed.
        if (cp & 0xffff) != run_idx[-1]: bad.append('closing pop is not on the last reclaimed index %d: %s' % (run_idx[-1], tail[0][1]))
        exp_sliver = '0'
    elif (cp & 0xffff) != 3 or (cp >> 16) != max(gens): bad.append('closing pop is not index 3 at the top generation: %s' % tail[0][1])
    if tail[1][1] != exp_sliver: bad.append('LCC(sliver) = %s, expected %s (%s)' % (tail[1][1], exp_sliver, 'retired -- separates nothing' if retired else ('control' if control else 'A5')))
    if tail[2][1] != '1': bad.append('LCC(base) = %s, expected 1' % tail[2][1])
    if tail[3][1] != '0': bad.append('traps = %s, expected 0 (a CORRUPT pop would show here)' % tail[3][1])
print('%s: %d runs %s; first run: %d pops of index %d at generations %d..%d (%d unreadable)%s; %d bump ids %d..%d; %d dropped handles; tail %s' % (
    path.split('/')[-1], len(runs), run_idx[:4] + (['...'] if len(run_idx) > 4 else []), len(sliver), first_idx,
    min(gens), max(gens), unread, (', RETIRED at the bound %d, then %d pops on later indices' % (bound, len(post))) if retired else '',
    len(bumps), bumps[0], bumps[-1], len(dropped), tail))
for b in bad: print('  VIOLATION:', b)
print('  ' + ('PASS' if not bad else 'FAIL') + (' (control column: stale sliver expected to read 1)' if control else ''))
sys.exit(1 if bad else 0)
