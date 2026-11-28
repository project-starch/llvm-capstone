#!/usr/bin/env python3
# eseq.py EDGES.npz : the edge sequences that start at each M-timer trap, and the record events between its MRET and the next edge.
import sys, numpy as np, collections
z = np.load(sys.argv[1]); Ei, w0, w1 = z['Ei'], z['Ew0'], z['Ew1']; Ti = z['Ti']
op = ((w0 >> 56) & 3).astype(int); cause = (w0 & 0xffffffff).astype(np.int64)
lab = np.where(op == 0, cause, -1 - op)  # trap -> cause; root -> -2, sret -> -3, mret -> -4
def name(x):
    return {-2: 'ROOT', -3: 'SRET', -4: 'MRET'}.get(int(x), 'trap0x%x' % (int(x) & 0xffffffff))
# 4-grams starting at each MTI trap
mti = np.nonzero(lab == 0x80000007)[0]; mti = mti[mti + 5 < len(lab)]
c = collections.Counter(tuple(name(lab[i + k]) for k in range(6)) for i in mti)
for k, n in c.most_common(6): print(n, ' -> '.join(k))
# events between MTI's MRET and the next edge (non-E events = record/I/T events in between)
nxt = mti + 1; gap = Ei[nxt + 1] - Ei[nxt] - 1
print('events between MRET(after MTI) and the next edge: zero', int((gap == 0).sum()), 'of', len(gap), 'median', int(np.median(gap)), 'p99', int(np.percentile(gap, 99)))
