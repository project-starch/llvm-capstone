#!/usr/bin/env python3
# eana.py EDGES.npz : tally E records by op, trap causes with their flush bit, and the trap that precedes each MRET.
import sys, numpy as np, collections
z = np.load(sys.argv[1])
Ei, w0, w1, Ti, Ii, Iv = z['Ei'], z['Ew0'], z['Ew1'], z['Ti'], z['Ii'], z['Iv']
op = ((w0 >> 56) & 3).astype(int); cause = (w0 & 0xffffffff).astype(np.int64); fl = (w1 & 1).astype(int)
print('E', len(Ei), 'by op', np.bincount(op, minlength=4).tolist(), 'flush by op', [int(fl[op == k].sum()) for k in range(4)])
print('T', len(Ti), 'I', len(Ii))
# trap causes, split flush / no flush
tr = op == 0
c = collections.Counter(zip(cause[tr].tolist(), fl[tr].tolist()))
for (cs, f), n in sorted(c.items(), key=lambda x: -x[1])[:15]:
    print('trap cause 0x%x async=%d code=%d flush=%d n=%d' % (cs, cs >> 31, cs & 0x7fffffff, f, n))
# for each MRET: the cause of the most recent preceding trap edge
tidx = np.nonzero(tr)[0]; midx = np.nonzero(op == 3)[0]
pos = np.searchsorted(tidx, midx) - 1
prev = np.where(pos >= 0, cause[tidx[np.maximum(pos, 0)]], -1)
pf = np.where(pos >= 0, fl[tidx[np.maximum(pos, 0)]], -1)
c2 = collections.Counter(zip(prev.tolist(), pf.tolist()))
for (cs, f), n in sorted(c2.items(), key=lambda x: -x[1])[:12]:
    print('MRET preceded by trap cause 0x%x code=%d trapflush=%d n=%d' % (cs & 0xffffffff, cs & 0x7fffffff, f, n))
# instruction position (protected icount is the next I record? use I value at or before)
