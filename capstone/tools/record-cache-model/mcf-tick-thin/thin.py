#!/usr/bin/env python3
# thin.py TRACE NPZ OUT KEEP_NUM KEEP_DEN: copy TRACE to OUT, clearing the flush bit (word 1 bit 0) of every E record of
# the ticks NOT kept. A tick = an M-timer trap edge (cause 0x80000007) through the MRET that follows the next cause-9
# (ecall from S) trap edge. Tick t is kept iff (t * KEEP_NUM) // KEEP_DEN != ((t + 1) * KEEP_NUM) // KEEP_DEN.
import sys, os, numpy as np
trace, npz, out, kn, kd = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
z = np.load(npz); Ei, w0, w1 = z['Ei'], z['Ew0'], z['Ew1']
op = ((w0 >> 56) & 3).astype(int); cause = (w0 & 0xffffffff).astype(np.int64); fl = (w1 & 1).astype(bool)
mti = np.nonzero((op == 0) & (cause == 0x80000007))[0]
ec9 = np.nonzero((op == 0) & (cause == 9))[0]
t = np.arange(len(mti)); keep = (t * kn) // kd != ((t + 1) * kn) // kd
clear = np.zeros(len(Ei), bool); bad = 0
for j in mti[~keep]:
    p = np.searchsorted(ec9, j); 
    if p >= len(ec9): bad += 1; continue
    k = ec9[p] + 1
    if k >= len(op) or op[k] != 3: bad += 1; continue
    clear[j:k + 1] = True
clear &= fl
gidx = Ei[clear]
print('ticks', len(mti), 'kept', int(keep.sum()), 'flush E cleared', len(gidx), 'unmatched', bad, flush=True)
n = os.path.getsize(trace) // 16
mm = np.memmap(trace, dtype='<u8', mode='r', shape=(n, 2))
CH = 1 << 24
with open(out, 'wb') as f:
    for s in range(0, n, CH):
        blk = np.array(mm[s:s + CH])
        a, b = np.searchsorted(gidx, [s, s + len(blk)])
        sel = gidx[a:b] - s
        assert np.all((blk[sel, 0] >> 58) == 10) and np.all(blk[sel, 1] == 1)
        blk[sel, 1] = 0
        blk.tofile(f)
print('wrote', out, n, 'events', flush=True)
