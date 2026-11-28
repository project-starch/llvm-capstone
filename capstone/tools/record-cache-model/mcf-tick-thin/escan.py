#!/usr/bin/env python3
# escan.py TRACE OUT.npz : extract every E (kind 10), T (kind 7), I (kind 8) and P (kind 9) record with its event index.
import sys, numpy as np, os
path, out = sys.argv[1], sys.argv[2]
n = os.path.getsize(path) // 16
mm = np.memmap(path, dtype='<u8', mode='r', shape=(n, 2))
CH = 1 << 24
Ei, Ew0, Ew1, Ti, Ii, Iv, Pi, Pv = [], [], [], [], [], [], [], []
for s in range(0, n, CH):
    blk = np.asarray(mm[s:s + CH])
    kind = (blk[:, 0] >> 58).astype(np.uint8)
    e = np.nonzero(kind == 10)[0]; t = np.nonzero(kind == 7)[0]; i = np.nonzero(kind == 8)[0]; q = np.nonzero(kind == 9)[0]
    Ei.append(e + s); Ew0.append(blk[e, 0]); Ew1.append(blk[e, 1]); Ti.append(t + s)
    Ii.append(i + s); Iv.append(blk[i, 0] & ((1 << 56) - 1))
    Pi.append(q + s); Pv.append(blk[q, 0] & ((1 << 56) - 1))
cat = np.concatenate
np.savez(out, n=n, Ei=cat(Ei), Ew0=cat(Ew0), Ew1=cat(Ew1), Ti=cat(Ti), Ii=cat(Ii), Iv=cat(Iv), Pi=cat(Pi), Pv=cat(Pv))
print('events', n, 'E', sum(len(x) for x in Ei), 'T', sum(len(x) for x in Ti), 'I', sum(len(x) for x in Ii))
