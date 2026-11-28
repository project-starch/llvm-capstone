#!/usr/bin/env python3
# thin_keep.py TRACE NPZ OUT FRACTIONS [--dry]: re-place a trace's timer ticks at silicon's positions (rtl lane, 2026-11-27).
#
# NPZ is escan.py's extraction of TRACE, with its P (kind 9) markers. FRACTIONS is one float per line, F_k in [0, 1): silicon
# tick k's cumulative PROGRAM instructions from the window's first tick over the window's total. The key must be program
# instructions on both sides: all-mode counts drift, because each side's kernel runs at its own tick rate (17.2 % of ic7's
# instructions_all is non-protected) and silicon's kernel share follows its ticks. The trace's window is its protected run,
# from the first to the last marker pair at which the protected count moves. A tick's position is its protected count,
# interpolated by event index between kind-9 markers (ic7: a median 5.5 k protected instructions apart, against a
# median tick gap of 27.5 k). Ticks in a non-protected stretch share a position and are interchangeable for the model.
# (First cut, withdrawn before use: rank by all-mode fraction.)
#
# Ticks outside the window are kept. Inside, F_k takes the tick nearest in position. If that tick is taken, an unused
# tick at the SAME position (a tie: ticks taken while no protected code ran) is used. Otherwise it is a COLLISION and
# that silicon tick's flush is lost, which only raises n (toward BUILD): the kit's registered rule (b5b2-ticklog PREREG,
# Collisions; Amendment 1, ties). A forward-push variant, which displaced ticks by up to 53,653 positions on silicon's
# log and so could move flushes INTO the slice, was withdrawn before its n was read. Every in-window tick not kept
# has its flush window
# cleared, as in thin.py: the MTI edge through the MRET after the next cause-9 edge, inclusive. --dry prints the selection
# and writes nothing.
import sys, os, numpy as np


def tick_windows(op, cause):
    mti = np.nonzero((op == 0) & (cause == 0x80000007))[0]
    ec9 = np.nonzero((op == 0) & (cause == 9))[0]
    p = np.searchsorted(ec9, mti)
    ok = p < len(ec9)
    end = np.full(len(mti), -1)
    end[ok] = ec9[p[ok]] + 1
    ok &= end < len(op)
    ok[ok] &= op[end[ok]] == 3
    return mti, end, ok


def main():
    a = [x for x in sys.argv[1:] if x != '--dry']
    dry = '--dry' in sys.argv
    trace, npz, out, fr = a
    z = np.load(npz)
    Ei, w0, w1 = z['Ei'], z['Ew0'], z['Ew1']
    Ii, Iv, Pi, Pv = z['Ii'], z['Iv'].astype(np.float64), z['Pi'], z['Pv']
    op = ((w0 >> 56) & 3).astype(int)
    cause = (w0 & 0xffffffff).astype(np.int64)
    fl = (w1 & 1).astype(bool)
    mti, end, ok = tick_windows(op, cause)
    if not ok.all():
        sys.exit('ERROR: %d ticks without a closing MRET' % int((~ok).sum()))
    # the window: P markers sit beside I markers; the protected count's first and last moves
    mv = np.nonzero(np.diff(Pv) > 0)[0]
    if len(mv) == 0:
        sys.exit('ERROR: the protected count never moves; no window')
    lo_ev, hi_ev = Pi[mv[0]], Pi[mv[-1] + 1]
    lo, hi = np.interp([float(lo_ev), float(hi_ev)], Ii.astype(np.float64), Iv)
    plo, phi = np.interp([float(lo_ev), float(hi_ev)], Pi.astype(np.float64), Pv.astype(np.float64))
    inw = (Ei[mti] >= lo_ev) & (Ei[mti] <= hi_ev)
    w = np.nonzero(inw)[0]
    g = (np.interp(Ei[mti[w]].astype(np.float64), Pi.astype(np.float64), Pv.astype(np.float64)) - plo) / (phi - plo)
    F = np.array([float(x) for x in open(fr) if x.strip()])
    if len(F) == 0 or F.min() < 0 or F.max() >= 1:
        sys.exit('ERROR: fractions empty or outside [0, 1)')
    F = np.sort(F)
    near = np.clip(np.searchsorted(g, F), 1, len(g) - 1)
    near = np.where(np.abs(g[near - 1] - F) <= np.abs(g[near] - F), near - 1, near)
    # tie groups: runs of equal position; a taken tick may hand over to an unused member of its own group only
    gstart = np.searchsorted(g, g, side='left'); gend = np.searchsorted(g, g, side='right')
    nxt = gstart.copy()  # next candidate inside each group, indexed by the group's first tick
    kept = np.zeros(len(w), bool); coll = 0; ties = 0
    for k in range(len(F)):
        m = int(near[k])
        if not kept[m]:
            kept[m] = True; continue
        s = gstart[m]
        while nxt[s] < gend[m] and kept[nxt[s]]:
            nxt[s] += 1
        if nxt[s] < gend[m]:
            kept[nxt[s]] = True; ties += 1
        else:
            coll += 1
    print('tie_resolved', ties, 'collisions', coll, flush=True)
    print('ticks', len(mti), 'in_window', len(w), 'targets', len(F), 'kept', int(kept.sum()), 'collisions', coll,
          'window_instructions %.0f..%.0f' % (lo, hi), flush=True)
    clear = np.zeros(len(Ei), bool)
    for t in w[~kept]:
        clear[mti[t]:end[t] + 1] = True
    clear &= fl
    gidx = Ei[clear]
    print('flush E cleared', len(gidx), flush=True)
    if dry:
        return
    n = os.path.getsize(trace) // 16
    mm = np.memmap(trace, dtype='<u8', mode='r', shape=(n, 2))
    CH = 1 << 24
    with open(out, 'wb') as fo:
        for s in range(0, n, CH):
            blk = np.array(mm[s:s + CH])
            a0, b0 = np.searchsorted(gidx, [s, s + len(blk)])
            sel = gidx[a0:b0] - s
            assert np.all((blk[sel, 0] >> 58) == 10) and np.all(blk[sel, 1] == 1)
            blk[sel, 1] = 0
            blk.tofile(fo)
    print('wrote', out, n, 'events', flush=True)


main()
