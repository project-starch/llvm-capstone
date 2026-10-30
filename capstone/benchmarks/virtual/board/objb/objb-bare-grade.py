#!/usr/bin/env python3
"""Grade one objb-bare boot by value, with the RTL lane's rules (land/rtl-objb beb5384abb3b,
capstone/tests/rtl-smoke/objb-bare-2026-10-30/PREREG.md, and its Addendum 1 at 177c6fd9f6e9: both arms on board 2,
one verdict per arm; the vectors below are its tables, verbatim).

  objb-bare-grade.py OUT_DIR ARM                 ARM = bs3 (bitstream 3, e7af48457) or objb (ObjBounds)
  objb-bare-grade.py --compare BS3_DIR OBJB_DIR  the cost pattern, silicon against silicon

OUT_DIR holds one run_sup_bare.py output dir per image (uart.txt, run.log, rc). A reading is the image's `SR` count
and its `SV` values, from uart.txt.

Rules (Addendum 1's verdicts; each one but the passing one stops the driver):
- VOID unless objb-pcc-ctrl reads SR 3 `0xbb5 0x0 0x0`, and every image that returns on this bitstream (all ten on
  ObjBounds, all but objb-pcc on bitstream 3) returned and reads its table SR.
- ObjBounds: STOP if objb-stale reads bitstream 3's vector (the variant is not on the board, or the macro did not
  reach it). CONFIRMED when objb-stale reads the ObjBounds vector exactly AND objb-pcc returns `0xbb5 0x0 0x0`.
  Any other identity reading is a FINDING. So is a cost shape whose non-cycle positions (minstret, value, trap count,
  cause) differ from the table: if ObjBounds changes anything but cycles, the rung measures the wrong thing (the
  coordinator's ruling, 2026-10-30). CONFIRMED needs the identity AND every cost shape's non-cycle positions.
- Bitstream 3: AS EXPECTED only when the control holds, every returning image reads its SR, objb-stale reads
  bitstream 3's vector, every cost shape's non-cycle positions equal the table, and objb-pcc does NOT return.
  STOP (wrong bitstream) if objb-stale reads the ObjBounds vector or objb-pcc returns; FINDING for any other mismatch.
- The cycle positions are printed. The pattern (--compare): s1 and s5's churn equal within 1 %; s2, s3,
  s4 (both passes), s5's passes, s6 and s7 lower on ObjBounds.
Exit 0 = CONFIRMED (ObjBounds) / AS EXPECTED (bitstream 3), 1 = VOID, 3 = STOP, 4 = FINDING, 2 = no data.
"""
import re, sys
from pathlib import Path

ORDER = ['objb-pcc-ctrl', 'objb-stale'] + [f'objb-perf-s{i}' for i in range(1, 8)] + ['objb-pcc']
# image: (SR, bitstream-3 vector, ObjBounds vector, cycle positions)
T = {
    'objb-pcc-ctrl': (3, [0xbb5, 0, 0], [0xbb5, 0, 0], ()),
    'objb-stale': (9, [0x19, 0x19, 0x19, 0x19, 0x0, 0x19, 0x19, 0x77, 0x6], [0, 0, 0, 0, 0, 0, 0, 0x3333, 0], ()),
    'objb-perf-s1': (5, [0x414f, 0x1403, 0x5a, 0, 0], [0x414f, 0x1403, 0x5a, 0, 0], (0,)),
    'objb-perf-s2': (5, [0x2423, 0xc03, 0x80004000, 0, 0], [0x145f, 0xc03, 0x80004000, 0, 0], (0,)),
    'objb-perf-s3': (5, [0x2b709, 0x1804, 0x80004000, 0, 0], [0x22744, 0x1804, 0x80004000, 0, 0], (0,)),
    'objb-perf-s4': (7, [0x290, 0xc3, 0x2c4, 0xc3, 0x80004000, 0, 0], [0x1d0, 0xc3, 0x1d0, 0xc3, 0x80004000, 0, 0], (0, 2)),
    'objb-perf-s5': (6, [0xa0d8f, 0x4da4d, 0x400a, 0xffc00, 0, 0], [0xa0c5f, 0x3b049, 0x400a, 0xffc00, 0, 0], (0, 1)),
    'objb-perf-s6': (5, [0x2430, 0x503, 0, 0, 0], [0x1422, 0x503, 0, 0, 0], (0,)),
    'objb-perf-s7': (5, [0x2c72, 0x503, 0x80004000, 0, 0], [0x18c4, 0x503, 0x80004000, 0, 0], (0,)),
    'objb-pcc': (3, None, [0xbb5, 0, 0], ()),
}
# the pattern: (image, position, 'equal' within 1 % or 'lower' on ObjBounds)
PATTERN = [('objb-perf-s1', 0, 'equal'), ('objb-perf-s5', 0, 'equal'), ('objb-perf-s2', 0, 'lower'),
           ('objb-perf-s3', 0, 'lower'), ('objb-perf-s4', 0, 'lower'), ('objb-perf-s4', 2, 'lower'),
           ('objb-perf-s5', 1, 'lower'), ('objb-perf-s6', 0, 'lower'), ('objb-perf-s7', 0, 'lower')]


def h(v):
    return ' '.join(hex(x) for x in v)


def read(d):
    """(returned, SR count or None, SV values) for one image's output dir."""
    u = d / 'uart.txt'
    if not u.exists():
        return False, None, []
    t = u.read_text(errors='replace')
    sr = [int(x, 16) for x in re.findall(r'SR ([0-9A-F]{16})', t)]
    sv = [int(x, 16) for x in re.findall(r'SV ([0-9A-F]{16})', t)]
    return 'SUPTEST END' in t, (sr[-1] if sr else None), sv


def grade(out, arm):
    lines, void, stop, find = [], [], [], []
    if not out.is_dir() or not any((out / i).is_dir() for i in ORDER):
        print(f'NO DATA: no image output under {out}'); return 2
    hard = [i for i in ORDER if (out / i / 'run.log').exists() and 'HARD STOP' in (out / i / 'run.log').read_text(errors='replace')]
    if hard:
        print(f'OBJB-BARE {arm} VOID: resident bitstream mismatch (HARD STOP) at {hard[0]}'); return 1
    R = {i: read(out / i) for i in ORDER}
    for i in ORDER:
        sr, v3, vo, cyc = T[i]
        ret, n, sv = R[i]
        want = vo if arm == 'objb' else v3
        expect_return = not (arm == 'bs3' and i == 'objb-pcc')
        if not expect_return:
            lines.append(f'  {i:15} returned={ret} (bitstream 3: no return expected) SR={n} SV={h(sv)}')
            if ret:
                stop.append(f'objb-pcc returned on the bitstream-3 board: {h(sv)}')
            continue
        lines.append(f'  {i:15} returned={ret} SR={n} SV={h(sv)}')
        if not ret or n != sr:
            void.append(f'{i}: returned={ret} SR={n}, want SR {sr}')
            continue
        if cyc:
            nc = [k for k in range(len(want)) if k not in cyc]
            if len(sv) != len(want) or any(sv[k] != want[k] for k in nc):
                find.append(f'{i}: a non-cycle position differs ({h(sv)} vs {h(want)}); its cycles are not read')
            else:
                lines[-1] += '   non-cycle positions EQUAL; cycles ' + ' '.join(f'[{k}]={sv[k]}' for k in cyc)
    ret, n, sv = R['objb-pcc-ctrl']
    ctl_bad = not (ret and n == 3 and sv == [0xbb5, 0, 0])
    if ctl_bad:
        void.insert(0, f'objb-pcc-ctrl reads returned={ret} SR={n} {h(sv)}, want SR 3 0xbb5 0x0 0x0')
    stale = R['objb-stale'][2]
    if arm == 'objb':
        if ctl_bad:
            verdict, rc = 'VOID: the control failed', 1
        elif stale == T['objb-stale'][1]:
            verdict, rc = "STOP: objb-stale reads bitstream 3's vector on the ObjBounds board", 3
        elif void:
            verdict, rc = 'VOID: a returning image did not read its SR', 1
        elif not (stale == T['objb-stale'][2] and R['objb-pcc'][0] and R['objb-pcc'][2] == [0xbb5, 0, 0]):
            verdict, rc = f'FINDING: identity not confirmed (objb-stale {h(stale)}, objb-pcc returned={R["objb-pcc"][0]} {h(R["objb-pcc"][2])})', 4
        elif find:
            verdict, rc = 'FINDING: identity confirmed, but a cost shape changed something other than cycles', 4
        else:
            verdict, rc = 'CONFIRMED', 0
    else:
        if stale == T['objb-stale'][2]:
            stop.append("objb-stale reads the ObjBounds vector on the bitstream-3 board")
        elif stale != T['objb-stale'][1] and R['objb-stale'][0]:
            find.append(f'objb-stale reads {h(stale)}, not the predicted bitstream-3 vector (first silicon reading on e7af48457)')
        if ctl_bad:
            verdict, rc = 'VOID: the control failed', 1
        elif stop:
            verdict, rc = 'STOP: wrong bitstream: ' + '; '.join(stop), 3
        elif void:
            verdict, rc = 'VOID: a returning image did not read its SR', 1
        elif find:
            verdict, rc = 'FINDING', 4
        else:
            verdict, rc = 'AS EXPECTED (objb-pcc did not return)', 0
    print(f'OBJB-BARE {arm} {verdict}')
    for l in lines:
        print(l)
    for v in void:
        print(f'  VOID {v}')
    for f in find:
        print(f'  FINDING {f}')
    return rc


def compare(b3, ob):
    bad = 0
    for i, k, kind in PATTERN:
        r3, ro = read(b3 / i), read(ob / i)
        if not (r3[0] and ro[0]) or len(r3[2]) <= k or len(ro[2]) <= k:
            print(f'  {i}[{k}] NO DATA'); bad += 1; continue
        a, b = r3[2][k], ro[2][k]
        ok = abs(b - a) <= 0.01 * a if kind == 'equal' else b < a
        bad += not ok
        print(f'  {i}[{k}] bitstream 3 {a} ObjBounds {b} ratio {b / a:.4f} predicted {kind}: {"HOLDS" if ok else "OFF"}')
    print(f'OBJB-BARE pattern: {len(PATTERN) - bad} of {len(PATTERN)} hold')
    return 1 if bad else 0


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--compare':
        sys.exit(compare(Path(sys.argv[2]), Path(sys.argv[3])))
    if len(sys.argv) != 3 or sys.argv[2] not in ('bs3', 'objb'):
        sys.exit(__doc__)
    sys.exit(grade(Path(sys.argv[1]), sys.argv[2]))
