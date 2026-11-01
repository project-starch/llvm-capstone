#!/usr/bin/env python3
"""spatial-gate.py <pilot result.json> <reference result.json>: the go/no-go for the Spatial (ObjBounds) SPEC arm.

The pilot is a capstone-spatial stage (MapLife's images) run on the ObjBounds bitstream; the reference is the same
stage's bitstream-3 (MapLife) result. Same images, so the gate requires, per the pre-registration:
  G1 the run is done and every command ended rc 0;
  G2 every host specdiff validation PASSes;
  G3 every hash the reference records is equal (inputs and outputs; stderr *.err excluded, it carries ASLR bases);
  G4 every command's in-context instret is within 0.1 % of the reference's.
The elapsed and cycle ratios are printed, not gated: they are the measurement.
Exit 0 all PASS, 1 any MISS, 2 no data (a missing file, arm or command list is an error, never a pass)."""
import json, sys


def arm_of(path):
    try:
        d = json.load(open(path))
    except (OSError, ValueError) as e:
        print(f"NO DATA: {path}: {e}")
        sys.exit(2)
    arms = d.get('arms') or {}
    if len(arms) != 1:
        print(f"NO DATA: {path}: {len(arms)} arms, expected 1")
        sys.exit(2)
    (name, a), = arms.items()
    if not a.get('commands'):
        print(f"NO DATA: {path}: arm {name} has no commands")
        sys.exit(2)
    return d, name, a


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    pd, pn, p = arm_of(sys.argv[1])
    rd, rn, r = arm_of(sys.argv[2])
    print(f"pilot {pn} {sys.argv[1]}\nref   {rn} {sys.argv[2]}")
    miss = 0

    def check(tag, ok, what):
        nonlocal miss
        print(f"{tag} {'PASS' if ok else 'MISS'}: {what}")
        miss += not ok

    pc, rc = p['commands'], r['commands']
    check('G1', pd.get('done') is True and len(pc) == len(rc) and all(c.get('rc') == 0 for c in pc),
          f"done={pd.get('done')} commands {len(pc)}/{len(rc)} rc={[c.get('rc') for c in pc]}")
    v = p.get('validation') or []
    check('G2', bool(v) and all(x.get('verdict') == 'PASS' for x in v),
          f"specdiff {[x.get('verdict') for x in v]}")
    ph, rh = p.get('hashes') or {}, r.get('hashes') or {}
    # *.err is the launcher's stderr (CAPSTONE_VM_ADD_OK lines with ASLR'd bases): it differs between any two runs.
    errs = sorted(k for k in rh if k.endswith('.err') and ph.get(k) != rh[k])
    keys = [k for k in rh if not k.endswith('.err')]
    diff = sorted(k for k in keys if ph.get(k) != rh[k])
    check('G3', bool(keys) and not diff, f"{len(keys)} reference hashes (stderr excluded), differing or missing: {diff}")
    if errs:
        print(f"INFO stderr files differ (expected): {errs}")
    worst = None
    for a, b in zip(pc, rc):
        if not a.get('instret') or not b.get('instret'):
            worst = float('inf')
            break
        dev = abs(a['instret'] / b['instret'] - 1)
        worst = dev if worst is None else max(worst, dev)
    check('G4', worst is not None and worst <= 0.001, f"largest instret deviation {worst}")
    ps, rs = sum(c.get('seconds') or 0 for c in pc), sum(c.get('seconds') or 0 for c in rc)
    pcy, rcy = sum(c.get('cycles') or 0 for c in pc), sum(c.get('cycles') or 0 for c in rc)
    if ps and rs:
        print(f"INFO elapsed pilot {ps:.2f} s, ref {rs:.2f} s, ref/pilot {rs / ps:.4f}")
    if pcy and rcy:
        print(f"INFO cycles  ref/pilot {rcy / pcy:.4f}")
    print('GATE PASS' if not miss else f'GATE MISS ({miss})')
    return 1 if miss else 0


if __name__ == '__main__':
    sys.exit(main())
