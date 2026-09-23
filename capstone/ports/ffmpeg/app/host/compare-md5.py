#!/usr/bin/env python3
"""Compare the hash column of a program's frame lines against a framemd5 reference.

    compare-md5.py <reference.framemd5> <candidate.out> [--control <flipped.out>]

PASS only if:
  * the candidate has exactly as many frame lines as the reference, every hash equal;
  * with --control: the control output DIFFERS from the reference (a frame count mismatch
    or at least one changed hash). A comparison that cannot fail is not evidence, so a
    control that matches the reference is a FAILURE of this check, not a pass.
No frame lines in the reference is an ERROR, never an empty pass.
"""
import re
import sys

FRAME = re.compile(r'^\s*0,\s*-?\d+,\s*-?\d+,\s*\d+,\s*\d+,\s*([0-9a-f]{32})\s*$')


def hashes(path):
    return [m.group(1) for m in (FRAME.match(line) for line in open(path)) if m]


def main(argv):
    if len(argv) not in (3, 5) or (len(argv) == 5 and argv[3] != '--control'):
        sys.exit(__doc__)
    ref, cand = hashes(argv[1]), hashes(argv[2])
    if not ref:
        sys.exit(f'ERROR: no frame lines in the reference {argv[1]}')
    diff = [i for i, (a, b) in enumerate(zip(ref, cand)) if a != b]
    ok = len(cand) == len(ref) and not diff
    print(f'oracle: reference {len(ref)} frames, candidate {len(cand)} frames, '
          f'{len(diff)} hash mismatches -> {"MATCH" if ok else "MISMATCH"}')
    if diff:
        print(f'  first mismatching frame: {diff[0]}')
    if len(argv) == 5:
        ctl = hashes(argv[4])
        changed = sum(1 for a, b in zip(ref, ctl) if a != b)
        fired = len(ctl) != len(ref) or changed > 0
        print(f'control: {len(ctl)} frames, {changed} changed hashes -> '
              f'{"FIRES" if fired else "DID NOT FIRE"}')
        ok = ok and fired
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main(sys.argv)
