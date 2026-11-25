#!/usr/bin/env python3
"""read-red.py BOARD_TXT TWIN_SERIAL (nested lane, 2026-11-25): M-16's reduction boot on P1 (stage p1red/red1:
r0 = mode s 100 packets, r1 = mode s, r2 = mode f, r3 = mode a; the protected tsapp-stress-red image a15d30b2...).
TWIN_SERIAL is the twin run of the same stage; its STRESS lines are the reference (the output is a function of argv).
Per command, in the twin's order:
  RETURNED     @@SPEC-END rc 0 and every STRESS line equal to the twin's;
  NO RETURN    no @@SPEC-END: the last heartbeat, its fast-allocation count, the twin's next heartbeat (the stop lies
               between the two), and the wedge apertures, with the commit pc compared with this image's CDERIVE in
               wmem_block_fast_sublet_alloc (0x3ff676ca68, the instruction P1's stress stop named at 0x3ff676c9ee in
               its own image) and sw=220 with P1's 0x8c. Every aperture row is printed verbatim;
  OTHER        a STRESS line differs from the twin's, or rc nonzero;
  NOT STARTED  no @@SPEC-BEGIN.
Then the protected fast allocations completed (returned commands' totals plus the last heartbeat of a stopped one).
Exit 0 when every started command is RETURNED or NO RETURN; 1 on OTHER; 2 with no transcript or no twin reference."""
import re, sys
CDERIVE_PC = '0x0000003ff676ca68'


def stress_lines(t):
    out = {}
    for m in re.finditer(r'^@@SPEC-BEGIN (\d+) run (\S+) 0$', t, re.M):
        i = int(m.group(1)); rest = t[m.end():]
        e = re.search(r'^@@SPEC-END %d run ' % i, rest, re.M)
        body = rest[:e.start()] if e else rest
        out[i] = (m.group(2), re.findall(r'^STRESS (.*)$', body, re.M), bool(e))
    return out


def fast(line):
    m = re.search(r'fast=(\d+)', line or '')
    return int(m.group(1)) if m else 0


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    try:
        t = open(sys.argv[1], errors='replace').read().replace('\r', '')
        w = open(sys.argv[2], errors='replace').read().replace('\r', '')
    except OSError as e:
        print('NO DATA:', e); sys.exit(2)
    B, W = stress_lines(t), stress_lines(w)
    if not W or not all(W[i][2] for i in W):
        print('NO DATA: the twin reference has no complete command'); sys.exit(2)
    if not B:
        print('NO DATA: no @@SPEC-BEGIN in the board transcript'); sys.exit(2)
    bad, done = 0, 0
    for i in sorted(W):
        name, ref, _ = W[i]
        if i not in B:
            print(f'{name}: NOT STARTED'); continue
        _, got, ended = B[i]
        n = len(got)
        same = got == ref[:n]
        end = re.search(r'^@@SPEC-END %d run \S+ 0 rc=(\d+) up0=([\d.]+) up1=([\d.]+)' % i, t, re.M)
        if not same:
            k = next(j for j, (a, b) in enumerate(zip(got, ref)) if a != b) if any(a != b for a, b in zip(got, ref)) else min(n, len(ref))
            print(f'{name}: OTHER, first STRESS line differing from the twin at #{k}: {got[k] if k < n else None}')
            bad = 1; continue
        if ended:
            if end and end.group(1) == '0' and n == len(ref):
                print(f'{name}: RETURNED, {n} STRESS lines equal to the twin\'s; {ref[-1]}; '
                      f'span {float(end.group(3)) - float(end.group(2)):.1f} s')
                done += fast(ref[-1])
            else:
                print(f'{name}: OTHER, rc {end.group(1) if end else None}, {n} of {len(ref)} STRESS lines'); bad = 1
            continue
        last = got[-1] if n else None
        nxt = ref[n] if n < len(ref) else None
        print(f'{name}: NO RETURN after {n} STRESS lines equal to the twin\'s; last: {last}')
        print(f'  stop between {fast(last):,} and {fast(nxt):,} fast allocations (twin\'s next: {nxt})')
        done += fast(last)
        blk = re.search(r'^===== vbatch wedge apertures =====\n(.*?)(?:\n=====|\Z)', t, re.M | re.S)
        if not blk:
            print('  NO APERTURE BLOCK in the transcript'); continue
        for l in blk.group(1).split('\n'):
            if l.strip():
                print('  |', l)
        pc = re.search(r'commit pc \(.*?\) (0x[0-9a-f]+)', blk.group(1))
        s220 = re.search(r'^sw=220 .*? (0x[0-9a-f]{2}) ', blk.group(1), re.M)
        print(f'  commit pc at this image\'s CDERIVE {CDERIVE_PC}: {bool(pc) and pc.group(1) == CDERIVE_PC}; '
              f'sw=220 {s220.group(1) if s220 else None} equal to P1\'s 0x8c: {bool(s220) and s220.group(1) == "0x8c"}')
    print(f'protected fast allocations completed: {done:,}')
    sys.exit(1 if bad else 0)


main()
