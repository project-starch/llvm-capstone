#!/usr/bin/env python3
"""read-stress.py BOARD_TXT TWIN_SERIAL (nested lane, 2026-11-25): the M-16 stress boot (stage p1st/stress1: s0 = 100
packets, s1 = 400,000 packets, both the protected tsapp_stress 84d347d8). TWIN_SERIAL is the twin run of the same
stage, whose STRESS lines are the reference (the checksum is a function of argv only).
Per command: RETURNED (@@SPEC-END rc 0 and its STRESS done line equal to the twin's), NO RETURN (no @@SPEC-END: the
last heartbeat packet, its fast-allocation count, and the wedge apertures, with the commit pc compared with the
stress image's CDERIVE in wmem_block_fast_sublet_alloc, 0x3ff676c9ee, the site of tsu-p1's stop at 0x3ff6780f04), or
OTHER. Every heartbeat of the board is compared with the twin's line for the same packet. Exit 2 with no transcript or
no twin reference."""
import re, sys
CDERIVE_PC = '0x0000003ff676c9ee'
P1 = {'sw=225': '0x96', 'sw=220': '0x8c'}


def stress_lines(t):
    """per command index, the STRESS lines between its @@SPEC-BEGIN and @@SPEC-END (or the end of the text)"""
    out = {}
    for m in re.finditer(r'^@@SPEC-BEGIN (\d+) run (\S+) 0$', t, re.M):
        i = int(m.group(1)); rest = t[m.end():]
        e = re.search(r'^@@SPEC-END %d run ' % i, rest, re.M)
        body = rest[:e.start()] if e else rest
        out[i] = (m.group(2), re.findall(r'^STRESS (.*)$', body, re.M), bool(e))
    return out


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
    for i in sorted(W):
        name, ref, _ = W[i]
        if i not in B:
            print(f'{name}: NOT STARTED'); continue
        _, got, ended = B[i]
        bad = [k for k, (a, b) in enumerate(zip(got, ref)) if a != b]
        end = re.search(r'^@@SPEC-END %d run \S+ 0 rc=(\d+) up0=([\d.]+) up1=([\d.]+)' % i, t, re.M)
        if ended and end and end.group(1) == '0' and got == ref:
            print(f'{name}: RETURNED, {len(got)} STRESS lines equal to the twin\'s; {ref[-1]}; '
                  f'span {float(end.group(3)) - float(end.group(2)):.1f} s')
        elif not ended:
            last = got[-1] if got else None
            pk = re.search(r'pkt=(\d+) fast=(\d+)', last or '')
            print(f'{name}: NO RETURN; {len(got)} STRESS lines, mismatches against the twin at {bad or "none"}; '
                  f'last line: {last}')
            if pk:
                print(f'  last heartbeat: packet {pk.group(1)}, {int(pk.group(2)):,} fast allocations before it')
            ap = {'commit pc': re.search(r'commit pc \(.*?\) (0x[0-9a-f]+)', t),
                  'sw=225': re.search(r'^sw=225 .*? (0x[0-9a-f]+) ', t, re.M),
                  'sw=220': re.search(r'^sw=220 .*? (0x[0-9a-f]+) ', t, re.M),
                  'serving': re.search(r'rev_node_serving_idx_ex (0x[0-9a-f]+)', t)}
            ap = {k: (m.group(1) if m else None) for k, m in ap.items()}
            print(f'  apertures {ap}')
            print(f'  commit pc at the stress image\'s CDERIVE {CDERIVE_PC}: {ap["commit pc"] == CDERIVE_PC}; '
                  f'sw=225/sw=220 equal to tsu-p1\'s: {[ap[k] == v for k, v in P1.items()]}')
        else:
            print(f'{name}: OTHER (rc {end.group(1) if end else None}; {len(got)} STRESS lines, '
                  f'mismatches at {bad or "none"}; last {got[-1] if got else None})')


main()
