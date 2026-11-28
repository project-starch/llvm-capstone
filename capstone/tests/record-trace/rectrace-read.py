#!/usr/bin/env python3
"""rectrace-read.py TRACE [--text [--from N] [--lines M]] [--summary]

Reader for the record-access trace of the tracing twin QEMU (cap_rev_trace.h; nested lane, 2026-11-20).
TRACE is a file of little-endian u64 words, one per run of identical events, in execution order:
  bits 63:56 code = kind << 2 | op;  bits 55:32 count;  bits 31:0 slot (header ops: byte offset of the header word)
  a root marker (kind 7) carries root >> 4 in bits 55:0;
  an instruction-count marker (kinds 8, 9) carries in bits 55:0 the guest instructions executed so far: kind 8 all
  of them, kind 9 those of protected code. A pair is written every 1,024 runs and at the end; it places the run
  that follows it.
  a context-edge word (kind 10): op 0 a trap out of protected code (slot = cause, bit 31 for an interrupt),
  op 2 an SRET into it, op 3 an MRET into it; count 1.
kind: 0 c check in U, 1 k check outside U, 2 p PCC check, 3 a allocation, 4 r release, 5 v release's read-only
pass (model only), 6 o other operation, 7 T root marker.  op: r record read, w record write, hr/hw header read/write.

--summary (default): per kind and op, events (sum of counts), runs (lines), distinct slots; root markers.
--text: one line per run, `<kind><op> <slot> <count>`, or `T <root hex>`, `I <all>`, `P <protected>`;
  --from/--lines select a window.
Exits 2 on an empty or malformed trace: no data is an error, not a zero."""
import sys
import numpy as np
KIND = 'ckparvoT'
OP = ['r', 'w', 'hr', 'hw']
def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    path = a[0]
    text = '--text' in a
    start = int(a[a.index('--from') + 1]) if '--from' in a else 0
    nlines = int(a[a.index('--lines') + 1]) if '--lines' in a else None
    import os
    size = os.path.getsize(path)
    if size == 0 or size % 8:
        print(f'{path}: {size} bytes is not a non-empty multiple of 8', file=sys.stderr); sys.exit(2)
    w = np.memmap(path, dtype='<u8', mode='r')
    if text:
        end = len(w) if nlines is None else min(len(w), start + nlines)
        out = []
        for x in w[start:end].tolist():
            code = x >> 56
            kind, op = code >> 2, code & 3
            if kind == 7:
                out.append('T %x' % ((x & ((1 << 56) - 1)) << 4))
            elif kind in (8, 9):
                out.append('%s %d' % ('IP'[kind - 8], x & ((1 << 56) - 1)))
            elif kind == 10:
                out.append('E%s %x' % ({0: 'trap', 2: 'sret', 3: 'mret'}.get(op, str(op)), x & 0xffffffff))
            else:
                out.append('%s%s %d %d' % (KIND[kind], OP[op], x & 0xffffffff, (x >> 32) & 0xffffff))
            if len(out) >= 100000:
                sys.stdout.write('\n'.join(out) + '\n'); out = []
        if out:
            sys.stdout.write('\n'.join(out) + '\n')
        return
    print(f'{path}: {len(w):,} words')
    ev = {}; runs = {}; slots = {}
    roots = []; marks = {}; edges = {}
    CH = 1 << 24
    for i in range(0, len(w), CH):
        c = np.asarray(w[i:i + CH])
        code = (c >> np.uint64(56)).astype(np.int64)
        cnt = ((c >> np.uint64(32)) & np.uint64(0xffffff)).astype(np.int64)
        slot = (c & np.uint64(0xffffffff)).astype(np.int64)
        for cd in np.unique(code).tolist():
            m = code == cd
            if cd >> 2 == 7:
                roots.extend(((c[m] & np.uint64((1 << 56) - 1)) << np.uint64(4)).tolist())
                continue
            if cd >> 2 in (8, 9):
                marks[cd >> 2] = (marks.get(cd >> 2, (0, 0))[0] + int(m.sum()), int((c[m][-1]) & np.uint64((1 << 56) - 1)))
                continue
            if cd >> 2 == 10:
                edges[cd & 3] = edges.get(cd & 3, 0) + int(m.sum())
                continue
            if cd >> 2 > 10:
                print(f'malformed code {cd} near word {i}', file=sys.stderr); sys.exit(2)
            ev[cd] = ev.get(cd, 0) + int(cnt[m].sum())
            runs[cd] = runs.get(cd, 0) + int(m.sum())
            slots.setdefault(cd, set()).update(np.unique(slot[m]).tolist())
    print('root markers: %d, distinct roots: %d %s' % (len(roots), len(set(roots)), ' '.join('%x' % r for r in sorted(set(roots))[:8])))
    for k, name in ((8, 'all'), (9, 'protected')):
        if k in marks:
            print('instruction-count markers (%s): %d, last value %s' % (name, marks[k][0], f'{marks[k][1]:,}'))
    if edges:
        print('context edges: trap-out %d, sret-in %d, mret-in %d' % (edges.get(0, 0), edges.get(2, 0), edges.get(3, 0)))
    print('%-4s %16s %14s %12s %10s' % ('code', 'events', 'runs', 'slots', 'ev/run'))
    for cd in sorted(ev):
        print('%-4s %16s %14s %12s %10.2f' % (KIND[cd >> 2] + OP[cd & 3], f'{ev[cd]:,}', f'{runs[cd]:,}', f'{len(slots[cd]):,}', ev[cd] / runs[cd]))
    if not ev:
        print('no events', file=sys.stderr); sys.exit(2)
main()
