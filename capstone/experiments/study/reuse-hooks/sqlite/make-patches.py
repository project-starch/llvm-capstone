#!/usr/bin/env python3
"""Generate the SQLite reuse-distance hook patches, one per arm.

  make-patches.py <stock sqlite3-capstone.c> <protected sqlite3-capstone.c> <out dir>

The stock base is the prepared amalgamation (sha256 7fc92e19...); the protected
base is the same file after ../../../../ports/sqlite/sublet/sublet-3220000-memsys5.patch.
The same insertions, at the same anchors, are made in both; the anchors are
lines the lifetime patch does not change, so the two patches differ only in
line numbers and in the context lines next to the lookaside returns, which the
lifetime patch rewrites. Every anchor must match the expected number of times
or the script stops.
"""
import difflib
import sys
from pathlib import Path

HOOK = '#ifdef EXP_REUSE_GAP\n{}\n#endif\n'

# (anchor line, expected count, where, inserted body)
EDITS = [
    ('/************** Begin file mem5.c ********************************************/\n', 1,
     'before', '#include "exp-reuse-gap.h"'),
    # memsys5MallocUnsafe: the block is taken; i is its first atom, iFullSz its size.
    ('  mem5.aCtrl[i] = iLogsize;\n', 1, 'after',
     '  exp_gap_issue(&exp_gap_memsys5, (uint64_t)(uintptr_t)&mem5.zPool[i*mem5.szAtom],\n'
     '                (uint64_t)iFullSz);'),
    # memsys5FreeUnsafe: iBlock is the block's first atom, before any merge.
    ('  iBlock = (int)(((u8 *)pOld-mem5.zPool)/mem5.szAtom);\n', 1, 'after',
     '  exp_gap_release(&exp_gap_memsys5,\n'
     '                  (uint64_t)(uintptr_t)&mem5.zPool[iBlock*mem5.szAtom]);'),
    # sqlite3DbMallocRawNN: a slot from pFree, then one from pInit.
    ('      db->lookaside.anStat[0]++;\n', 2, 'after',
     '      exp_gap_issue(&exp_gap_lookaside, (uint64_t)(uintptr_t)pBuf,\n'
     '                    (uint64_t)db->lookaside.sz);'),
    # sqlite3DbFreeNN: the slot goes back on the free list.
    ('      pBuf->pNext = db->lookaside.pFree;\n', 1, 'before',
     '      exp_gap_release(&exp_gap_lookaside, (uint64_t)(uintptr_t)pBuf);'),
]


def instrument(lines):
    out = []
    counts = [0] * len(EDITS)
    for line in lines:
        hits = [k for k, e in enumerate(EDITS) if line == e[0]]
        assert len(hits) <= 1, line
        if hits:
            k = hits[0]
            counts[k] += 1
            block = HOOK.format(EDITS[k][3]).splitlines(keepends=True)
            if EDITS[k][2] == 'before':
                out += block + [line]
            else:
                out += [line] + block
        else:
            out.append(line)
    for k, e in enumerate(EDITS):
        if counts[k] != e[1]:
            sys.exit(f'anchor {e[0]!r}: {counts[k]} matches, expected {e[1]}')
    return out


def main():
    stock, protected, out = map(Path, sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=True)
    for arm, base in (('stock', stock), ('protected', protected)):
        lines = base.read_text().splitlines(keepends=True)
        diff = difflib.unified_diff(lines, instrument(lines),
                                    'a/sqlite3-capstone.c', 'b/sqlite3-capstone.c', n=3)
        (out / f'hooks-{arm}.patch').write_text(''.join(diff))


if __name__ == '__main__':
    main()
