#!/usr/bin/env python3
"""DIAGNOSTIC ONLY: a copy of PoisonCap's published amalgamation with print statements added.

    diag-amalgamation.py SRC_DIR OUT_DIR

Never used for a verdict. It exists to explain individual results (cases 12 and 15 of the temporal
run): which blocks their memsys5 frees and when, when the quarantine is flushed, whether a reallocation
stays in place or moves, and which buffer the FTS5 readers in those cases read. Every insertion is a
print to stderr; nothing else in the source changes, and each anchor must occur exactly once.
"""
from pathlib import Path
import shutil
import sys

HOOKS = [
    # their quarantine: every flush, with the queue length it drains
    ("static void quarantine_flush(void){\n",
     'static void quarantine_flush(void){\n  fprintf(stderr, "DIAG FLUSH q=%zu\\n", q_count);\n'),
    # their memsys5Free: the block it poisons and quarantines
    ("  quarantine_insert(p, size);\n",
     '  quarantine_insert(p, size);\n'
     '  fprintf(stderr, "DIAG FREE base=%#lx size=%zu q=%zu poison_base=%#lx poison_len=%#lx stale_arg_base=%#lx stale_arg_len=%#lx\\n",'
     ' (unsigned long)cheri_getaddress(p), size, q_count, (unsigned long)cheri_getbase(bounded), (unsigned long)cheri_getlen(bounded),'
     ' (unsigned long)cheri_getbase(pPrior), (unsigned long)cheri_getlen(pPrior));\n'),
    # memsys5Realloc: in place, or a move (which frees the old block)
    ("  nOld = memsys5Size(pPrior);\n  if( nBytes<=nOld ){\n    return pPrior;\n  }\n",
     "  nOld = memsys5Size(pPrior);\n  if( nBytes<=nOld ){\n"
     '    fprintf(stderr, "DIAG REALLOC-INPLACE base=%#lx old=%d new=%d\\n", (unsigned long)cheri_getaddress(pPrior), nOld, nBytes);\n'
     "    return pPrior;\n  }\n"
     '  fprintf(stderr, "DIAG REALLOC-MOVE from=%#lx old=%d new=%d\\n", (unsigned long)cheri_getaddress(pPrior), nOld, nBytes);\n'),
    # case 15's read site: the buffer the poslist reader is handed
    ("  i64 *piOff                      /* IN/OUT: Current offset */\n){\n  int i = *pi;\n",
     "  i64 *piOff                      /* IN/OUT: Current offset */\n){\n  int i = *pi;\n"
     '  fprintf(stderr, "DIAG PLNEXT a=%p n=%d i=%d base=%#lx len=%#lx perm_poison=%d\\n", (const void *)a, n, i,'
     ' (unsigned long)cheri_getbase(a), (unsigned long)cheri_getlen(a), (int)((cheri_getperm(a) >> 12) & 1));\n'),
    # case 12: the aSeg array fts5StructureExtendLevel reallocates
    ("    aNew = sqlite3_realloc(pLvl->aSeg, nByte);\n",
     '    fprintf(stderr, "DIAG EXTENDLEVEL aSeg=%p nByte=%d\\n", (void *)pLvl->aSeg, nByte);\n'
     "    aNew = sqlite3_realloc(pLvl->aSeg, nByte);\n"
     '    fprintf(stderr, "DIAG EXTENDLEVEL new=%p\\n", (void *)aNew);\n'),
]


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    text = (src / 'sqlite3.c').read_text()
    for anchor, replacement in HOOKS:
        if text.count(anchor) != 1:
            sys.exit(f'anchor occurs {text.count(anchor)} times, not once: {anchor[:60]!r}')
        text = text.replace(anchor, replacement)
    (out / 'sqlite3.c').write_text(text)
    for name in ('sqlite3.h', 'sqlite3ext.h'):
        shutil.copyfile(src / name, out / name)
    print(f'{len(HOOKS)} diagnostic hooks -> {out}')


if __name__ == '__main__':
    main()
