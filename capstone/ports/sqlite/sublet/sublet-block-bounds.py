#!/usr/bin/env python3
"""Turn the Sublet memsys5 port (sublet-3220000-memsys5.patch, already applied) back to stock rounding.

    sublet-block-bounds.py SOURCE      (the patched sqlite3-capstone.c, edited in place)

A comparison arm, not the port. The port bounds each block child to its request (ROUND8, then the
representable size) and has memsys5 report that size, so SQLite sees a reallocation whenever a request
grows. Stock memsys5 rounds every request up to a power-of-two block and reports the block, so a request
that still fits stays in place without SQLite calling realloc at all. That is the rounding PoisonCap's
memsys5 keeps, and case 12 of the engine corpus only frees its array when the request rounding applies.

This arm keeps every Sublet lifetime (CDERIVE per block, CREVOKE at free, the pool, the lookaside) and
changes only the rounding back: the child is bounded to the whole block, memsys5Size and memsys5Roundup
report the block, and memsys5Realloc's assertion is stock again. Each edit must apply exactly once.
"""
import sys

EDITS = [
    ("  mem5.aReq[i] = (u32)capstone_repr_round(ROUND8(nByte));\n",
     "  mem5.aReq[i] = (u32)iFullSz;   /* block-bounds arm: the whole power-of-two block */\n"),
    ("  capstoneQ15Note(mem5.aReq[i] - ROUND8(nByte));\n",
     "  capstoneQ15Note(0);\n"),
    ("  assert( (nBytes&7)==0 );\n",
     "  assert( (nBytes&(nBytes-1))==0 );  /* EV: R-46199-30249 */\n"),
    ("  if( n > 0x40000000 ) return 0;\n  return ROUND8(n);\n",
     "  int iFullSz;\n  if( n > 0x40000000 ) return 0;\n"
     "  for(iFullSz=mem5.szAtom; iFullSz<n; iFullSz *= 2);\n  return iFullSz;\n"),
]


def main():
    path = sys.argv[1]
    text = open(path).read()
    for old, new in EDITS:
        if text.count(old) != 1:
            sys.exit(f'{path}: expected exactly one {old.strip()!r}, found {text.count(old)}')
        text = text.replace(old, new)
    open(path, 'w').write(text)
    print(f'block-bounds: {len(EDITS)} edits applied to {path}')


if __name__ == '__main__':
    main()
