#!/usr/bin/env python3
"""Turn the request-bounds port (old/sqlite3-capstone.c, sublet-3220000-memsys5.patch applied) into the
block port: every memsys5 block bounded to its whole power-of-two block with stock rounding, sizes and
heap geometry, and every successful realloc a new lifetime, in place or not.
    make-sublet-block-port.py OLD NEW
Each edit must apply exactly once."""
import sys
from pathlib import Path

EDITS = [
    # Remove request-size bookkeeping.
    ("""  /*
  ** Sublet: the request each checked-out block was issued for, rounded to
  ** 8 bytes, by the block's first atom.  It bounds the caller's capability
  ** and is what memsys5Size reports.
  */
  u32 *aReq;

""",
     ''),
    # Report the whole block.
    ("""  iSize = (int)mem5.aReq[i];
""",
     """  iSize = mem5.szAtom * (1 << (mem5.aCtrl[i]&CTRL_LOGSIZE));
"""),
    # Issue the whole block before changing the free lists.
    ("""  /* Return the block as a child of the pool, bounded to the request: the
  ** request rounded to 8, then up to its bounds granule (Q-15), which stays
  ** inside the block because a block is a power-of-two number of atoms. */
  mem5.aReq[i] = (u32)capstone_repr_round(ROUND8(nByte));
  assert( mem5.aReq[i]<=(u32)iFullSz );
  capstoneQ15Note(mem5.aReq[i] - ROUND8(nByte));
  pNew = capstone_cap_derive(mem5.zPool, (unsigned long)i*mem5.szAtom, mem5.aReq[i]);
""",
     """  /* Return the block as a child of the pool, bounded to the whole block,
  ** as stock memsys5 hands it out and reports it.  A block is a power-of-two
  ** number of atoms at an index that is a multiple of its size, and the pool
  ** base carries the heap's granule alignment, so the child is representable
  ** without a pad (Q-15 counts it with pad 0). */
  capstoneQ15Note(0);
  pNew = capstone_cap_derive(mem5.zPool, (unsigned long)i*mem5.szAtom, iFullSz);
"""),
    # Restore the stock power-of-two assertion.
    ("""  assert( pPrior!=0 );
  assert( (nBytes&7)==0 );
  assert( nBytes>=0 );
  if( nBytes==0 ){
    return 0;
  }
  nOld = memsys5Size(pPrior);
  if( nBytes<=nOld ){
    return memsys5Renew(pPrior);
  }
""",
     """  assert( pPrior!=0 );
  assert( (nBytes&(nBytes-1))==0 );  /* EV: R-46199-30249 */
  assert( nBytes>=0 );
  if( nBytes==0 ){
    return 0;
  }
  nOld = memsys5Size(pPrior);
  if( nBytes<=nOld ){
    return memsys5Renew(pPrior);
  }
"""),
    # Restore stock rounding.
    ("""**
** Sublet: the caller's capability is bounded to its request, so the size
** reported back is the request rounded to 8 bytes, as sqlite3MemRoundup
** does; memsys5MallocUnsafe still takes a power-of-two block for it.
*/
static int memsys5Roundup(int n){
  if( n > 0x40000000 ) return 0;
  return ROUND8(n);
}
""",
     """*/
static int memsys5Roundup(int n){
  int iFullSz;
  if( n > 0x40000000 ) return 0;
  for(iFullSz=mem5.szAtom; iFullSz<n; iFullSz *= 2);
  return iFullSz;
}
"""),
    # Restore stock heap geometry.
    ("""  mem5.nBlock = ((nByte-8) / (mem5.szAtom+sizeof(u8)+sizeof(u32)));
""",
     """  mem5.nBlock = (nByte / (mem5.szAtom+sizeof(u8)));
"""),
    # Remove the request-size table.
    ("""  mem5.aReq = (u32 *)&mem5.aCtrl[(mem5.nBlock+3)&~3];
""",
     ''),
]


def main():
    old, new = Path(sys.argv[1]), Path(sys.argv[2])
    text = old.read_text(errors='surrogateescape')
    for before, after in EDITS:
        n = text.count(before)
        if n != 1:
            sys.exit(f'edit applies {n} times, not once:\n{before[:200]}')
        text = text.replace(before, after)
    new.write_text(text, errors='surrogateescape')
    print(f'{len(EDITS)} edits applied')


if __name__ == '__main__':
    main()
