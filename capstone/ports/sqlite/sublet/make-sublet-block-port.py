#!/usr/bin/env python3
"""Turn the request-bounds port (old/sqlite3-capstone.c, sublet-3220000-memsys5.patch applied) into the
block port: every memsys5 block bounded to its whole power-of-two block with stock rounding, sizes and
heap geometry, and every successful realloc a new lifetime, in place or not.
    make-sublet-block-port.py OLD NEW
Each edit must apply exactly once."""
import sys
from pathlib import Path

EDITS = [
    # The aReq table goes: block bounds need no record of the request.
    ("""  /*
  ** Sublet: the request each checked-out block was issued for, rounded to
  ** 8 bytes, by the block's first atom.  It bounds the caller's capability
  ** and is what memsys5Size reports.
  */
  u32 *aReq;

""", ""),
    # memsys5Size: stock, the block.
    ("  iSize = (int)mem5.aReq[i];\n",
     "  iSize = mem5.szAtom * (1 << (mem5.aCtrl[i]&CTRL_LOGSIZE));\n"),
    # memsys5MallocUnsafe: the child is the whole block.
    ("""  /* Return the block as a child of the pool, bounded to the request: the
  ** request rounded to 8, then up to its bounds granule (Q-15), which stays
  ** inside the block because a block is a power-of-two number of atoms. */
  mem5.aReq[i] = (u32)capstone_repr_round(ROUND8(nByte));
  assert( mem5.aReq[i]<=(u32)iFullSz );
  capstoneQ15Note(mem5.aReq[i] - ROUND8(nByte));
  return capstone_cap_derive(mem5.zPool, (unsigned long)i*mem5.szAtom, mem5.aReq[i]);
""", """  /* Return the block as a child of the pool, bounded to the whole block,
  ** as stock memsys5 hands it out and reports it.  A block is a power-of-two
  ** number of atoms at an index that is a multiple of its size, and the pool
  ** base carries the heap's granule alignment, so the child is representable
  ** without a pad (Q-15 counts it with pad 0). */
  capstoneQ15Note(0);
  return capstone_cap_derive(mem5.zPool, (unsigned long)i*mem5.szAtom, iFullSz);
"""),
    # memsys5Realloc: stock assertion, and the in-place branch renews the lifetime.
    ("""  assert( pPrior!=0 );
  assert( (nBytes&7)==0 );
  assert( nBytes>=0 );
  if( nBytes==0 ){
    return 0;
  }
  nOld = memsys5Size(pPrior);
  if( nBytes<=nOld ){
    return pPrior;
  }
""", """  assert( pPrior!=0 );
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
    # memsys5Renew, before memsys5Realloc's comment block.
    ("""/*
** Change the size of an existing memory allocation.
**
** The outer layer memory allocator prevents this routine from
""", """/*
** Sublet: a realloc that keeps its block still ends the old allocation, as
** realloc's contract says, so the block goes out again as a new child of
** the pool: every copy of the old pointer faults, and the returned pointer
** reaches the same bytes at the same address.  The old child is revoked
** before the new one is derived, so the pool's children stay disjoint; the
** new one is a sibling, not a child of the old one.  With
** SUBLET_REALLOC_KEEPS_LIFETIME (the ablation) the old child is returned.
*/
static void *memsys5Renew(void *pPrior){
#ifdef SUBLET_REALLOC_KEEPS_LIFETIME
  return pPrior;
#else
  int i, iSize;
  void *p;
  memsys5Enter();
  i = (int)(((u8 *)pPrior-mem5.zPool)/mem5.szAtom);
  iSize = memsys5Size(pPrior);
  capstone_cap_revoke_child(mem5.zPool, pPrior);
  p = capstone_cap_derive(mem5.zPool, (unsigned long)i*mem5.szAtom, iSize);
  memsys5Leave();
  return p;
#endif
}

/*
** Change the size of an existing memory allocation.
**
** The outer layer memory allocator prevents this routine from
"""),
    # memsys5Roundup: stock.
    ("""**
** Sublet: the caller's capability is bounded to its request, so the size
** reported back is the request rounded to 8 bytes, as sqlite3MemRoundup
** does; memsys5MallocUnsafe still takes a power-of-two block for it.
*/
static int memsys5Roundup(int n){
  if( n > 0x40000000 ) return 0;
  return ROUND8(n);
}
""", """*/
static int memsys5Roundup(int n){
  int iFullSz;
  if( n > 0x40000000 ) return 0;
  for(iFullSz=mem5.szAtom; iFullSz<n; iFullSz *= 2);
  return iFullSz;
}
"""),
    # memsys5Init: stock geometry.
    ("""  mem5.nBlock = ((nByte-8) / (mem5.szAtom+sizeof(u8)+sizeof(u32)));
""", """  mem5.nBlock = (nByte / (mem5.szAtom+sizeof(u8)));
"""),
    ("""  mem5.aReq = (u32 *)&mem5.aCtrl[(mem5.nBlock+3)&~3];
""", ""),
    # The lookaside's realloc in place renews the slot's lifetime the same way.
    ("""static LookasideSlot *lookasideRelease(sqlite3 *db, void *p){
  capstone_cap_revoke_child(db->lookaside.pLife, p);
  return (LookasideSlot*)&((u8*)db->lookaside.pStart)[(u8*)p-(u8*)db->lookaside.pStart];
}
""", """static LookasideSlot *lookasideRelease(sqlite3 *db, void *p){
  capstone_cap_revoke_child(db->lookaside.pLife, p);
  return (LookasideSlot*)&((u8*)db->lookaside.pStart)[(u8*)p-(u8*)db->lookaside.pStart];
}
/* A realloc that keeps its slot ends the old lifetime and issues the slot
** again (memsys5Renew says why); the ablation returns the old child. */
static void *lookasideRenew(sqlite3 *db, void *p){
#ifdef SUBLET_REALLOC_KEEPS_LIFETIME
  return p;
#else
  return lookasideIssue(db, lookasideRelease(db, p));
#endif
}
"""),
    ("  if( isLookaside(db,p) && n<=db->lookaside.sz ) return p;\n",
     "  if( isLookaside(db,p) && n<=db->lookaside.sz ) return lookasideRenew(db, p);\n"),
    # SQLITE_DEBUG trashes a freed slot through the lookaside's own authority, not the revoked child.
    ("      memset(p, 0xaa, db->lookaside.sz);\n",
     "      memset(pBuf, 0xaa, db->lookaside.sz);\n"),
    # sqlite3Realloc: no shortcut for an unchanged size; xRealloc renews the lifetime.
    ("""  nNew = sqlite3GlobalConfig.m.xRoundup((int)nBytes);
  if( nOld==nNew ){
    pNew = pOld;
  }else if( sqlite3GlobalConfig.bMemstat ){
""", """  nNew = sqlite3GlobalConfig.m.xRoundup((int)nBytes);
  /* Sublet: no shortcut when the rounded size is unchanged.  The realloc
  ** still ends the old allocation, so it goes to xRealloc, which issues the
  ** block again (memsys5Renew).  The ablation keeps the shortcut. */
#ifdef SUBLET_REALLOC_KEEPS_LIFETIME
  if( nOld==nNew ){
    pNew = pOld;
  }else
#endif
  if( sqlite3GlobalConfig.bMemstat ){
"""),
    # The lookaside buffer's memsys5 block is now a power of two at least sz*cnt.
    ("""    /* Q-15: sz*cnt rounded up to its granule (tail pad only), inside pStart,
    ** whose memsys5 block was rounded the same way. */
""", """    /* Q-15: sz*cnt rounded up to its granule (tail pad only), inside pStart,
    ** whose memsys5 block is a power of two at least that large. */
"""),
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
