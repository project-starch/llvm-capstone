# c7def600bd_0 -- c7def600bd

Upstream fix `c7def600bd`. Collected in round R3 (backport filter).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
c7def600bd_0 -- upstream corrupt-database regression image, backported to 3.22.0.
 *
 * Collected by the backport filter: upstream fix c7def600bd removes code that is still
 * present in 3.22.0 (or guards a function 3.22.0 has unguarded), and its own regression
 * test ships the triggering database as a hex dump. Host ASan on stock 3.22.0:
 *
 *     heap-buffer-overflow in fts3EvalUpdateCounts
 *
 * The image is embedded and served by repro322_memfs.c, an in-domain in-memory FILESYSTEM.
 * ext/misc/memvfs.c cannot be used here: it serves only the main database, so the pager
 * cannot create a rollback journal, and it takes its buffer as an INTEGER in the URI --
 * a pointer forged from an integer, which on a capability machine is untagged and faults
 * on first use.
 *
 * STATEMENT BY STATEMENT, deliberately. Handing the whole script to one sqlite3_exec()
 * stops at the first error, and these scripts routinely take SQLITE_CORRUPT on an early
 * statement while the defect is reached by a later one. Screening the whole script in one
 * call scored 3 of 17; per statement it scores 10 of 17, matching a real file exactly.
 *
 * REACHABILITY: every statement's rc is printed. SQLITE_CORRUPT (11) on the statement that
 * should reach the defect means SQLite rejected the image first -- a real negative, not a
 * silent pass.
```

## Review 2026-10-16 -- holds in full, and the address judgement is too coarse to see it

### The four readings

| reading | result |
|---|---|
| host ASan | re-run: `heap-buffer-overflow`, 0 bytes after a 16-byte region, `fts3EvalUpdateCounts <- fts3EvalGatherStats <- sqlite3Fts3EvalPhraseStats <- fts3ExprGlobalHitsCb` -- identical to the recorded bundle |
| virtual-malloc | **MISSED**, `RETURNED`, controls as declared |
| virtual-nested-pools | **CAUGHT**, `cause=28` offset 360 in `fts3EvalUpdateCounts`, fault address `0x3fafaf57a0` |
| cheribsd-revocation | **MISSED**, every statement `rc=0`, `RETURNED rc=0`, positive control trapping |

### The walk has no bound in it at all

```c
char *p = pPhrase->doclist.pList;
assert( *p );
while( 1 ){
  u8 c = 0;
  int iCnt = 0;
  while( 0xFE & (*p | c) ){
    if( (c&0x80)==0 ) iCnt++;
    c = *p++ & 0x80;
  }
  ...
  if( *p==0x00 ) break;
```

It stops only at a zero byte. `pPhrase->doclist.nList` -- the length, in the same structure --
is never read.

### The address judgement cannot witness this case

Probe site 63 prints the entry and hands each walked byte to the judgement. On CheriBSD:

```
LB_OBJECT site=63 ENTER   pList=0x2433c1 nList=3 (the loop below consults no length)
LB_SUMMARY site=63 hits=8 witnesses=0 first_witness_at_hit=0
```

`pList=0x2433c1` is an **interior** pointer: the doclist is a sub-range of a larger memsys5
block, not a block of its own. The probe asks memsys5 which block an address belongs to and
compares against *that block's* requested size, so reads a few bytes past a 3-byte doclist are
still inside the enclosing block and nothing is reported.

That is a fourth distinct form of the blind spot, and the most fundamental of them: not a
reissue (cases 07, 21), not a misclassification (12, 21, 22, 25), but **the oracle's granularity
being coarser than the object**. An allocator-level oracle cannot bound a slice of an
allocation.

It is **not** evidence that no over-read happened. `nList=3` with eight walk iterations means at
least five bytes were read past the doclist, which is what the ASan reading says independently:
0 bytes after a 16-byte region, where on the host the doclist is its own allocation.

### How far the walk runs is decided by the following bytes

On CheriBSD it met a zero byte after eight iterations and stopped while still inside the
enclosing block, so that arm completes with every statement `rc=0`. On the Capstone Sublet image
the following bytes were not zero, the walk left the block, and Sublet's bounds check fired with
`cause 28`.

This is the spatial counterpart of what case 24 showed for the temporal family: the *distance*
of the over-read is data-dependent. Both arms without per-sub-allocation bounds are blind either
way -- CheriBSD's bounds and revocation are at `malloc` granularity and the whole arena is one
`malloc` -- so their silence is a genuine reached-and-silent. But the walk is unbounded in
principle and bounded only by accident in any given run, which is worth recording rather than
treating it as if it stopped where this image happened to stop.
