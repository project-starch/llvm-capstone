# fts5near -- fix-2026-06-08

Upstream fix `2026-06-08`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row22 / sqlite-2026-06-08 (b677c5afd4) -- fts5ExprNearIsMatch reads a lookahead
 * reader from each poslist buffer while WriterAppend rewrites the same buffer in
 * place; the append may realloc, so the reader reads freed memory (fts5_expr.c
 * 560/596/627). Fixed 3.53.3. CONTROL: run a NEAR/phrase query that drives
 * fts5ExprNearIsMatch. On unprotected capstone completes -> NOTRAP.
 * NOTE: build in fts5 group.
```

## 2026-10-16: the trigger was replaced and the defect is now reached

The 2026-10-11 note framed the open question exactly: the in-place rewrite does run under live
lookahead readers, but whether any append grows the buffer past its allocation -- the reallocation
the stale read needs -- was not established. It is established now, and the answer is that the
recorded trigger never could.

**The mechanism, and what the code's own comment misses.** `fts5ExprNearIsMatch` points a reader at
each phrase's poslist buffer and then writes the trimmed poslist back into that same buffer:

```c
fts5LookaheadReaderInit(pPoslist->p, pPoslist->n, &a[i].reader);   /* reader keeps pPoslist->p */
pPoslist->n = 0;
a[i].pOut = pPoslist;                                              /* writer appends to the same */
...
sqlite3Fts5PoslistWriterAppend(a[i].pOut, pWriter, iPos);
```

The comment above that loop argues the rewrite is safe "as the entries for the new poslist are a
subset of the old, so it is not possible for data yet to be read to be overwritten". That covers
overwriting and not reallocation, and `sqlite3Fts5PoslistWriterAppend` opens with

```c
if( fts5BufferGrow(&rc, pBuf, 5+5+5) ) return rc;
```

which reaches `sqlite3Fts5BufferSize` and `sqlite3_realloc` as soon as `n + 15 > nSpace`. (The
earlier record said upstream's fix replaces this call with `sqlite3Fts5PoslistSafeAppend`; in this
pin `WriterAppend` already is a grow followed by `SafeAppend`, so the defect is the grow.)

**Why the recorded trigger could not reach it.** `nSpace` doubles from 64 and the retained poslist
is a subset of the original, so `n_new <= n_old <= nSpace`: the realloc fires only when the retained
length lands in the top 15 bytes of a bucket. The recorded trigger indexed two rows of ten-odd
words, a poslist of a few bytes against `nSpace` 64. Sweeping one row of repeated "aa bb" under host
ASan maps the condition exactly:

| repeats | 50 | 100 | 118 | 125 | 150 | 200 | 245 | 250 | 255 | 400 | 500 | 505 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reports | . | . | X | X | . | . | X | X | X | . | . | X |

which is the windows (49,64], (113,128], (241,256] and (497,512]. The replacement indexes four
rows, one per window, so the case does not depend on the exact encoded length.

Host ASan on the replacement:

```
READ of size 1, 52 bytes inside a freed 64-byte region
  use   sqlite3Fts5PoslistNext64 sqlite3.c:189645 <- fts5LookaheadReaderNext
          <- fts5ExprNearIsMatch <- fts5ExprNearTest <- fts5ExprNodeTest_STRING
  free  realloc <- sqlite3MemRealloc <- sqlite3_realloc <- sqlite3Fts5BufferSize
          <- sqlite3Fts5PoslistWriterAppend
```

**Probe site 49** marks the reader init, the grow (old pointer, new pointer, `n`, `nSpace`,
requested) and the read. On native memsys5: 8 reader inits, 46 grows, **8 of which move a buffer a
reader was initialised on, and 68 reads from a moved-away buffer -- all 68 witnessed as
`kind=UAF off=52 n=1 req=64`**. The first stale read follows the grow with `n=50 nSpace=64 want=65`,
which is the `n + 15 > nSpace` condition verbatim.

| platform | witnesses | reading |
| --- | --- | --- |
| native memsys5 | 68 of 1892 hits | `kind=UAF off=52 n=1 req=64` |
| virtual Capstone, lookaside off | 68 | identical offset and size |
| CheriBSD purecap | 68 | identical, plus `tag=1 cb=<arena>+262144` |

Unlike cases 05, 06 and 09 the freed block is not reissued before the read, so here the address
judgement gets both the fact and the kind right. The CheriBSD line also measures why that arm is
silent: the pointer is tagged and bounded to the whole 256 KiB memsys5 arena, so the access is in
bounds and the realloc's free is nothing revocation can see.

Arms, measured 2026-10-16 on this platform:

| arm | verdict | evidence |
| --- | --- | --- |
| host ASan | reports | as above |
| virtual-malloc | MISSED | both controls as declared; 68 witnessed stale reads, then rows=4 and RETURNED |
| virtual-nested-pools | CAUGHT | both controls as declared; `cause=25 pc=0x3f90a37854 in sqlite3Fts5PoslistNext64`, attributed by function |
| cheribsd-revocation | MISSED | positive control traps; rows=4, RETURNED rc=0, 68 witnesses |

No count in the study's tables moves. This arm's Sublet cell was already detected, but it rested on
one of the 28 halts at kernel pc 0x800239e4 and now rests on a cause-25 fault at the declared site.
The three physical rows in `results/matrix.tsv` measured the old trigger and are marked superseded.

The case's own reachability markers are "indexed" after the inserts and "scanning" before the first
step, because the stale read happens inside that first step, before any row is returned.
