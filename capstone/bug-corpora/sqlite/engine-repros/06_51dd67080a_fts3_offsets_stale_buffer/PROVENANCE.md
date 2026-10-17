# fts3offsets -- 51dd67080a

Upstream fix `51dd67080a`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row13 / sqlite-51dd67080a -- sqlite3Fts3Offsets() stores a poslist pointer per
 * phrase, then an incremental phrase triggers a NEAR-group restart (fts3EvalRestart,
 * fts3.c:5540) that frees/reloads already-visited doclists, dangling the saved
 * pointers (fts3_snippet.c:1544). Fixed 3.49.0. CONTROL: offsets() over an OR of
 * phrases read incrementally; on unprotected capstone completes -> NOTRAP.
 * NOTE: build in fts3 group.
```

## The trigger does not reach it, and the recorded reason was wrong (2026-10-16)

The 2026-10-11 note said `sqlite3Fts3Offsets` runs twice and all three `fts3EvalRestart` calls
come after both of them. Probe site 45 marks three points in one run -- the save
(`fts3ExprTermOffsetInit`, where a fresh poslist pointer is taken), the invalidate
(`fts3EvalInvalidatePoslist` inside `fts3EvalRestart`) and the read (`*pTerm->pList` in
`sqlite3Fts3Offsets`) -- and the order is:

    SAVE x2 -> 6 reads -> SAVE -> RESTART -> SAVE -> 4 reads

One restart, not three, and it falls inside the second call rather than after both.

**What makes the trigger harmless is the save that follows the restart.** Every group of reads is
preceded by its own `fts3ExprTermOffsetInit`, so the four reads after the restart use pointers
taken after it, not the ones it invalidated. There is no read across a restart, which is why host
ASan is silent and why all three arms report MISSED: none of them is a miss.

Identical line for line on native memsys5 and on virtual Capstone with lookaside off.

The correction matters for anyone trying to write a trigger that does reach this defect: it is not
enough to move a restart earlier, it has to fall between a save and a read of the same pointer.
Upstream `51dd67080a` makes that impossible from the other side, cancelling the incremental phrases
before the pointers are taken (`sqlite3Fts3MsrCancel` from a new `fts3ExprRestartIfCb`).

The correction above stands as the record of the earlier trigger. What follows replaces its
conclusion: the trigger was rewritten rather than withdrawn.

## 2026-10-16: the trigger was replaced and the defect is now reached

The previous section established that the recorded trigger could not reach this defect, and why --
`fts3EvalRestart` walks the subtree of `pNear`, the phrase's most senior NEAR ancestor, so a flat
`"a" OR "b"` gives every phrase `pNear` = itself and a restart can only free the poslist of the
phrase currently being fetched, whose address is stored after that fetch returns. Three facts in
the 3.22.0 evaluator then say what a trigger has to look like:

1. the freed phrase and the phrase whose pointer was saved must share one NEAR group, since that
   subtree is all a restart touches;
2. the restart branch of `sqlite3Fts3EvalPhrasePoslist` is reached only below an OR (`bOr`) and
   only for a phrase that is incremental and not positioned on the current row;
3. `fts3EvalTestExpr` zeroes the poslists of a NEAR group that does not match the current row
   before any auxiliary function runs, which would leave nothing to save -- unless an AND operand
   to its left fails first, because `bHit = (TestExpr(pLeft) && TestExpr(pRight) && NearTest())`
   short-circuits and never descends into the NEAR node.

Hence `(xx AND ("pp1 pp2" NEAR/3 qq)) OR rr` over two rows. At row 2, `xx` is exhausted so the
cleanup is skipped, `"pp1 pp2"` is on the row and its poslist address is saved in `aTerm[]`, `qq`
is exhausted and incremental so its fetch restarts the group and the recursion frees that buffer,
and `rr` is why row 2 is visited at all. The query needs `-DSQLITE_ENABLE_FTS3_PARENTHESIS`, so
this case moves to the `fts3P` group in both builders, beside case 07.

Host ASan (prepared amalgamation, system allocator, lookaside off, 4 GiB quarantine):

```
READ of size 1, 1 bytes inside of 8-byte region
  use   sqlite3Fts3Offsets                      sqlite3.c:166213
  free  fts3EvalInvalidatePoslist <- fts3EvalRestart <- fts3EvalRestart
          <- sqlite3Fts3EvalPhrasePoslist <- fts3ExprTermOffsetInit <- fts3ExprIterate2
  alloc fts3EvalIncrPhraseNext                  the merged poslist of the two-token phrase
```

The two `fts3EvalRestart` frames are the recursion into the sibling phrase: the free is not the
fetched phrase's own buffer, which is all the earlier trigger could ever reach.

Probe site 45 now prints the pointer at all three points -- the save in `fts3ExprTermOffsetInit`
after the fetch returns, the invalidate inside `fts3EvalRestart`, and the read in
`sqlite3Fts3Offsets` -- and the same trace comes back on native memsys5, on virtual Capstone with
lookaside off, and on CheriBSD purecap:

```
SAVE    aTerm[] takes pList p=<base>+1
RESTART frees poslist p=<base> bFreeList=1
RESTART frees poslist p=0x0 bFreeList=0        the fetched phrase's own, already empty
OFFSETS reads saved pList p=<base>+1           twice
OFFSETS reads saved pList p=<base>+2
```

The one-byte offset is not a discrepancy: `fts3ExprTermOffsetInit` consumes the first varint with
`fts3GetDeltaPosition` before storing the pointer, so the saved address is one past the block base
that is freed -- the same 8-byte block, at the offsets host ASan names.

`LB_SUMMARY` reports `witnesses=0` on every arm. That is the blind spot case 05 documented, not a
contradiction: the restart frees the buffer and immediately reloads the phrase, memsys5 reissues
the block to that reload, and an address judgement asked at the read therefore finds a live block.
The pointer values are the evidence here, and they are identical across three platforms.

What the unprotected arms do with the stale read is worth recording. It completes, and the garbage
it returns walks the position-list iterator off the end, so `offsets()` fails and the row comes
back as an error instead of a row: `rows=1 row1_expected=1 last_step_rc=11`, SQLITE_CORRUPT,
"database disk image is malformed" -- on a `:memory:` database built three statements earlier. The
defect surfaces as data corruption, not as a crash.

Arms, measured 2026-10-16 on this platform:

| arm | verdict | evidence |
| --- | --- | --- |
| host ASan | reports | as above |
| virtual-malloc | MISSED | both controls as declared; reached, read, `last_step_rc=11`, RETURNED |
| virtual-nested-pools | CAUGHT | both controls as declared; `cause=25 pc=0x3fa026a73c in sqlite3Fts3Offsets`, attributed by function to the site host ASan reported |
| cheribsd-revocation | MISSED | positive control traps (`In-address space security exception`); the case RETURNS rc=0 with `last_step_rc=11` |

CheriBSD's revocation has nothing to revoke: the freed 8-byte block is a sub-allocation inside the
memsys5 arena, which stays live, and the pointer's bounds are the arena's -- measured directly for
this allocator in cases 02, 03 and 04, so it is not re-measured here.

A negative control settles the direction of the mechanism rather than just its presence: with the
NEAR operands swapped, `(xx AND (qq NEAR/3 pp)) OR rr`, the restart happens before the save instead
of after it, and host ASan is silent. A build without `SQLITE_ENABLE_FTS3_PARENTHESIS` returns zero
rows and the case says so, so it cannot pass vacuously; row 1's `offsets()` is compared against a
fixed string for the same reason.

This case is no longer proposed for withdrawal. It is distinct from case 07, which shares the
restart site but dangles the 208-byte `Fts3SegReader` freed by `fts3SegReaderCursorFree` and reads
it through `fts3SnippetAdvance`; here the object is the 8-byte merged poslist freed by
`fts3EvalInvalidatePoslist` and the read is in `sqlite3Fts3Offsets`, which is what upstream fixed
separately in 3.49.0. Case 09 is the same family and its trigger has the same flat-OR shape, so
its review has to settle whether it is reachable at all or is case 07 again.
