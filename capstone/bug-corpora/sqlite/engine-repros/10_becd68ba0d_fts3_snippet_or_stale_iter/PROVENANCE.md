# fts3snipor -- becd68ba0d

Upstream fix `becd68ba0d`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row6 / sqlite-becd68ba0d -- fts3EvalNextRow() nested-OR branch keeps evaluating
 * phrase nodes whose doclist was freed because bEof was not set on the exhausted
 * side (fts3.c:5134). Fixed 3.32.0. CONTROL: snippet() over an expression with
 * nested OR phrases; on unprotected capstone completes -> NOTRAP.
 * NOTE: build in fts3 group.
```

## 2026-10-11: the trigger is the CheriBSD copy

`case.c` is now byte-identical to `ports/sqlite/cheribsd/cases/` (the copy the CheriBSD probe runs used).
The source above is the earlier trigger. Why it was replaced: the corpus copy queried a flat OR and returned 3 rows; host ASan was silent on it. The CheriBSD copy runs upstream's fts3snippet2.test 2.1/2.2 nested-OR query, checks the snippet value and warns when no row comes back, and host ASan reports the heap-use-after-free in fts3SnippetAdvance. Evidence:
`results/2026-10-11-host-asan/results.json`, from `shared/run-host-asan.py` (every SQLite object its own
malloc, lookaside off).

## 2026-10-16: re-measured, and the mechanism is in the AND/NEAR branch

Every recorded verdict holds. What this corrects is where the defect lives, and it adds a probe so
the arms' behaviour is read rather than inferred.

**The branch.** The case recorded "fts3EvalNextRow()'s nested-OR branch advances the exhausted side
without setting bEof, ... fts3.c:5134". In the 3.22.0 tree, `fts3.c:5134` is the declaration line of
`fts3EvalNextRow` itself, and its OR branch does set `pExpr->bEof = (pLeft->bEof && pRight->bEof)`
(ext/fts3/fts3.c:5216). The branch that leaves a side live is the AND/NEAR one at :5177, where
`pExpr->bEof = (pLeft->bEof || pRight->bEof)` marks the node EOF as soon as either side is exhausted
and leaves the other with `bEof==0` at a live docid. `fts3EvalTestExpr` then still counts that phrase
as a hit at the current row, `fts3BestSnippet` saves `pHead`/`pTail` into its reader, and a later
phrase's poslist fetch restarts the group, whose reload frees the reader.

Applying becd68ba0d's one line, `pRight->bEof = pLeft->bEof = 1;`, to each candidate branch of the
pinned amalgamation settles it:

| where the line goes | host ASan | the query |
| --- | --- | --- |
| nowhere (the pin) | heap-use-after-free in `fts3SnippetAdvance` | `rows=1`, snippet `<b>one</b>` |
| after the AND/NEAR `bEof` (:5177) | **silent** | `rows=1`, snippet `<b>one</b>` -- unchanged |
| after the OR `bEof` (:5216) | silent | **`rows=0`** -- matches nothing |

Only the AND/NEAR placement removes the defect while preserving the result upstream's own test
asserts, which is what a fix looks like; the OR placement silences the report by breaking iteration.

**Probe site 46** marks three points and prints the pointer at each: the save in
`fts3SnippetFindPositions` (`pPhrase->pHead = pCsr`), the free in `sqlite3Fts3SegReaderFree`, and the
read in `fts3SnippetAdvance`, where it also judges the pointer by address.

| platform | saved / freed base / read | witness |
| --- | --- | --- |
| native memsys5 | `p=base+180`, base freed, same `p` read | `kind=UAF off=180 req=256` |
| virtual Capstone, lookaside off | `p=base+268` | `kind=UAF off=268 req=512` |
| CheriBSD purecap | `p=base+268` | `kind=UAF off=268 req=289 tag=1 cb=<arena>+262144` |

The offset differs because capability pointers widen `Fts3SegReader`; the saved pointer is inside the
freed reader on all three, and host ASan's 180-byte offset into a 208-byte region is the same
reading. The witness fires everywhere, so unlike cases 07 and 08 this block is not reissued before
the stale read: both kinds of evidence, pointer identity and address judgement, are available here.

The CheriBSD line also measures why that arm is silent rather than inferring it: `tag=1` and
`cb=<arena>+262144` say the pointer is tagged and bounded to the whole 256 KiB memsys5 arena, so the
access is in bounds and the block's free is not something revocation can see. Cases 03, 05 and 06
established that for this allocator; here it is read off the defect's own pointer.

Arms on this platform, 2026-10-16:

| arm | verdict | evidence |
| --- | --- | --- |
| host ASan | reports | heap-use-after-free, 180 bytes inside a 208-byte region, `fts3SnippetAdvance` |
| virtual-malloc | MISSED | both controls as declared; reached, read, RETURNED |
| virtual-nested-pools | CAUGHT | both controls as declared; `cause=25 pc=0x3faea69838 in fts3SnippetAdvance`, attributed by function |
| cheribsd-revocation | MISSED | positive control traps; the case RETURNS rc=0 |

Both of the case's own reachability checks passed in every run: the snippet is `<b>one</b>` and the
offsets are `0 1 0 3 0 3 0 3`, and the case prints that the nested-OR path was reached on both
queries. The three physical rows in `results/matrix.tsv` predate the 2026-10-11 trigger replacement
and are marked superseded rather than altered.
