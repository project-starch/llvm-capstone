# fts3snip -- fix-2026-08-17

Upstream fix `2026-08-17`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row23 / sqlite-2026-08-17 (a87bd2471a9c) -- fts3BestSnippet() saves poslist
 * pointers (pHead/pTail) per phrase; a later incremental phrase under an OR restarts
 * the NEAR group (fts3EvalRestart, fts3.c:5540), freeing/reloading already-visited
 * doclists, so the saved pointers dangle while scoring (fts3_snippet.c:507/479).
 * Distinct from becd68ba0d. Trunk-only fix. CONTROL: snippet() over an OR of phrases
 * read incrementally; on unprotected capstone completes -> NOTRAP.
 * NOTE: build in fts3 group.
```

## 2026-10-16: proposed for withdrawal as a duplicate of case 10

Two findings, and the second is the one that matters.

**The recorded trigger cannot reach the defect.** The reason is the one case 08 established:
`fts3EvalRestart` walks the subtree of `pNear`, a phrase's most senior NEAR ancestor, so in a flat
`'"sun moon" OR "star sun"'` every phrase has `pNear` = itself, a restart can free only the poslist
of the phrase currently being fetched, and that phrase's `pHead`/`pTail` are stored after the fetch
returns. The 2026-10-11 note gave a different reason -- that `fts3BestSnippet` runs twice and all
three `fts3EvalRestart` calls come after both of them -- which is the same wrong kind of reason as
case 08's: a per-run ordering observation rather than the structure.

**Rewriting the trigger does not produce an independent case.** Case 08's replacement shape was
applied here, with `snippet()` as the consumer, and measured against case 10's own query in the same
build (prepared amalgamation, system allocator, lookaside off):

| trigger | access | freed object | free chain |
| --- | --- | --- | --- |
| `(xx AND ("pp1 pp2" NEAR/3 qq)) OR rr` | `fts3SnippetAdvance` sqlite3.c:164924 | 8-byte merged poslist | `fts3EvalInvalidatePoslist <- fts3EvalRestart <- fts3EvalRestart <- EvalPhrasePoslist <- fts3SnippetFindPositions` |
| `(xx AND (pp NEAR/3 qq)) OR rr` | `fts3SnippetAdvance` sqlite3.c:164924 | 176-byte `Fts3SegReader` | `sqlite3Fts3SegReaderFree <- SegReaderFinish <- fts3SegReaderCursorFree <- fts3TermSelect <- fts3EvalPhraseLoad` |
| case 10's `(def AND (one NEAR abc)) OR one` | `fts3SnippetAdvance` sqlite3.c:164924 | 208-byte `Fts3SegReader` | identical to the row above |

All three report the same instruction through the same four frames --
`fts3SnippetAdvance <- fts3SnippetNextCandidate <- fts3BestSnippet <- sqlite3Fts3Snippet` -- which
is also the stack `results/2026-10-11-host-asan` records for case 10. The pointer being dereferenced
is `pPhrase->pHead`, saved at `fts3_snippet.c:479` and read at `:381`/`:400`: this case's declared
object. The only thing a rewrite can vary is which object the restart frees, and a two-token saved
phrase moves that to the merged poslist without moving the read.

Case 08 survived exactly this test; this case does not, and the difference is specific. There the
consumer is a different function, `sqlite3Fts3Offsets`; the saved pointers live in a different array,
`aTerm[]` rather than `aPhrase[]`; and upstream fixed the two defects separately, in 3.49.0
(`51dd67080a`) and 3.32.0 (`becd68ba0d`). Here the consumer, the array, the instruction and the free
chain are case 10's, and the upstream reference this case carries, `a87bd2471a9c` / 2026-08-17,
appears nowhere outside this corpus's own files -- no release, no fix id that can be checked.

Case 10's own `case.c` already runs `snippet()` **and** `offsets()` on its query and reports each, so
there is no reading this case could add.

No arm is re-run. One record is corrected: the physical `sublet: detected` row for this case in
`results/matrix.tsv` is another of the 28 halts at kernel pc 0x800239e4, measured on a trigger that
cannot reach the defect, and it is marked there rather than altered.

The withdrawal joins case 01 (not reached) and case 04 (not a memory-safety defect) in the single
renumber at the end of this review. The reason differs from both and should be recorded as such:
not a trigger that misses, and not a defect out of scope, but a second row for one defect.
