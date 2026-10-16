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

The verdicts are unchanged, and this case is proposed for withdrawal with case 01 when the review
of this corpus ends, under the same reason: the arms had nothing to catch.
