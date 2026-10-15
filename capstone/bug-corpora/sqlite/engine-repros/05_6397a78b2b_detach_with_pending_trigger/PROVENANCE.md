# detachtrig -- 6397a78b2b

Upstream fix `6397a78b2b`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row4 / sqlite-6397a78b2b -- after DETACH the auxiliary schema is freed, but a
 * TEMP TRIGGER keeps pTabSchema pointing at it (attach.c detachFunc frees the aux
 * btree+schema; the temp trigger in aDb[1] survives). A later DROP TRIGGER calls
 * sqlite3DropTriggerPtr -> tableOfTrigger(pTrigger) ->
 *   sqlite3HashFind(&pTrigger->pTabSchema->tblHash, pTrigger->table)
 * which READS the freed aux schema (trigger.c). Fixed 3.30.0. Core, needs TEMPDB.
 * CONTROL: reproduce; on unprotected capstone the freed-schema read completes or
 * faults incidentally. Build in the coreT group (-USQLITE_OMIT_TEMPDB).
```

## Only one arm rejects the invalid access (reviewed 2026-10-15)

`fault_sites` named the whole call chain -- `sqlite3DropTriggerPtr`, `tableOfTrigger`,
`sqlite3HashFind`, `findElementWithHash` -- so a fault anywhere along it counted as a catch. It is
now the one function where the stale load happens. The field's own reasoning already said why:
`tableOfTrigger` forms `&pTrigger->pTabSchema->tblHash` without loading from the freed schema, and
the first load is `pH->ht` inside `findElementWithHash`. Host ASan's access stack agrees.

The narrowing matters because the defect produces a second, downstream fault that the wider list
was crediting as a detection on two arms. Probe site 43 separates them.

**What every arm does at the stale load.** On stock memsys5 -- natively, on virtual Capstone and on
CheriBSD purecap -- site 43 witnesses the load as `kind=UAF` and the load SUCCEEDS; nothing rejects
it. On CheriBSD the probe also prints `tag=1 cb=<arena>+262144`: the pointer's bounds are the whole
256 KiB memsys5 arena and the access is inside them, so CHERI is right to stay silent. What comes
back is `ht=0 htsize=0 count=0 first=0`, identical on all three.

**What follows from that.** With `ht` zero, `findElementWithHash` takes its else branch, `count` is
zero, the loop does not run, and it returns `&nullElement`, whose `data` is 0. `sqlite3HashFind`
returns NULL, `pTable` is NULL, and `sqlite3DropTriggerPtr` dereferences it -- its `assert(pTable)`
is compiled out by NDEBUG. Both arms fault there, at `address=0x0`; CheriBSD reports `si_code=2`,
`PROT_CHERI_TAG`, an untagged dereference, which is neither a bounds check nor revocation.
`ports/sqlite/cheribsd/verify-cheribsd-arm.sh` already lists `detachtrig` among the faults that
occur with revocation on and off alike and says they are not detections; this is the measurement
behind that sentence.

**Sublet is the only arm that rejects the operation before it takes effect.** It faults with
cause 25, the lifetime check, in `findElementWithHash` -- at the stale load itself, which is the
standard the corpus and the paper both state. The other two arms let the invalid access complete
and die on what it returned.

**This correction widens the gap the study reports**, from base 8 of 33 to 7, so it is the
direction in which a review should be most careful. It rests on two lines that can be re-run: the
site 43 witness with no fault, and the fault arriving afterwards at address 0x0. The verdicts
themselves are written by derive-verdicts.py from a run bundle, so this change takes effect at the
regeneration at the end of this review, not here.
