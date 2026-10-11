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
