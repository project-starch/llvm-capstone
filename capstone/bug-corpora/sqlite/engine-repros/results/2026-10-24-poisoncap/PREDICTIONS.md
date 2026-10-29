# Predictions: the temporal cases on PoisonCap's published memsys5 (written before the run)

Plan: `PLAN.md` (this folder; pre-registered as docs/plans/2026-10-24-sqlite-poisoncap-corpus.md). Scope: the 25 temporal cases of
`results/matrix.tsv`; 19 declare `fault_sites` (qualified), 6 do not (01, 04, 08, 09, 11, 13: no
qualified trigger, reported as such on every arm). Case 18 is a constructed demonstrator and is
reported beside the 18 upstream defects.

Runs (`ports/sqlite/poisoncap/corpus/build-cases.py`, then named build-poisoncap.py, one boot each through `guest-run.py`): their
`mem5.c` as published, with lookaside on and with lookaside off. Temporal only, so no spatial arm:
a poison fault (SIGPROT code 3 with their console's poison line) can only come from their mechanism,
as a cause-25 (revoked) fault can only come from ours. Verdicts by `verdicts.py`: fault at site, and
mechanism-signed at site.

## Gate, already passed before this file (controls only, no case)

`published` lookaside on: `reuse_clean` returns; `uaf_read` faults with SIGPROT code 3 and
`poison exception` on the console (poison bounds 0x2f0800..0x2f0880, the freed 128-byte block);
heap handed over with `heap_perm_poison=0`; repeated identically in `runs/ctl-published-la2`, where
the evidence control `uaf_keepperm` (the heap with CHERI_PERM_POISON kept, `heap_perm_poison=1`)
read the freed block without a fault. The `spatial` build (their switch off) passed its own gate
(both controls return, `runs/ctl-spatial-la`); it is not part of the corpus run.

## What the mechanism predicts, from their source

- A block freed through `memsys5Free` is poisoned granule by granule and held in a 4,096-entry
  quarantine; it is not reallocated before a flush, and a flush needs 4,096 frees or 16 MiB in use,
  which a 1 MiB case heap never reaches. Any later load or store through a capability whose bounds
  lie inside the poisoned block (every pointer `memsys5Malloc` hands out carries the block's bounds)
  raises the poison fault (QEMU `cheri-helper-utils.h:420-434`). So a stale access to a freed
  memsys5 block faults at its first access.
- Lookaside slots are freed onto the lookaside free list, not through memsys5, so a stale access to a
  slot is not poisoned while the lookaside buffer itself is alive.
- A fault that is not the poison report (an untagged or misaligned dereference after a reused or
  corrupted pointer) counts as "fault at site" only, not as signed.
- Risk: freed blocks never return to a 1 MiB heap, so an allocation-heavy case can run out of heap
  (NO-READING) before its defect.

## Per case (qualified temporal)

Their published arm; "site" = a fault in a declared fault site, "poison" = the mechanism's report.

| case | object (case.json) | lookaside on | lookaside off |
|---|---|---|---|
| 00 | fts5 leaf page buffer (sqlite3_malloc) | site, poison | site, poison |
| 02 | the backup's cached Btree | site, poison | site, poison |
| 03 | Incrblob, a lookaside slot; close frees the lookaside buffer itself | site, poison (the slot's bounds are the freed buffer's) | site, poison |
| 05 | the detached Schema and its hash | site, poison | site, poison |
| 06 | IdxRemCtx (sqlite3_malloc) | site, poison | site, poison |
| 07 | Fts3Table (sqlite3_malloc) | site, poison | site, poison |
| 10 | doclist of the exhausted OR side | site, poison | site, poison |
| 12 | shared Fts5Structure, reallocated | site, poison, or NO-READING (heap) | same |
| 14 | Fts5Structure | site, poison | site, poison |
| 15 | fts5 poslist data | site, poison | site, poison |
| 16 | json_each zRoot copy | site (memcmp), poison | site, poison |
| 17 | json_each zJson copy | site (memcmp), poison | site, poison |
| 18 | a freed block's bytes (constructed) | site, poison | site, poison |
| 19 | RtreeNode | site, poison | site, poison |
| 20 | RtreeNode left in aHash | site, poison | site, poison |
| 21 | RtreeNode payload bound to a statement | site (memcpy), poison | site, poison |
| 22 | EditDist3Config (case turns lookaside off itself) | site, poison | site, poison |
| 23 | fts3 segment writer root buffer | site (memcpy), poison | site, poison |
| 24 | Table, a lookaside-sized Db allocation | not poisoned: returns or incidental site fault | site, poison |

## Totals predicted (19 qualified, case 18 included)

| row | fault at site | mechanism-signed at site |
|---|---|---|
| lookaside on | 18 or 19 | 18 |
| lookaside off | 19 | 19 |

Case 24 is the expected difference between the rows; case 12 may read NO-READING (heap) in either.

## Ours, recorded (not a prediction)

Our Sublet memsys5 arm (`virtual-nested-pools`, lookaside on, `results/2026-10-11-virtual-request-bounds`,
equal in all cells to `results/2026-10-20-q15-repair`), same tiers: fault at site 19 of 19;
mechanism-signed (cause 25, revoked) 17: case 00 faults with cause 28 (bounds) and case 02 with
cause 24 (untagged). Our lookaside-off row is not run.
