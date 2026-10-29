# Results: the temporal cases on PoisonCap's published memsys5 (2026-10-24)

Read with `PREDICTIONS.md` (committed 3cd43d008369, about 17 s before the first run started) and
`PREDICTIONS-REVIEW.md` (committed 378c8ea8f8bf, after the first pair of runs finished and before the
repetitions started; that no result had been read is its author's statement). The run host's clock is
1597.5 s slow, so its `run.json` times read earlier than the commits; `EVIDENCE.md` has the measurement
and the corrected times.

Platform: PoisonCap's published artifact, built from its pins in the Ubuntu 24.04 container
(`ports/sqlite/poisoncap/`), booted single-user with `runtime_revocation_default=0`
(`platform-smoke.json`: their four cheribsdtest poison tests PASS). SQLite: their `sqlite/` tree's
amalgamation (`prepare-sqlite.json`). Each run is one boot; the gate (controls) passed in every run.
Every case was run twice per row (`runs/published-*` and `runs/published-*-rep2`). The catch readings
agree in every case; the second run also records failing `rc=` lines, which reclassified case 23 from
"returned" to "returned-with-error". Result lines: `runs/*/results.jsonl`; scoring:
`verdicts/*/verdicts.md` (`ports/sqlite/poisoncap/corpus/score-cases.py`).

The tools were renamed on 2026-10-26, after every run here; PREDICTIONS.md and PREDICTIONS-REVIEW.md keep the
names they were written with: `build-poisoncap.py` is `corpus/build-cases.py`, `guest-run.py` is
`corpus/run-cases.py`, `verdicts.py` is `corpus/score-cases.py`, `diag-amalgamation.py` is
`corpus/make-diag-amalgamation.py`, `guest-smoke.py` is `platform/smoke-test.py`
(all under `ports/sqlite/poisoncap/`).

This is the second version of this file. The first (62a3999c) scored 19 cases and left 12, 15 and 21
open. Below, the population is corrected, and each of PoisonCap's five non-catches has an explanation
from runs made after that version. One of its readings is withdrawn (21, see "Heap").

## Population: 17 cases

Of the 25 cases on the temporal axis (00-24), 17 are scored: those whose stale access happens in the
corpus's reproducing configuration (00 03 05 06 07 10 12 14 15 16 17 19 20 21 22 23 24). A case whose
setup fails on one allocator stays in the population and is reported as such (07, 12, 21, 23 below).
Left out, the same on both sides:
- 01, 09: the trigger does not reach the defect (gdb on the host build; `case.json` notes).
- 04: the trigger reaches the defect's condition and makes no invalid memory access (`case.json`).
- 08, 11: the trigger does not reach the defect (`case.json`).
- 13: reaching the defect is not established (`case.json`).
- 18: memsys5's in-band free list, a constructed demonstrator with no upstream fix rather than an engine
  defect (`case.json`).
- 02: the declared site dereferences a NULL `pPage1` (btree.c:8983) and frees nothing its access reads
  (`PREDICTIONS-REVIEW.md`; the case's own `case.json` still describes the older reading). It faults
  at its site on both mechanisms, without either's temporal signature (theirs code 2, tag; ours cause
  24).

Separately, open PR #259 (branch `sqlite-review-v2`) marks 01, 04, 09 and 18 WITHDRAWN in their
`case.json` notes; this population does not depend on that PR.

## Caught

| | fault at site, of 17 | of which the temporal signature |
|---|---|---|
| ours, Sublet memsys5, lookaside on (recorded) | **17** | 16 (cause 25, revoked); 00 is cause 28 (bounds) |
| theirs, lookaside on | **12** | 11 (SIGPROT code 3 + console poison line); 00 is code 1 (bounds) |
| theirs, lookaside off | **11** | 10; as lookaside on, without 19 |

Ours is `results/2026-10-20-q15-repair` run-B. It was rerun today on the dev-v6 virtual platform
(`runs/v6-sublet`). The rerun equals run-B in all 25 temporal-axis cases, cause and function both, and
both controls fault in their control functions (cause 25 in `control_read`, cause 28 in
`control_write`).

Theirs, case by case (lookaside on): caught 00 03 05 06 10 14 16 17 19 20 22 24; not caught 07 12 15 21
23. With lookaside off, 19 is not caught for a reason outside the mechanism. Their tree still allocates
a RowSet as 64 bytes (`sqlite3.c:72323`), and the case takes a bounds fault in `sqlite3RowSetInit`
before its defect. `ports/sqlite/cheribsd/adapt-sqlite-cheri.py` fixes that line for our CheriBSD arm.

## Why theirs does not catch 07, 12, 15, 21, 23

**Heap: 07, 21, 23.** Their quarantine holds freed blocks. It revokes and flushes when it holds at
least 16 MiB and a quarter of all allocated bytes, and flushes without revoking when its 4,096 entries
fill (`experiments/study/results/20260927-poisoncap-source-audit.json`). The corpus gives SQLite a 1 MiB
heap. At 1 MiB:
- 23 stops with `optimize rc=7` (SQLITE_NOMEM);
- 07 reports that some iterations could not set up;
- 21 reports that no node was written then freed.

Rerun with an 8 MiB and a 32 MiB heap, all three produce the poison report (SIGPROT code 3, poisoned
blocks of 2048, 1024 and 4096 bytes) (`runs/heap8-la`, `runs/heap32-la`). 07 faults in the case's own
object; 21 and 23 fault in `libc.so.7`, unsymbolized, so for them "at site" is not established. The builds differ
from the published one by `-DSQLITE_HEAP_SIZE` alone (`EVIDENCE.md`). That 07 and 21 fail to set up
because the heap runs out is inferred from this; no failed allocation was recorded for them.
**Withdrawn:** the first version gave 21 as "trigger not set up on their SQLite tree". It is set up
once the heap is larger.

**Rounding: 12.** The defect needs `fts5StructureExtendLevel`'s `aSeg` array to move when it grows.
Their memsys5 rounds every request up to a power-of-two block, so growing from 12 to 24 bytes stays in
the same block (`runs/diag-hooks-la`, a diagnostic copy of their source with print statements only, made by
`corpus/make-diag-amalgamation.py`:
aSeg 0x2a3840 before and after). Nothing is freed, there is no stale access to catch, and the scan
completes with 5 rows. Our Sublet port bounds each block to the request and reports that size, so the
array moves and the stale read happens.

The matched pair (`runs/v6-sublet`, `runs/v6-block-bounds`) changes only the rounding: the same Sublet
port with stock rounding (`ports/sqlite/sublet/sublet-block-bounds.py`, every Sublet lifetime kept).
That arm reads as theirs on 12: no fault, the scan completes with 5 rows (the process's whole output is
in its result line). On the other 24 temporal-axis cases it equals the port. So whether 12 is caught
depends on the allocator's rounding, not on either lifetime check. The in-place reallocation itself was
observed on their tree only; on the block-bounds arm it is what the unchanged scan implies.

**Their QEMU: 15.** The stale read is at 0x236c37, in a 128-byte block at 0x236c00 that their memsys5
had poisoned and quarantined (entry 751; the quarantine never flushed in this case). It goes through a
capability with base 0x236c00, length 0x80 and the poison permission clear (`runs/diag3-la`).

Their QEMU reads the capability-sized word holding the poison marker at `addr & mask`, with
`mask = ~((1<<0)|(1<<2)|(1<<3)|(1<<4))`, which keeps bit 1. The mask is in `cap_check_common_reg`
(`cheri-helper-utils.h:387`, used at :393) and in `cgetpoison` (`op_helper_cheri_common.c:1841`).
Both files are as published in the artifact, and the mask came in with CTSRD-CHERI/qemu d4441202f4fc
(`EVIDENCE.md`).

Measured with a fresh 128-byte block per offset (`runs/ev3-la`, control `uaf_offsets`, offsets 0-31,
55 and 127): a stale read faults at exactly the offsets with bit 1 clear and is silent at every offset
with bit 1 set, 34 of 34. 0x37 has bit 1 set. This explains 15, but has not isolated its cause
experimentally: an arm with only that mask corrected has not been run.

Two other sources read the word without this mask: their `poison-cap-smode` branch aligns with
`QEMU_ALIGN_DOWN(addr, CHERI_CAP_SIZE)`, and their Toooba RTL selects it by shifting the address. If
they behave as read, neither would miss 15 this way; neither was built or run here.

## Against the predictions

Predicted, over the 19-case population: lookaside on 18 or 19 at site, 18 signed; lookaside off 19 and
19. Measured over the same population: 14/12 and 13/11. Over the corrected 17: 12/11 and 11/10.

The review's corrections held where they could be checked: 02 faults at its site without the poison
report, 24 is caught in both rows, and 23 runs out of heap (07 by inference). Not foreseen: 15 (their QEMU's
mask), 21 (heap), 12 (rounding), and 19's RowSet fault with lookaside off.

## Limits

- QEMU only, on both sides. Their poison check is their QEMU's model, and 15 concerns that model, not
  necessarily their hardware.
- Function-level sites: on ours, 16 and 17 fault in `memcmp` and 21 and 23 in `memcpy`, which match by
  name only. On theirs, 21 and 23 (heap arms) fault in `libc.so.7` at offset 0x1b0a34, unsymbolized;
  their console poison bounds are in `results.jsonl`.
- The two allocators differ in more than the check (request against block bounds, quarantine, in-heap
  against external free-list metadata). By the runs above:
  - 07, 21 and 23 come down to the quarantine against a 1 MiB heap (for 07 and 21, inferred);
  - 12 comes down to the rounding;
  - only 15 concerns the check itself.
- The v6 rerun of ours used a substitute musl header tree (`runs/v6-*/run.json`); its readings equal
  the recorded run-B.
