# Paper notes: temporal SQLite defects, Sublet against PoisonCap

Material for the paper, not paper text. Every figure here is from `RESULTS.md` and `EVIDENCE.md` in
this directory, and each row names its evidence. Temporal axis only.

## The claim, in one line

On the 17 temporal SQLite 3.22.0 engine defects whose stale access fires, Capstone with Sublet faults
at the defect's site on all 17. PoisonCap's published artifact faults on 12.

## What was run

- **Corpus:** `bug-corpora/sqlite/engine-repros`, temporal cases 00-24. 17 are scored; the other 8 are
  left out on both sides, each for the reason its own `case.json` records (see "Left out").
- **Ours:** memsys5 on its Sublet port (`ports/sqlite/sublet/sublet-3220000-memsys5.patch`), lookaside
  on, on virtual Capstone (QEMU). Recorded: `results/2026-10-20-q15-repair/run-B`. Rerun on the dev-v6
  platform on 2026-10-24 (`runs/v6-sublet`), equal to run-B in every case.
- **Theirs:** the PoisonCap CCS 2026 artifact, as published (repository commit 731c5720, archive
  sha256 c5e8b07d), unmodified. It provides their memsys5, SQLite tree, QEMU, CheriBSD and LLVM, built
  from its own pins. Run on their QEMU, booted single-user with `runtime_revocation_default=0`. Their
  four cheribsdtest poison tests pass on it (`platform-smoke.json`).
- **Same on both sides:** case sources (`*/case.c`), the 1 MiB memsys5 heap, `-O0`, and a positive
  control in every run. Their side ran with lookaside on and off, each twice.
- **Pre-registered:** predictions in 3cd43d008369, before the first run. A review was recorded in
  378c8ea8f8bf, before the repeat runs started; that no result had been read by then is our statement,
  not provable from timestamps (`EVIDENCE.md`). Results are in 62a3999c (first version) and 80bd2f91
  (current).

## Headline

| system | fault at site, of 17 | of which the temporal signature |
|---|---|---|
| Capstone + Sublet | **17** | 16 (cause 25, revoked); case 00 is cause 28 (bounds) |
| PoisonCap, lookaside on | **12** | 11 (SIGPROT code 3 + poison report); case 00 is code 1 (bounds) |
| PoisonCap, lookaside off | **11** | 10 (as above, without 19) |

## Per case

"Theirs" is the second run of each row (`verdicts/published-*-rep2/verdicts.md`); the first agrees in
every case.

| case | defect | ours | theirs, lookaside on | theirs, lookaside off | why theirs misses |
|---|---|---|---|---|---|
| 00 | fts5 getvarint stale read | cause 28, `sqlite3Fts5GetVarint` | code 1, same function | code 1 | caught (bounds on both) |
| 03 | blob close after db close | cause 25, `sqlite3DbFreeNN` | code 3, same | code 3 | caught |
| 05 | detach with pending trigger | cause 25, `findElementWithHash` | code 3, same | code 3 | caught |
| 06 | expert rem ctx freed | cause 25, `idxRemFunc` | code 3, same | code 3 | caught |
| 07 | fts3 destroy OOM stale table | cause 25, `fts3DestroyMethod` | returned | returned | heap (inferred) |
| 10 | fts3 snippet OR stale iter | cause 25, `fts3SnippetAdvance` | code 3, same | code 3 | caught |
| 12 | fts5 in-place stale leaf | cause 25, `fts5SegIterNextPage` | returned | returned | rounding: defect never fires |
| 14 | fts5 structure release refcount | cause 25, `fts5StructureRelease` | code 3, same | code 3 | caught |
| 15 | fts5 vocab EOF stale poslist | cause 25, `sqlite3Fts5PoslistNext64` | returned | returned | their QEMU's marker address |
| 16 | json_each root stale key | cause 25, `memcmp` | code 3, same | code 3 | caught |
| 17 | json_each static stale key | cause 25, `memcmp` | code 3, same | code 3 | caught |
| 19 | rtree cursor stale node | cause 25, `nodeRelease` | code 3, same | code 1 in `sqlite3RowSetInit` | caught on; off: their port's RowSet bug fires first |
| 20 | rtree inode0 stale node | cause 25, `nodeHashLookup` | code 3, same | code 3 | caught |
| 21 | rtree static bind stale text | cause 25, `memcpy` | returned | returned | heap (inferred) |
| 22 | spellfix OOM stale vtab | cause 25, `editDist3FindLang` | code 3, same | code 3 | caught |
| 23 | fts3 static bind stale text | cause 25, `memcpy` | returned, `optimize rc=7` | same | heap (NOMEM read) |
| 24 | writable schema stale table | cause 25, `sqlite3OpenTable` | code 3, same | code 3 | caught |

## The five misses, by cause

| cause | cases | what it is | evidence |
|---|---|---|---|
| quarantine against the corpus heap | 07, 21, 23 | Their quarantine keeps freed blocks; at 1 MiB, setup fails. 23 records NOMEM; for 07 and 21 the failure is inferred. With 8 and 32 MiB heaps (only `-DSQLITE_HEAP_SIZE` differs) all three produce their poison report. | `runs/heap8-la`, `runs/heap32-la`, `EVIDENCE.md` |
| allocator rounding | 12 | Power-of-two blocks let the array grow in place, so nothing is freed and there is no stale access. Our Sublet port with the same rounding also does not fire on 12, and is unchanged on the other 24 cases. | `runs/v6-sublet`, `runs/v6-block-bounds` |
| their QEMU model | 15 | Their QEMU reads the poison marker at `addr & ~0x1D`, keeping bit 1. Per offset of a freed block, a read faults exactly where bit 1 is clear (34/34); case 15 reads at an address ending in 0x37. Their RTL and their later QEMU branch align differently; neither was run. | `runs/ev3-la`, `EVIDENCE.md` |

How to report each:
- **Do not adjust the 12.** The larger heaps are a sensitivity result. 21 and 23 then fault in
  unsymbolized libc, so "at site" is not established for them.
- **12** may be called "defect not triggered on their allocator" rather than a detection failure.
- **15** is a miss of their artifact as published. Say it is a QEMU-model defect, and that whether
  their hardware has it is unknown.

## Left out, the same on both sides

| case | reason (from its `case.json`) |
|---|---|
| 01, 08, 09, 11 | trigger does not reach the defect (gdb on the host build) |
| 04 | trigger reaches the defect's condition, no invalid memory access |
| 13 | reaching the defect not established |
| 18 | memsys5's in-band free list, our constructed demonstrator, no upstream fix |
| 02 | NULL `pPage1` dereference, not a use-after-free (`PREDICTIONS-REVIEW.md`); faults on both sides without a temporal signature |

## Caveats to state

- QEMU on both sides; each side's check is its own emulator's model. No FPGA.
- The allocators differ in more than the check: request-sized against power-of-two blocks, quarantine,
  free-list metadata. 07/21/23 (quarantine) and 12 (rounding) are policy differences; only 15 concerns
  the check.
- Their artifact is a proof of concept by their own description, and its kernel revoker is experimental.
  We booted it single-user with `runtime_revocation_default=0` (multi-user panics on this build). With
  that setting their libc does not strip the poison permission, so our harness hands memsys5 its heap
  with `CHERI_PERM_POISON` removed (`repro322_common.h`, `REPRO322_POISONCAP`). The evidence control
  `uaf_keepperm` shows that keeping the permission bypasses their check. The memsys5 quarantine's
  revocation threshold (at least 16 MiB held) is never reached in a 1 MiB heap.
- Diagnostic only, never counted: `runs/diag-hooks-la` and `runs/diag3-la` ran a copy of their SQLite
  with print statements added, to locate the array in 12 and the read address in 15. Every counted
  run uses their code unmodified.

## Their own SQLite evaluation (speedtest1), for related work

- The published `sqlite/test/speedtest1.c` runs 20 of the 32 tests in its main test set. The other 12
  are commented out in the artifact copy: 150, 161, 170, 180, 200, 230, 270, 290, 310, 500, 510, 520.
- Their fork's history (CTSRD-CHERI/sqlite, last PoisonCap commit a879a48e) does not have these
  comments. The disabling exists only in the artifact, and neither its README nor any commit message
  gives a reason.
- Their paper (arXiv v1, 2026-05-13): with revocation enabled, 15 of 32 tests completed (8 without a
  revocation pass, 7 with 32 nested passes between them); "the remaining 17 tests which did not succeed
  were either hanging or triggering a kernel panic", attributed to "recursive page fault handling as a
  result of poison probes in the revoker". All 32 passed on FPGA without revocation, and the reported
  results are "limited to a subset of the SQLite tests that were able to complete".
- v1's Figure 6 plots "MEMSYS5 revocation+PoisonCap" bars for 13 phases only: 110, 120, 130, 140,
  142, 145, 161, 230, 240, 250, 260, 320, 400 (read off the figure; the text says 15 completed).
- v3 (2026-09-29) drops the counts and the "subset" sentence (searched the full text for hang, panic,
  subset, "were able to complete"). Its Figure 4 plots "PoisonCap revocation(MEMSYS5)" bars for all 32
  phases.
- The three published sets disagree:

  | source | what it reports for PoisonCap with revocation |
  |---|---|
  | v1 text | completed: 15 of 32; hung or panicked: 17 |
  | v1 Figure 6 | plotted: 13 phases |
  | artifact `speedtest1.c` | enabled in the driver: 20; commented out: 12 |
  | v3 Figure 4 | plotted: all 32 phases |

  These are different predicates (completed, plotted, enabled), so the rows need not agree; what the
  table shows is that no published source names the completed set.

  - 161 and 230 have v1 bars but are disabled in the artifact.
  - 9 artifact-enabled phases have no v1 bar: 100, 160, 190, 210, 280, 300, 410, 980, 990.
  - v3 shows bars for the 12 phases the artifact disables.
  - Nothing published says which of these is current.
- Run on their QEMU, their code unedited (`ports/sqlite/poisoncap/results/2026-10-25-speedtest1/`), on a
  kernel with their FPGA kernel's options:
  - the artifact's 20 tests complete with both revocation layers active (one run, 4.2 h at their
    default `--size 100`);
  - their own 32-test driver (commit a879a48e) ended in the same kernel panic (`Poison probe missing
    page`) in 4 of 4 runs; where the output was copied (3 runs), test 150, "CREATE INDEX five times",
    was the last test started. It also happens with libc revocation off throughout; their memsys5
    calls the revoker itself, the plausible trigger (not identified in the records).
  - With libc revocation on from boot, copying the binary into the guest already fails: the debug
    kernel panics, the NODEBUG kernel stops answering.
  - Their published build recipe does not enable memsys5; configure's `--enable-memsys5` is needed for
    their `main()` to run.
