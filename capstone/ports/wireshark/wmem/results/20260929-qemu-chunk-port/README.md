# The per-chunk Sublet port of wmem's block allocator (capstone-qemu, 2026-09-29)

**Question.** The region-granular hooks give a wmem block one region. So a chunk freed inside a live
block kept its authority until the block was reset (fixture 4, "neither mode faults"). That was the
one lifetime the PoisonCap arm covered and the Sublet arm did not. Does giving every chunk a region
of its own close it, without changing what the allocator does?

**Answer: yes, for the lifetimes the fixtures exercise, with a one-variable control behind each
fault.** Predictions were written and pushed in `PREREGISTRATION-chunk-port.md` (`b9ecf27b20bd`),
before the first ported line. The v2 predictions were added in its addendum (`3abcdbca84f8`),
committed before v2's first boot; the v2 builds ran just before that commit, and its message
reports them.

This folder holds two runs:

- **v1** (`e5d20a1b27d3`) met P1-P5. A claim audit then upheld its security result, found one defect
  in the allocator, and showed that three of its claims said more than the evidence did.
- **v2** fixes the defect, adds the fixtures and controls the audit asked for, and restates the
  claims.

## Verdicts (v2)

| | predicted | result |
|---|---|---|
| **P1** | a chunk freed in a live block FAULTS in the protected mode, completes in spatial | fixture 4 faults at the labelled read (`pc == expected_pc`). The **ablation arm**, the port with only `wm_chunk_retire`'s one give stubbed out, completes: the fault is that revoke's |
| **P2** | the unprotected stale read sees the old byte, not a free-list node | checked in fixtures 3 and 4 and passes. The same check in the OFF control fails, status 519 and 520, re-run from the committed tree |
| **P3** | H4: metadata > hierarchy; workaround 0; H1: application 0 | metadata 212 of 340 added code lines (62%, 70% without `#if` guards); workaround 0; application 0. **Holds only at patch scope**: see "Scope decides H4" |
| **P4** | a reset is one revoke per retained block; a chunk free is one | the chunk file's `revokes` = `reset_revokes + close_revokes + retires` in every protected run that printed counts (replay: 140 = 40 + 20 + 80; fixture 17: 1003 = 1 + 1 + 1001; fixture 0: 7 = 2 + 2 + 3). The check can fail: the ablation arm prints `retires=1 revokes=2`. The regions file's revokes are now reported beside it (replay: 900) |
| **P5** | behaviour unchanged | both QEMU replays equal 2026-09-21's reports on 16 of 16 fields, and the native replay matches upstream. What that can and cannot see is stated below |

**The new fixtures** (v2, mode 1; every one completes in mode 0, as predicted):

| case | port | OFF (hooks only) | ablation |
|---|---|---|---|
| 13 read after a GROWING realloc | FAULT at `wm_probe_read` | completes (by upstream's code this chunk grows in place, so the old pointer is the live one; the arm has no address check to show it) | completes |
| 14 read after a SHRINKING realloc that splits | FAULT at `wm_probe_read` | completes | — |
| 15 read after freeing a BLOCK jumbo | FAULT at `wm_probe_read` | FAULT at `wm_probe_read` too: **not discriminating**, as registered | — |
| 16 second free of a chunk in a live block | FAULT at `wm_widen_probe` | completes: no fault at the probe | — |
| 17 1000 alloc/free of 8 bytes reuse ONE address | completes; `splits=2 issues=1002` | completes | — |

The positive control for 17 swaps v1's generated allocator into the same build. It fails with
status 528 in both modes: under v1, each of those frees would have carved new space.

## What the audit found, and what changed

- **A defect.** v1 kept upstream's free-list minimum, which is room for two pointers in the chunk's
  data (32 bytes in purecap), although the links had moved to the record. A freed chunk below that
  size joined no list, and since chunks never rejoin, it was lost until the block's next reset. v2
  lists every chunk that can serve a request, and a shrink no longer splits off a remainder too
  small to list. Fixture 17 and its v1 control are the evidence.
- **Withdrawn: "every address handed out is upstream's".** No run compared addresses, and after a
  free that upstream would merge, placement diverges. What holds is narrower: a block's layout, its
  header sizes included, is upstream's.
- **Withdrawn: `revokes=140` as the domain's total.** It was the chunk file's count only;
  `sublet.h` counts per file. Both files' counts are now reported.
- **Restated: P4's "regardless of chunk count".** It holds for the number of revoke instructions,
  not for their cost. Every reset and close now returns the block UNINIT, because its chunks are
  linear children, and pays a capability-initialising fill of the whole block: `inits=60` in the
  replay, 40 + 20. Revoke latency also grows with the descendants' node count.
- **Restated: P5.** The replay's checksum is a function of the trace, so the equality means the
  same 1661 events complete with every payload check passing. It cannot see placement. And on this
  trace, by a reading of its generator (`tests/native/test-replay.py`), upstream never merges or
  grows in place: every freed BLOCK chunk has used neighbours. The payload checks are shown to fire: a mutation
  that makes a split overlap its neighbour fails with `WM failure 301`, and the restored build
  passes (result-lines section 9). The realloc and reuse paths are now covered by fixtures 13, 14
  and 17 instead.
- **Provenance.** v1's P2 evidence was built from an uncommitted `Replay.cmake` and `lifetimes.c`.
  Both are committed in v2, and P2 was re-run from them.

Two runner defects surfaced on the way, and both are fixed in this change:

- `port_support.run_guest` waited 60 s for the shared QEMU lock. Another lane's boot holds it
  longer, and the cell was then recorded as FAIL with no guest output: two cells of the first,
  aborted v2 matrix did exactly that, each after 60 s (result-lines section 5a). The wait is now
  `CAPSTONE_QEMU_LOCK_WAIT`.
- An expired lock wait now exits 75 (`flock -E 75`), and the security runner records no verdict
  for it. Any other launch failure keeps its own status, so it is never read as infrastructure.
  Both directions were tested (section 9): a held private lock gives 75 and the INFRA line; a
  missing emulator gives 1.

## Scope decides H4

At patch scope, as every port's `.classes` counts, metadata is 62%, and H4 holds. All of the Sublet
calls live in `src/allocators/sublet/chunks.c`, beside the patch. Counted as hierarchy, that file
and its header bring metadata to 44%; with `backing.c` and `regions.c` too, 37%. Had the records
and the index gone into a side file, as PostgreSQL's pools did, the patch's metadata would be 90
lines against 128, and H4 would be refuted even at patch scope. `effort.txt` has the table. This port's H4 outcome is as much about where its code was put
as about what the discipline demands.

## Costs and limits, stated

1. **QEMU only.** capstone-qemu untags a revoked capability on load; the deployed silicon lets a
   stale access retire (ISSUES **Q-11**). No board arm.
2. **No coalescing.** Freed chunks serve only requests no larger than themselves until a reset. A
   long-lived block allocator with growing requests accumulates free space that nothing reuses. The
   directed replay cannot show this. Real tshark (step 2) is where it is measured.
3. **A shrinking realloc that splits the chunk now revokes the caller's old pointer**, although the
   address it returns is the same. That is realloc's contract, which upstream does not enforce, and
   code that ignores realloc's return value would fault here (fixture 14). A shrink too small to
   split returns a narrowed alias and leaves the old pointer live.
4. **Capacity.** The records and the index live in an 8 MiB area: about 56k live records. A boot
   has 65,536 revocation nodes, at about two per chunk. Both are far above the replay's 15 chunks
   per block and have not been tested near their limits.
5. **Shared state.** The index and the spare-record lists are global across block allocators,
   where upstream has none. That is unsafe if two threads use different allocators, and a chunk
   freed through the wrong allocator can leave a record dangling.
6. **N = 1 per cell.** QEMU's capability semantics are deterministic. One guest stall (mode 1, case
   4) was retried, and the attempt is kept. The first, aborted v2 matrix's 18 attempts are listed in
   result-lines section 5a; every one that ran agrees with the final cell.
7. **Compiler** `3979abd8`, which predates the C-50 fix. The gate finds 0 hits in all ten v2 images.

## Files

- `result-lines.txt`: v1's cells (sections 1-4), then v2's matrix, counts, controls, replays, and
  the native, mutation and C-50 checks (sections 5-9).
- `effort.txt`: `port-effort.py` for v1 and v2, the guard-free counts, and the two-scope table.
- `SHA256SUMS`: every image each run loaded, for both versions.
