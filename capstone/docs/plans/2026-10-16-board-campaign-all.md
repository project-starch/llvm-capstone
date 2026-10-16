# Board campaign: every benchmark and every bug corpus on caplifive_sublet_d93a8d66b

*Filed 2026-10-16. The lead's request: run all the benchmarks and all the bug corpora on the board.*

Who does what:
- **The helper lane** organizes and dispatches. It keeps this file's status tables current, computes predictions where a model exists, and folds results into the records.
- **The board lane** executes every boot, and builds the new corpus kinds in `bug-corpora/tools/run-silicon-cases.py`.
- **The compiler lane** builds the whole-program silicon ports (wave 5) and never touches the board.

Everything runs on the resident bitstream. No reflash is part of this campaign.

## 1. Board facts that set the order

**Two module lines, which cannot share a boot** (`runtime/silicon/b0run.sh:5-8`; `b0-silicon-delegated-runtime.md:3146-3163`).
- **Classic line:** the SQLite hosts, R1/M1/H1/M2, the 17-rung ladder, P1 and SLT. It runs on the shared pins d04bd83 / df0e9a8 / 93440d5, which stay put.
- **Process-ABI line:** the B-series, mruby, memcached and the corpora. It runs on private firmware: the exact-regions monitor and the 127-slot module.
- A boot carries one line. Waves alternate by boot, not within one.

**Cost of a boot.**
- Fixed: about 6-7 min for the upload and Linux.
- Then, per run: about 5 s for a domain of 2 MiB or less, about 20-30 s at 40-88 MiB.
- The 2026-10-16 corpus sets ran 717 runs in 209 boot-minutes.

**Limits.**
- Firmware: about 20 MB working budget. The hard cap is 32 MB, and a rejected upload (HTTP 413) arrives only after the board has been power-cycled.
- Each corpus run has a 120 s watchdog (`b0run.sh:206`). Benchmarks therefore need their own rung.

**Faults.** Under capstone-exec a domain fault ends only that run, so fault cases mix freely in a boot. A real core wedge (an FP instruction, a non-linear delin) still ends the boot: at most one such risk per boot, and last.

**Timing.** Routed WNS is −15.856. A single unexpected failure is repeated before it is attributed (`b0-silicon-delegated-runtime.md:2829`).

**Image validity.** FP-era images may need a rebuild under the full bounds format (`state/current-state.md:178-187`). Every reused image passes the qemu-pass / check-repr gate before it is staged.

**Pre-registration is serialized.** It runs on the silicon-mimic QEMU (capstone-qemu `silicon/cderive-round` ce1be0c642, `CAPSTONE_CDERIVE_ROUND=1`) under the one shared QEMU lock.

## 2. Benchmarks

**Done on d93a8d66b (`fpga-silicon-measurements-for-paper.md`, the 2026-10-16 sections):**
- the five bare Sublet arms, S-16, R-51, s10b and s1gap;
- Linux, boot C+F, R1 a1/a3 and R-43's sweeps;
- the 17-rung ladder, capability half only;
- P1 cell 5, and the cell 6 fixed/unfixed pair;
- C5u, memcached B2/B3, mruby B5 (boot P), B6 and the R-55 a10 probe.

None of these gives a capability/native ratio on this bitstream.

| # | Benchmark | Line | Driver | On d93a8d66b | Wave | Owner | Status |
|---|---|---|---|---|---|---|---|
| 1 | 15-kernel spatial cost, the bare half, paired with the capability half (layouts K = 0..3) | classic | `tests/rtl-smoke/fpga_driver/run_base_bare_fpga.py`, `build-ladder-base-bare.sh`, `ladder-revival-2026-09-22/pair-halves.py` | capability half only | 1 | board | K = 0 DONE: both halves in boot s10, paired by pair-halves.py with its control passing; 17 rows, 0.981x (beebs_cover) to 2.227x (beebs_insertsort) (measurements doc, dev 58ad0068f4ba; tests/rtl-smoke/ladder-sublet-2026-10-16). K = 1..3 pre-registered there, next on the board |
| 2 | P1 native baseline in the same boot as cell 5 and cell 6 (fixed) | classic | `tests/rtl-smoke/drivers/board-c6var.sh` | cells 5 and 6, no native | 2 | board | half done, per the board lane: the a8 boot ran cell 5 (5f4b44257c933477) beside the native -O2 warm baseline (9a80c1cd2a576ed2), and cell5/native = 1.3513 (1,156,968,951 against 856,160,837 cycles; measurements doc, dev 58ad0068f4ba). P1's cell 6 is the -O2 image df484d98b489aeab, which predates the sublet-runtime fix (the fixed 4bfe3b0e is the -O1 A/B image), so it needs an -O2 rebuild with the fix and its QEMU stats records. Both cells enter at 0x10000, so each gets its own boot beside its own native. Scoped after the ladder |
| 3 | H1 calibration | classic | `drivers/board-r1e4.sh` + `drivers/lists/h1-v2-calibration.txt` | no | 1 | board | queued |
| 4 | S1-SQLite use-after-free (lookaside, memsys5) | classic | `drivers/board-s1sql.sh` | no | 1 | board | queued |
| 5 | mruby B5n (native) beside B5, per-benchmark times | process ABI | `runtime/silicon/build-b5-mruby.sh`, `b0run.sh` b5 rungs | B5 only | 2 | board | DONE (a5da26aae1b1, firmware 2edb66370270cdab). The domain costs 1.25x native -O1 and 1.50x native -O2; per benchmark 1.18-1.38 and 1.38-1.67. Totals 298.85 s (domain, two runs), 239.16 s (-O1, one run), 199.79 s (-O2, two runs), recomputed from the transcript's TIME lines. These are upper bounds on the cost of capabilities: soft-float, musl, LTO, the allocator and syscall delegation differ too |
| 6 | R1 warm (8 boots) and cold (5 boots) | classic | `drivers/chain-r1.sh` | a1, a3 only | 3 | board | queued |
| 7 | M1 node reclamation | classic | `drivers/board-m1v2.sh`, `drivers/chain-m1.sh` | no | 3 | board | queued |
| 8 | M2 dependent chase | classic | `drivers/board-r1e4.sh` + `drivers/lists/f5-chase.txt` (4 boots) | no | 3 | board | queued |
| 9 | SQLLogicTest, 7 files | classic | `tests/rtl-smoke/slt-corpus/run-slt-corpus-fpga.sh`, `build-slt-corpus-images.sh` | no | 3 | board | queued. Check its 1-4 MiB region carves under the exact-regions monitor first |
| 10 | speedtest1 domain vs native, size 100 | classic | the 2026-09-14 driver (`board-b73.sh`) was never committed; it is rebuilt from `run_sqlite_baked_fpga.py` | no | 3 | board | queued, one overnight boot (about 8.5 h) |
| 11 | memcached performance | process ABI | no driver exists | no | — | — | needs a workload definition first |
| 12 | Primitive and borrow cost (the paper's July tables) | classic | `tests/rtl-smoke/build-borrow-breakdown-fpga-nogp.sh`; the board run was driven by hand | no | 5 | compiler (rebuild), then board | needs work |
| 13 | Full BEEBS / CoreMark / RV8 as whole programs | process ABI | QEMU only today (`benchmarks/*/run-all-*.sh`) | no | 5 | compiler (B0 ports), then board | needs work |
| 14 | nginx / PostgreSQL E1 safety matrix | classic | `ports/nginx/e1-bundle`; PostgreSQL blocked by ISSUES M-8 on the old glue | no | 5 | compiler, then board | blocked |

## 3. Bug corpora

**Already on silicon:** memcached, FFmpeg and tshark. That is 12 corpora, 130 defects and the lifetime controls (§7 of `2026-10-15-bug-corpus-on-fpga.md`).

**The rest:** 194 cases in 13 corpora (58 of them now read: sets 7-10 and engine-repros' control arm), plus 3 corpora with nothing to run. Counts are from `bug-corpora/index.json`.

| Corpus | Cases | What a case needs | Wave | Owner | Status |
|---|---:|---|---|---|---|
| postgres/c-repros | 5 | `case.c` + `shared/driver.c` + `shared/upstream_*.c`. The `<N>` protocol, `PG_DEFECT` markers, and a second image per case built with `-DPGCLIENT_NEGATIVE_CONTROL` | 4 | board (new kind) | DONE, set 7 (1ec88e30a300): 5/5 CAUGHT with cause 28 on level0 and on lifetimes, every negative control completes; folded into case.json |
| postgres/mmgr-repros | 5 | `case.c` + the memory-context sources, with and without patch 0003 (CDERIVE/CREVOKE). The `0 N` protocol, like wmem | 4 | board (new kind) | DONE, set 8 (51bf1b392a40): stock 0/5 (all complete), patch 0003 5/5 CAUGHT with cause 25 (00 at its declared GetMemoryChunkMethodID; SubletRelease was pre-registered), corpus control 00 completes on stock and faults on 0003; folded |
| httpd/apr-pool-repros | 1 | `case.c` + `shared/driver.c` (the corpus protocol already) + APR pools + the lease adapter | 4 | board (new kind) | PAUSED (2026-10-16, the lead's virtual-first decision: a new physical-profile kind). Whether `sublet.h` works in a process-ABI app is open: check it on QEMU first. The protected arm is the lend API (`ports/apr/pools/src/allocators/sublet/`: `sublet_take` = MREV then DELIN, `sublet_give` = REVOKE), as the corpus recorded it on 2026-09-22. APR was not moved to CDERIVE/CREVOKE lifetimes when the eight other ports were. The silicon-mimic QEMU treats a DELIN of a NONLIN capability as a no-op (`helper_csdelin`) where silicon raises cause 26 (S-15). So each DELIN's operand is confirmed LINEAR before predicting |
| httpd/bucket-repros | 8 | the same, + apr-util buckets | 4 | board (new kind) | PAUSED, with apr-pool |
| cpython/pymalloc-repros | 20 | pymalloc + patch 0003 through B0. The replay opens a fixture file (`ports/cpython/pymalloc/src/native/main.c:8-12`), so the rung needs file staging and a writable output path | 4 | board (new kind + staging) | DONE (set 9, board 83302e277c36): stock 0/20, patch 0003 20/20 with cause 25 at read_probe; folded as silicon-pym-stock / silicon-pym-sublet |
| mruby/release-differential | 23 | the B5 interpreter + a `.rb` per case. Also needs: a judge for the cause-24 oracle (there is no fixed/buggy pair); the task and pack gems; a revoking allocator, since B5's `silicon-alloc.c` has bounds only | 4 | board (kind) + compiler (gems, allocator) | the bounds-only arm DONE (set 10, board 701bfbb399c2; folded as silicon-mruby-bounds: spatial 5/5 CAUGHT 28, temporal 9 MISSED as detections, 17 MISSED, 18-22 outside the totals, 05/11/23 not run); the revoking arm is PAUSED with wave 5, since it needed the compiler lane's allocator |
| sqlite/engine-repros | 33 | the whole 3.22.0 amalgamation per case. The unprotected control arm (8 cases) runs today on the classic line (`tests/rtl-smoke/repro322-silicon-2026-10-01/build-images.sh` + `cells.tsv`). The revoking arm needs a process-ABI SQLite with the CDERIVE memsys5 patch | 1 (control arm), 5 (revoking) | board; compiler for the port | control arm DONE (board 1185b8a4ff79, three boots; folded as `silicon-sqlite-stock`, configuration `silicon-sqlite-memsys5`). Of the eight cases, six are MISSED with both memsys5 controls completing, and 02 and 05 are CAUGHT under the corpus's fault_sites. Both of those are null-base loads, not detections, as on virtual-malloc. 24 completes where virtual-malloc faults. The emulator agrees with silicon, so the difference is not the silicon; which of the build and platform differences is responsible is open. Every cell is built from the corpus's `case.c` on the virtual arms' input 7fc92e19…. The external review on `origin/sqlite-review` (unlanded) changes 02-05: when it lands, the bundle is re-derived, and its 02 already ran as cell `backupattachrev`. The other 25 cases declare the arm `not written` |
| sqlite/capi-repros | 19 | per-row domains and hosts (`ports/sqlite/sqlite_row*`); the only required arm is host ASan | 5 | compiler | needs port |
| postgres/sql-repros | 9 | `postgres --single` on an initdb'd data directory, with contrib | 5 | compiler | needs port |
| perl/release-differential | 11 | the whole Perl interpreter | 5 | compiler | needs port |
| cpython/allocator-boundary-repros | 32 | the whole CPython interpreter, which builds for the virtual profile only (`ports/cpython/app/port.json`, "note") | 5 | compiler | needs port |
| xlang/repro | 15 | shim domains on the old domain interface; a board runner | 5 | compiler | needs port |
| xlang/lua-cdp | 13 | the same | 5 | compiler | needs port |
| mruby/gc-slot-repros | 0 | planned, no cases | — | — | nothing to run |
| xlang/toctou-double-fetch | 0 | triaged, not built | — | — | nothing to run |
| xlang/reuse-not-free | 2 | no Capstone column is defined | — | — | nothing to run |

## 4. Waves

1. **Classic line, runnable now.**
   - Benchmarks 1-4.
   - sqlite/engine-repros' control arm.
2. **Process-ABI line, runnable now.** Benchmark 5.
3. **Long classic campaigns, scheduled around 1-2, overnight where possible.** Benchmarks 6-10.
4. **Corpora that need only a new kind in `run-silicon-cases.py`:** postgres c/mmgr, httpd ×2, cpython/pymalloc, mruby/release-differential.
5. **Whole-program silicon ports** (compiler lane). **PAUSED by the lead on 2026-10-16:** the full virtual profile is built on silicon first ([`2026-10-16-virtual-profile-on-silicon.md`](2026-10-16-virtual-profile-on-silicon.md)), and applications are ported after it, as virtual images. The list below is kept as it was, in the compiler lane's order, with its planning estimates from reading (each is replaced by a measured figure after its first qemu-pass):
   1. SQLite on the process ABI, with the CDERIVE memsys5 arm (1-2 days). `build-sqlite-silicon.sh:86-87` applies any `SQLITE_SUBLET_PATCH` with `patch -F0` to the build's private copy, and `repro322-silicon-2026-10-01/build-images.sh` already builds 3.22.0 through it, so the arm is a parameter on an existing path. Measured: the patch applies 19/19 with `-F0` to pristine 3.22.0 and to the FTS-patched copy, and is refused on 3.53.3 (12 of 19 hunks), so the check can fail. Open: the virtual arms' input amalgamation is sha256 7fc92e19… (`engine-repros/results/*-virtual/*/inputs.json`), not stock 3.22.0 (7991acd0…); the silicon arm builds from the same input or records the difference, and the patch header's hash label is corrected to say which file it is. Alongside: the mruby gems and the revoking allocator (1 day).
   2. BEEBS / CoreMark / RV8 as whole programs through B0 (1-2 days).
   3. The primitive/borrow cost rebuild (half a day).
   4. The PostgreSQL server: --single, initdb at image-build time, contrib (3-5 days).
   5. The CPython interpreter (2-4 days).
   6. Perl (3-5 days).
   7. sqlite/capi-repros, xlang/repro and xlang/lua-cdp: last on value (capi's only required arm is host ASan; xlang declares none), still in scope.
   8. The E1 matrix: needs item 4 (a B0 PostgreSQL avoids M-8's old glue) plus a B0 nginx build.

   Each port hands back staged images, a build manifest and a qemu-pass on the silicon-mimic QEMU. It then joins wave 4's pipeline.

## 5. How every item runs

**Before the boot**, on the board lane's branch:
- a pre-registration: README, `qemu-and-predictions.json`, `SHA256SUMS`, `runs.txt`;
- predictions are computed, never written by hand;
- every corpus image compiles the case source its corpus's own runner compiles, checked with `cmp` before predicting. A build script written for an earlier campaign may compile a different copy;
- benchmarks pre-register their identity readings, and the ratio being measured.

**In the boot:**
- a known-good control first: `b0-hello` or `lpc`;
- `b0-stats` brackets on the process-ABI line;
- variants batched wide, at most one possible wedge, last;
- the next set baked during the current boot.

**After the boot:**
- RESULTS by image hash, the matrix with `--prereg`.
- Corpus results are folded by the helper lane:
  - `silicon-verdicts.py` / `silicon-controls.py`;
  - `arms.json`, `SCHEMA.md`, `check-corpus.py`, `corpus.json`;
  - `check-corpus` and `derive-verdicts --check`.
- Each fold is re-verified independently: the images are found in the booted firmware by hash, and the readings are regenerated from the transcript.
- Benchmark results go into `fpga-silicon-measurements-for-paper.md`. The paper itself is not edited by this campaign.

## 6. Dispatch log

- 2026-10-16:
  - board lane: waves 1 and 2, wave 3's schedule, wave 4's kinds;
  - compiler lane: wave 5's ports in the order above.
- 2026-10-16, the board lane accepted the queue with nothing rejected. Its order:
  1. wave 2 first;
  2. wave 4 interleaved (postgres c, mmgr, then pymalloc; mruby's bounds-only arm);
  3. wave 1 as each item's images pass the qemu-pass / check-repr gate, with a note of what was rebuilt;
  4. wave 3 and speedtest1-100 last in a day, announced before they start.
- 2026-10-16: the virtual engine-repros bundles cite input 7fc92e19…, which no committed script produced. The compiler lane traced it to `adapt-sqlite-322.sh` missing one sorter-alignment line, and is landing it as its own commit. The board's engine-repros control arm waits for it.
- 2026-10-16: the fix landed as 423d13f5b81f. Stock + script now reproduces 7fc92e19… exactly, and the new assertion is negative-tested. The engine-repros README records it, and the control arm is unblocked.
- 2026-10-16, the compiler lane accepted wave 5 and started with SQLite on the process ABI. The order above is its reordering (BEEBS/CoreMark/RV8 and the borrow-cost rebuild moved up as quick conversions). Scope settled: the plan's rows govern, including capi and xlang.
- 2026-10-16: sets 7 and 8 are folded (845a28954c93). Before engine-repros' control arm was built, its build script was checked against the corpus. The blobclose cell compiled the legacy domain instead of case 03's `case.c`, so the board lane builds every cell from the corpus file. The external review on `origin/sqlite-review` was passed on as well: 02's old trigger faults on a NULL dereference, 04 can make no invalid access, and 05's fault on an unprotected arm is a NULL dereference downstream of the stale load.
- 2026-10-16: engine-repros' control arm is folded. All 11 corpus images were rebuilt byte-identically from a fresh adapted input. The classic line does not keep its firmware, so image membership rests on the runner's check: each image's first 4 KiB found in rootfs.cpio, and those prefixes are distinct across the 13 images.
- 2026-10-16, the lead: silicon is to run the full virtual Capstone/Sublet profile first, and everything is ported after it. The RTL lane has the task (44d82a9eb3f3). Paused: every wave-5 port, including the SQLite process-ABI port, which the compiler lane leaves pushed at a stable point; httpd ×2; and mruby's revoking arm. Continuing: waves 1-3, which measure the current bitstream with images and drivers that already exist, and sets 9 and 10, which are already built.
- 2026-10-16: set 9 (CPython pymalloc) is folded. The converter gains a single-run mode for it.
- 2026-10-16: set 10 (mruby, bounds only) is folded into the corpus's own matrix.tsv records by probe/fold-silicon.py, under rules agreed with the board lane before its results.
