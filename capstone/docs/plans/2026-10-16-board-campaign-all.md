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
| 1 | 15-kernel spatial cost, the bare half, paired with the capability half (layouts K = 0..3) | classic | `tests/rtl-smoke/fpga_driver/run_base_bare_fpga.py`, `build-ladder-base-bare.sh`, `ladder-revival-2026-09-22/pair-halves.py` | capability half only | 1 | board | queued |
| 2 | P1 native baseline in the same boot as cell 5 and cell 6 (fixed) | classic | `tests/rtl-smoke/drivers/board-c6var.sh` | cells 5 and 6, no native | 1 | board | queued |
| 3 | H1 calibration | classic | `drivers/board-r1e4.sh` + `drivers/lists/h1-v2-calibration.txt` | no | 1 | board | queued |
| 4 | S1-SQLite use-after-free (lookaside, memsys5) | classic | `drivers/board-s1sql.sh` | no | 1 | board | queued |
| 5 | mruby B5n (native) beside B5, per-benchmark times | process ABI | `runtime/silicon/build-b5-mruby.sh`, `b0run.sh` b5 rungs | B5 only | 2 | board | queued |
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

**The rest:** 194 cases in 13 corpora, plus 3 corpora with nothing to run. Counts are from `bug-corpora/index.json`.

| Corpus | Cases | What a case needs | Wave | Owner | Status |
|---|---:|---|---|---|---|
| postgres/c-repros | 5 | `case.c` + `shared/driver.c` + `shared/upstream_*.c`. The `<N>` protocol, `PG_DEFECT` markers, and a second image per case built with `-DPGCLIENT_NEGATIVE_CONTROL` | 4 | board (new kind) | queued |
| postgres/mmgr-repros | 5 | `case.c` + the memory-context sources, with and without patch 0003 (CDERIVE/CREVOKE). The `0 N` protocol, like wmem | 4 | board (new kind) | queued |
| httpd/apr-pool-repros | 1 | `case.c` + `shared/driver.c` (the corpus protocol already) + APR pools + the lease adapter | 4 | board (new kind) | queued. Whether `sublet.h` works in a process-ABI app is open: check it on QEMU first |
| httpd/bucket-repros | 8 | the same, + apr-util buckets | 4 | board (new kind) | queued, with apr-pool |
| cpython/pymalloc-repros | 20 | pymalloc + patch 0003 through B0. The replay opens a fixture file (`ports/cpython/pymalloc/src/native/main.c:8-12`), so the rung needs file staging and a writable output path | 4 | board (new kind + staging) | queued |
| mruby/release-differential | 23 | the B5 interpreter + a `.rb` per case. Also needs: a judge for the cause-24 oracle (there is no fixed/buggy pair); the task and pack gems; a revoking allocator, since B5's `silicon-alloc.c` has bounds only | 4 | board (kind) + compiler (gems, allocator) | queued |
| sqlite/engine-repros | 33 | the whole 3.22.0 amalgamation per case. The unprotected control arm (8 cases) runs today on the classic line (`tests/rtl-smoke/repro322-silicon-2026-10-01/build-images.sh` + `cells.tsv`). The revoking arm needs a process-ABI SQLite with the CDERIVE memsys5 patch | 1 (control arm), 5 (revoking) | board; compiler for the port | queued |
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
5. **Whole-program silicon ports** (compiler lane), in order of what each unlocks:
   1. SQLite on the process ABI, with the CDERIVE memsys5 arm;
   2. the PostgreSQL server;
   3. Perl;
   4. the CPython interpreter;
   5. BEEBS / CoreMark / RV8 as whole programs;
   6. the primitive-cost rebuild.

   Each port hands back staged images, a build manifest and a qemu-pass on the silicon-mimic QEMU. It then joins wave 4's pipeline.

## 5. How every item runs

**Before the boot**, on the board lane's branch:
- a pre-registration: README, `qemu-and-predictions.json`, `SHA256SUMS`, `runs.txt`;
- predictions are computed, never written by hand;
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
