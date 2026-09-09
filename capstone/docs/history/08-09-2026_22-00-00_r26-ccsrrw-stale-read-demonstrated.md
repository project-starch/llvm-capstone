# R-26 demonstrated in simulation: a younger load checks CPMP against the entry a pending CCSRRW is about to replace

**Date:** 2026-09-08. **RTL:** `ef5a8eaf2` (`fpga-testing-dev`), Verilator, `S12_MEM_DELAY=40` for the deciding arm.
**Status:** the hazard the board lane filed as R-26 (RTL reading) is **measured** for the CPMP data check; the
CSCRATCH/switcher path and the capability memory instructions are the same check but were not exercised.

## The claim (R-26 as filed)

A `CCSRRW` write to `cpmp[i]` or `cscratch` takes effect only when it commits (`csr_regfile.sv:2392-2418`
under `ccsr_we_i`, driven from `commit_stage.sv:379-383`) and, unlike the ordinary side-effecting CSR writes,
never raises `flush_o`. A younger memory instruction is not held behind it (the CSR-buffer stall marks only the
CSR unit busy, `issue_read_operands.sv:413-491`) and reads `cpmp` combinationally in `pmp_data_if.sv:117-133`.
So it can run its check against the stale entry. The monitor's `fence.i` after such writes would be the flush
the write never triggers.

## The test

`verif/tests/custom/capstone/r26-*.S` (committed on branch `r26-ccsrrw-stale-read`, kept OUT of
`testlist_capstone.yaml` because one arm is expected to FAIL until the RTL is fixed — the SEAL precedent).
All in M-mode with `mstatus.MPRV=1, MPP=S`, so M-mode loads are CPMP-checked (`pmp_data_if` uses
`ld_st_priv_lvl`) and no `mret` sits between the CSR write and the load. CPMP[0] starts as a wide RW entry
over three cache lines of `.data`; a narrow entry (first line only) is prepared in a register.

| phase | sequence | expected | reads |
|---|---|---|---|
| 0 control | checked load OUTSIDE the wide entry | trap | proves the check fires |
| 1 the arm | `[older op]` · `CCSRRW CPMP0 ← narrow` · `[fence.i]` · load in wide, outside narrow | trap iff the load saw the new entry | the question |
| 2 post-fence | `fence.i` · the same load | trap | proves the write landed |

PASS = 1/1/1 traps. FAIL code 11 = phase 1 did not trap = **hazard**. Older-op variants: none; one `div`
(three `div`s); one `ld` from an untouched line inside both entries (a cache miss, 40-cycle memory).

## Results

| arm | older op | fence.i | verdict | phase-1 load |
|---|---|---|---|---|
| nodelay | — | — | PASS | trapped |
| div, div3 | `div` ×1 / ×3 | — | PASS | trapped |
| fence | `div` | yes | PASS | trapped |
| noolder (v2 layout) | — | — | PASS | trapped |
| **ldmiss** | `ld` (miss) | — | **FAIL 11** | **did not trap; retired with the data** (`0x3333…`, RVFI) |
| ldmiss-fence | `ld` (miss) | yes | PASS | trapped |

**Why the div arms were clean — the waveform, not an argument.** VCD of the `div` arm (2 ticks = 1 cycle):
`flu_ready_i` drops at t=1169 when the divider starts and stays low until t=1297; the CSR op cannot issue
while the shared fixed-latency unit is busy (`fus_busy[0].csr` follows `!flu_ready_i`). The CCSRRW issues at
1297, commits at 1301 (`ccsr_we_i`), the new bounds are in `cpmp_q[0]` at 1303 — and the younger load's check
(`pmp.lsu_valid_i`) is at 1303 as well: one cycle after commit, the earliest a load issued after the CSR op can
reach the check, and the same edge on which the write lands. So without an older instruction that is NOT on
the fixed-latency unit, the window is exactly zero cycles wide, by timing, not by design.

**The hazard — VCD of the `ldmiss` arm:**

| tick | event |
|---:|---|
| 1169 | older load (`older_line`, 0x…3040) issued; translation 1171; allowed; misses (memory delay 40) |
| 1173 | `CCSRRW CPMP0` in the CSR buffer (`csr_addr_i = 0x010`), waiting for commit behind the older load |
| 1181 | **probe load issued** (0x…3080, inside wide, outside narrow) |
| 1183 | **its CPMP check runs: `cpmp_allow = 1`** against the wide entry, no exception |
| 1191 | `CCSRRW` commits (`ccsr_we_i`) |
| 1193 | `cpmp_q[0]` bounds now narrow (`…t = 0x3080`) — **five cycles after the check** |

The load retired with the data (`x6 = 0x3333333333333333`), phases 0 and 2 trapped, verdict FAIL 11. With
`fence.i` between the write and the load the same arm PASSes. Records: `~/dev/llvm-capstone-rebuild/records/r26/`
(`ldmiss.vcd`, `ldmiss.rvfi.dasm`, `ldmiss-events.tsv`, `div-events.tsv`, build/sweep logs).

## What it means

* **R-26 is real for the CPMP data check.** Any older instruction that delays the CCSRRW's commit and is not on
  the fixed-latency unit — a load miss is the ordinary case — opens a window of that instruction's latency in
  which every younger load/store is checked against the old entry. A plain `ld` was used; `LDC`/`STC` reach the
  same `pmp_data_if` check through the LSU and are not expected to differ, but they were not run.
* **Not exercised:** the `cscratch`/`cepc` readers in `dom_switch_read_process` (`csr_regfile.sv:401-432`) —
  the domain switcher reads them after the CALL/RETURN commits, i.e. after the older CCSRRW has committed, so
  the same window needs the switcher's read to fall within one cycle of the write; that is a separate test with
  CALL semantics, and it is the shape with a board consequence (`sbi_capstone_init.S:44-51`: CCSRRW…CSCRATCH
  then domain entry with no `fence.i`).
* **The monitor's `fence.i` after CCSRRW is load-bearing** for CPMP writes and must stay until the RTL flushes.
* **Fix shape:** CVA6's own idiom — set `flush_o` in the CCSR write block for CPMPn/CSCRATCH/CEPC (as `satp`
  and friends do at `csr_regfile.sv:1160-1394`), so the controller flushes after the write commits and younger
  instructions re-execute against the new entry. An issue-time interlock (hold CAPSTONE_DYN and LOAD/STORE
  behind a pending cap-CSR write) is the alternative; the flush is smaller and matches the existing mechanism.
  Either is RTL, goes through lint + audit + synthesis before any board use, and would be validated by this
  test turning `ldmiss` into a PASS with the control still firing.

## Correction (2026-09-09)

The waveform in the table above was **not** a 40-cycle-memory run: the trace build re-verilated the model without
the `S12_MEM_DELAY` define (found by audit on 2026-09-09; the arm runner passed no `--isscomp_opts`). The
`ldmiss.vcd` events are a **delay-0** trace, in which the older load still misses the cache and the CCSRRW still
commits five cycles after the younger load's check — the hazard opens on an ordinary miss with no added memory
latency, which is the stronger statement. The FAIL 11 / PASS arm table was **also** taken at delay 0 — the model built for it never contained the define
(the build path logged the define and the built model did not have it; see the 09-09 note's banner). So every
number in this note is a delay-0 number, and the hazard is a delay-0 hazard: an ordinary cache miss opens it. At a
verified 40-cycle latency (`records/r26/pre-all-d40.log`, 2026-09-09) `r26-v2-ldmiss` reads FAIL 11 after 3202
cycles, `ldmiss-fence` PASS, the controls as in the table — the same verdicts, three times the cycle counts.

## Side observation, not chased

The first version of the test initialised its counters in memory right after the capability setup (`CAPENTER`,
two `CAPCREATE`s, `CCSRRW CPMP0`, `csrw mtvec`). That plain `sw` raised an exception whose cause the RVFI
tracer could not name (its table stops at the standard causes; `DEBUG_REQUEST` = 24 is one such), the core
went to the debug ROM at `0x800` and parked in its halt loop; the same `sw` placed before `CAPENTER` was fine,
and stores from the trap handler later in the same run were fine. `cpmp-if-check.S` performs the same store
after the same setup without incident, so the trigger is narrower than "store after CAPENTER". Trace:
`records/r26/` (first `r26-cpmp-nodelay` run). Recorded so it is not lost; it may be a harness artefact.
