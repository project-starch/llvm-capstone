# R-34 — every exception the load/store unit generates itself (the five capability causes 24–28 AND the misaligned causes 4/6) is LOST when the access is granted in its request cycle; it is delivered only if an exception is still being presented one cycle later

**Status (2026-09-15):** DEMONSTRATED in RTL simulation at `f6ec6c198` (capstone-ariane, the
`capstone-bootstrap` tip), with the mechanism read from source, the timing read from a waveform, and
the claim audited adversarially (the auditor read the same waveform independently: `cap_exception.valid`
rises 21 times in the run, one is delivered). **The stock `rv64mi-p-ma_addr` FAILS on this RTL with
capmode never set (TESTNUM 10, the `ld` that crosses an 8-byte word), so the loss predates the capability
check: it is the LSU's exception delivery, and the capability clauses ride on it.**
The gate and exception lines of `core/load_store_unit.sv` are unchanged between this revision and the
flashed `1bfff7776` (`git diff 1bfff7776 f6ec6c198 -- core/load_store_unit.sv` touches neither);
the board has not run this test. Sibling issues, so a reader with the wrong symptom is redirected
now: **R-24** is the cause-number collision (cause 24 is this core's `DEBUG_REQUEST`) and is what
makes the untagged-base clause enter debug mode instead of trapping; **R-10 / R-29** are data-path
(write-buffer) defects and are unrelated; the M-mode gate on the same block is
`docs/history/15-09-2026_lsu-capmode-gate-why-domains-cannot-satisfy-it.md`, whose "live for the
monitor's own accesses" reading this folder supersedes. This folder is one issue: the exception is
generated and then not delivered.

## Headline, and whether the measurements are contaminated

**A stock RISC-V compliance test fails on this core.** `rv64mi-p-ma_addr` — no Capstone instruction, capmode
never set — returns wrong data without a trap for the `ld` that crosses an 8-byte word (TESTNUM 10) and passes
the in-word cases on the data cache's shifted bytes. That is a base CVA6 LSU exception-delivery defect; the
capability clauses are swallowed by the same drop, so nothing about Capstone is needed to see it.

**Are the published measurements contaminated?** The risk is silent wrong data on a misaligned access, not a
missing trap, and R-34 is a base-core defect with no capmode dependence — so two arms agreeing on the same core
cannot detect it, and neither can run-to-run determinism (a deterministic bug returns the same wrong value every
time). What closes it is CROSS-MACHINE agreement with a machine that has no such defect: SQLite `speedtest1`'s
result hash `112006 38bb59fd` is produced under capstone-qemu (the ④ cell, `fpga-silicon-measurements-for-paper.md`
§7c) and on silicon by every measured cell, and the seven SQLLogicTest files (10,807 records, 8,746 checked
answers) are identical to the native x86 baseline on silicon and under the emulator (2026-09-07). The board's own
native-vs-Sublet arm agreement, E1's identical repetitions and R1's cycle-exact determinism are corroboration
only. Separately, the MEASUREMENT values — cycles and counters — are CSR reads and aligned counters, not loads
through the LSU path, so they are not exposed to this mechanism at all; the oracle argument is about the computed
results. The weaker half: the rv64imac toolchain rarely emits misaligned accesses, and none of the ported
allocators does.

**Readings elsewhere that rested on an absent plain-access trap (swept 2026-09-15):** the 2026-08-04 bounds
probes (`ob3`/`ob5`/`obb`/`oba`/`obn`, SILICON-BLOCKER.md, "the entire block is inert in our domains"), R-30's
and R-31's citation of that measurement, M-5's "inert-LSU" clause, and E1's s3/s5/s10/s11 (§7r). Each concluded
"no enforcement of plain accesses in a domain", and each conclusion SURVIVES — the mechanism now has two
independent halves (the privilege gate; this drop) where those entries name one. R-18 ("silently zeroed — no
trap") is a write-buffer data defect whose evidence is the zeroing, not the absent trap, and is unaffected.

## What it shows

In M-mode, with capmode set (witnessed) and `mstatus.MPRV = 0` (witnessed), so that the block's gate
`capmode_i && ld_st_priv_lvl_i == PRIV_LVL_M` (`load_store_unit.sv:966-967`) is satisfied and the
block demonstrably runs (its revnode-tracking side effect at `:985-989` updates), a directed test
drives its clauses and the ordinary misaligned check beside them. Every access RETIRES WITH A VALUE
and no trap:

| arm | access | expected if delivered | observed |
|---|---|---|---|
| control, before CAPENTER | `ld` through an integer base | the sentinel | `0x4C535550`, 0 traps |
| vector control | `.word 0` (illegal instruction) | cause 2, resume | **cause 2**, resumed (the only trap of the run) |
| misaligned `lw` at `buf+1` | plain load, odd address | cause 4 | **no trap**, value `0x004C5355` (the bytes at +1..+4) |
| in-bounds `ld` through a tagged NONLIN RW capability | plain load | the sentinel | `0x4C535550`, no trap |
| `ld` through a tagged NONLIN **write-only** capability | permission clause | cause 27 | **no trap**, value `0x4C535550` |
| `ld` at exactly `bound_end` through the RW capability | bounds clause | cause 28 | **no trap**, value `0x0BADB0B0` (the word past the buffer) |
| `sd` through a tagged NONLIN **read-only** capability | store permission clause | cause 27 | **no trap**, the store LANDED (`buf[1]` reads `0x5EC0DE`) |
| misaligned `sw` at `buf+1` | plain store, odd address | cause 6 | **no trap**, the store LANDED and corrupted `buf[0]` (`0x4C535550` → `0x1150`) |
| `ld` through an untagged base after CAPENTER | NOT_CAP clause (24 = DEBUG_REQUEST) | debug-mode entry | **no trap, no halt**, the sentinel |

And the one delivered exception shows the other half. After the two stores, `ld s4, 0(t1)` and
`ld s6, 8(t1)` (both through the untagged base) went in back to back: the second load's cause-24 held
`cap_exception.valid` high across the first load's IDLE→SEND_TAG boundary (t=1167..1173 in the
waveform), so the first load's exception WAS delivered — cause 24, this core's `DEBUG_REQUEST`: the
core entered the debug ROM (`0x800`…`0x890 dret`) with no debug request pending, and on `dret` the very
same instruction pair re-executed as two separate one-cycle pulses (t=2431, 2435) and completed with
values and no exception (RVFI: `x20 0x1150 mem 0x80003000`). The clauses are live; delivery depends on
whether an exception is still being presented one cycle after the request. (The RVFI line
`ILLEGAL_INSTR exception @ 0x80000198` is a STALE LABEL: `rvfi_tracer.sv:129-139` has no `default:`
arm, so an unknown cause keeps the previous string.)

## Mechanism (read at `f6ec6c198`; the drop site confirmed in the waveform by the audit)

0. In one sentence: the LSU raises its exceptions combinationally in the REQUEST cycle, the load unit
   pops the request in that same cycle and only looks for an exception in the NEXT one, and nothing in
   between holds the exception for it. At every one of the 21 fires `i_load_unit.ex_i.valid = 1` with
   `state_q = IDLE` and `ex_o.valid = 0`.
1. The exceptions are combinational on the LSU's current control word `lsu_ctrl`
   (`load_store_unit.sv:225`, the `lsu_bypass` output: `lsu_ctrl_o = lsu_req_i` while the FIFO is
   empty, `lsu_bypass.sv:111`): `data_misaligned`/`misaligned_exception` at `:820-951` and the
   capability block `cap_violation_detection` at `:957-1015`, merged into `misaligned_exception` at
   `:951`.
2. On an immediate data-cache grant the load unit pops that entry IN THE REQUEST CYCLE
   (`load_unit.sv`, IDLE with `dtlb_hit_i` and `req_port_i.data_gnt` → `pop_ld_o`); the store unit
   likewise. The waveform shows `lsu_valid_i`, `pop_ld`, `data_misaligned`, `lsu_cap_type` and
   `lsu_ea_full` all valid for exactly that one cycle (`sim/vcd-timing.txt`).
3. The MMU forwards the exception UNREGISTERED — `lsu_exception_o = misaligned_ex_i`
   (`cva6_mmu/cva6_mmu.sv:514`) — but asserts the request valid ONE CYCLE LATER, from a register:
   `lsu_valid_o = lsu_req_q` (`:513`, `:741`). `pmp_data_if` passes both through to
   `mmu_exception`, which is the load unit's and the store unit's `ex_i` (`load_store_unit.sv:585,
   :656`).
4. The units sample `ex_i` in that later cycle only: the load unit in `SEND_TAG`
   (`load_unit.sv:718`, whose own comment at `:715-717` says "an exception arrives one cycle after
   dtlb_hit_i is asserted"; `:423-424` pops in IDLE without consulting `ex_i.valid`), the store unit
   while `state_q != IDLE` (`store_unit.sv:349`; `ex_o = ex_i` at `:254`, `valid_o` is the qualifier —
   zero store-side deliveries in the run). By then `lsu_ctrl` has been popped and re-evaluates as
   empty (`lsu_ea_full` reads 0 in the waveform), so `misaligned_exception.valid` is 0 and nothing is
   delivered. It IS delivered when something keeps the exception asserted into that next cycle — in
   this run, a second faulting request presented back to back — which is also why the delivery is
   then attributed to the responding load's `trans_id` while the cause comes from the next request
   (`load_unit.sv:718-721`; both loads were NOT_CAP here, so the run cannot show the mis-attribution).
5. Where it came from: before upstream `23355d29f` ("Pmp/extracted pmp master (#2528)") the MMU
   registered the exception with the request — `misaligned_ex_n = misaligned_ex_i;
   misaligned_ex_n.valid = misaligned_ex_i.valid & lsu_req_i; lsu_exception_o = misaligned_ex_q` —
   and that commit replaced it with the combinational forward (its diff removes `misaligned_ex_q`
   from `cva6_mmu.sv` and the registered `mmu_exception <= misaligned_exception` from the LSU). The
   fork carries the post-#2528 form. The non-MMU configuration still registers
   (`load_store_unit.sv:459`, `pmp_exception <= misaligned_exception`); the MMU configuration this
   core is built with does not.

## What it explains, and what it does not

* **It is not capability-specific.** `rv64mi-p-ma_addr` from the stock riscv-tests list
  (`verif/tests/testlist_riscv-tests-cv64a6_imafdc_sv39-p.yaml:327`) fails on this RTL with capmode
  never set: `*** FAILED *** (tohost = 10) after 719 cycles` — tests 2–9 (`lh`/`lw`/`lwu` inside one
  8-byte word) pass because the data cache returns the shifted bytes correctly, and TESTNUM 10 (the
  `ld` at +1, crossing the word) returns wrong data with no trap. The capability clauses are lost by
  the same path the stock misaligned exception is lost by.
* Every in-domain and in-monitor reading of "plain loads and stores are not checked" — E1's stale
  loads retiring, the monitor's own rdtime emulation storing through an untagged base at every
  Linux clock read without halting — is this, not only the privilege gate. The gate keeps domains
  off the block; this loses the block's exceptions everywhere else. Both are true; neither alone
  explains the monitor.
* Misaligned plain accesses on this core silently complete with shifted data (wrong data when the
  access crosses an 8-byte word) unless a second exception happens to be presented one cycle later. In S-mode the trap would reach the monitor, whose
  `handle_exception` has no misaligned case (`EXCX` and halt). Which of the two a given misaligned
  access gets is decided by pipeline timing, not by the program.
* NOT shown here: the translation-on (S-mode, Linux) path was not driven (`ma_addr` and this test both
  run bare M-mode); the argument that it is
  the same is source-only (`cva6_mmu.sv:539` skips the translation branch on a misaligned request
  and leaves `:514` in force, and `lsu_dtlb_hit_o` is the same-cycle lookup). The board has not run
  this test. Stores were driven for the permission and misaligned clauses only.

## The fix is SUFFICIENT at the execute-stage boundary — measured 2026-09-15 (`sim/vcd-fix-boundary.txt`)

A sufficiency condition was raised against any R-34 fix: `ex_stage.sv:1013-1015` re-sources
`load_valid_o` from the DYN unit's load syncer (`forward_normal_load_valid`) and MASKS the LSU's
exception with it — `load_exception_o = forward_normal_load_valid ? load_exception : '0` — and the
syncer's message type carries only `trans_id` and `cap_result`, no exception. A one-cycle lag between
the syncer's valid and the LSU's exception would therefore lose every SINGLE-CYCLE exception while
every load still retired with correct data. Nothing in this folder discriminated it: the only delivery
ever observed here was a multi-cycle hold, and a restored SEND_TAG delivery is exactly the one-cycle
shape.

Measured on the RTL lane's fix branch (`r34-r24-exception-delivery` at `c77c65324`) with this folder's
own test: **it is not lost.** Each `i_load_unit.ex_o.valid` is high for exactly one cycle and is
followed, one cycle later, by `load_exception_o.valid` carrying the same cause, and one cycle after
that by `csr_regfile_i.ex_i.valid` — for causes 4, 24, 27 and 28 on the load side, with 6, 27 and 28
reaching the CSR from the store side. The reason is that the LSU registers `load_exception_o` in the
same spill register as `load_valid_o` (`load_store_unit.sv:685`), so the exception and the syncer's
forwarded valid arrive together and the mask never zeroes a live exception. `debug_mode_q` stays 0
across all four cause-24 deliveries, which is the renumber in the same batch doing its job — on the
unrenumbered tree those four would have entered the debug ROM.

## And the MISS path: also sufficient (RTL lane, `9a7bd598c`, `r34-coldmiss-deliver.S`)

One module below the boundary sat a second condition: the load unit delivers inside the
`req_port_i.data_rvalid` block while SEND_TAG asserts `kill_req` on `ex_i.valid`, so a faulting access that
MISSED would be killed, never see a response, and drop its exception — on the load side a silent read through
a capability that forbids it. **No run in this folder could answer that**, because the testbench's
`S12_MEM_DELAY` defaults to 0 and a response is always available in the tag cycle: the miss had never been
created. Nor does simply enabling the delay answer it — `lsu-mmode-gate` at delay 40 reproduces all thirteen
readings value for value with only the cycle count moving (868 → 2485), because the test hammers one buffer so
its faulting arms still HIT. **Turning the delay on is not the same as making the access miss**, and the first
reads like robustness. The matched pair that does answer it faults twice through the same write-only
capability, differing only in whether the line was brought in first, with every precondition witnessed in the
same run (the warm line resident, the warming access not itself trapping, the lines `0x1000` apart): warm arm
cause 27, **cold arm cause 27**, no data delivered on either, exactly two traps, identical at delay 0 and 40.
So the fix needs no companion RTL change. What stands between it and a bitstream is the monitor (this lane's
D3), and that cannot be validated on the deployed bitstream: with delivery broken the before and the after
both run clean, so only simulation of the fix branch can validate a monitor change.

## Fix direction (the RTL lane's call, not decided here)

Restore the pre-#2528 register in the MMU — carry the exception with `lsu_req_q` and mute it when
there is no request — or register `misaligned_exception` beside `lsu_paddr` in the MMU
configuration the way the non-MMU branch already does at `load_store_unit.sv:451-460`; or make the
load unit's IDLE pop consult `ex_i.valid` (`load_unit.sv:423-424`) and the store unit's `:349` not
require `state_q != IDLE`. Then the cause-24 collision (R-24) becomes visible on EVERY untagged plain
access in M-mode — including `RVTEST_PASS`'s own `sw` to `tohost` (it raised cause 24 at t=2515 in this
run) and the debug ROM's accesses — which is a second decision and is why the two issues are filed
apart. The blame for `cva6_mmu.sv:514` and the `load_unit.sv:715-720` contract is upstream CVA6; the
Capstone-specific part is a combinational check presented one cycle earlier than that contract expects.

## Reproduce (14 s once the Verilator model exists)

`src/lsu-mmode-gate.S` and `src/testlist_lsu_mmode_gate.yaml` go under `verif/tests/custom/capstone/`
and `verif/tests/` of capstone-ariane; `run.sh` is the exact container command (from the `rtl-sim`
skill; delete the previous run's artifacts first, and note that `SUCCESS` at the timeout is not a
pass). Readings are the `[Cycle N] Reg[k]: …` lines of the `.log.iss` and the `x<rd> <value> mem
<addr>` lines of the RVFI `.log`; `sim/readings.txt` holds the three runs' lines, `sim/vcd-timing.txt`
the original waveform extract, `sim/vcd-baseline-loadunit.txt` the load-unit signals on the unfixed tree
(21 fires, every one with `ex_i.valid = 1`, `state_q` IDLE, `ex_o.valid = 0`) and `sim/vcd-fix-boundary.txt`
the boundary reading on the fix branch (`TRACE_FAST=1`, read with `sim/readvcd.py`). The audited artifact set is the run of
2026-09-15 13:51:55: `.log.iss` sha256 `e9495d5e2ff104ce…`, RVFI `.log` `5670a53503d9c6af…` (a live
`verif/sim/out_*` directory is overwritten by every run; capture the hashes when you read).

## RTL lane, 2026-09-15: the store-bounds residual is CLOSED, and the translation residual is NARROWER than it looks

Taken over by the RTL lane. Run at the **flashed `1bfff7776`**, not at `f6ec6c198` — and the two were
checked to agree on the thing that matters rather than assumed to: the delivery condition
`ex_i.valid && (state_q inside {SEND_TAG, SEND_TAG_LDC}) && !…is_dom_switch` is **byte-identical text**
at `load_unit.sv:718` and `:699` respectively. The files do differ, by the S-07 probe ports and some
LDC-path comments, so "the files are identical" would have been wrong; the delivery logic is not.
**Every arm of run C reproduced first**, before anything was added.

### Residual (b), the store bounds arm — CLOSED, and it is the corrupting direction

`arm 4b` is added to `src/lsu-mmode-gate.S`: a `sd` at `bound_end` through the **same** RW capability
`arm 4` loads through. Same capability, same address, opposite direction, so a difference between the
two arms is about load-versus-store and nothing else.

    Reg[ 8]  arm 4b cause        0                 <- NO TRAP
    Reg[29]  guard word after    0x5B0             <- the store LANDED, past the buffer
    SUCCESS after 1303 cycles, 0 exceptions        (timeout 200000, so a real completion)

A store entirely outside the region is neither refused nor reported, and it overwrites the word past
the buffer. **The bounds clause is raised-and-dropped for stores exactly as for loads** — and this is
the direction that corrupts rather than merely leaks, which is why it was worth closing separately
rather than being assumed symmetric with `arm 4`.

### Residual (a), the translation-on path — MOOT for the capability clauses, live only for misaligned

The residual asks for a directed test with `satp` set. **For the capability clauses that test cannot
exist**, because translation-on and the gate are mutually exclusive by construction:

* data-access translation through MPRV requires `mstatus.mpp != PRIV_LVL_M`
  (`csr_regfile.sv:2292`);
* the load/store privilege is `mprv ? mstatus.mpp : priv_lvl` (`:2298`);
* so whenever MPRV turns translation on, `ld_st_priv_lvl != M` and
  `cap_violation_detection`'s gate **fails**. Without MPRV, M-mode data accesses are untranslated, and
  in S-mode the gate fails on privilege anyway.

**So no configuration reaches the capability clauses with translation enabled.** Writing that test
would have produced a clean-looking null result about the half it cannot reach — the shape this
project keeps paying for.

**What the residual does still cover is the misaligned causes 4/6**, which are not gated on capmode or
privilege at all and therefore *can* be exercised with `satp` set. That is the test worth building,
and it is a different test from the one the residual's wording implies.

### Built, and residual (a) is now CLOSED for the part a test can reach

`src/r34-misaligned-xlate.S`, run E. M-mode execution throughout so the harness, the vector and
`tohost` behave normally; **data accesses only** are translated, via `MPRV=1` with `MPP=S` and an
sv39 identity map (one 1 GiB megapage at `root[2]` covers code, stack, page table and buffer).

    X0   misaligned ld, translation OFF   shifted value, cause 0, no trap
    witness  mstatus                      0x20800 -- MPRV set, MPP = S
    precondition  aligned ld, translated  returns the sentinel, cause 0  <- the map WORKS
    X1   misaligned ld, translation ON    shifted value, cause 0, no trap
    total traps                           0            537 cycles / 200000 timeout

**Both arms silent with the mapping witnessed working — the first branch of the pair written down
before the run. The misaligned drop is INDEPENDENT of translation**, which is what the mechanism
predicts, since the MMU forwards `misaligned_ex_i` unregistered either way.

**The first run was refused by its own precondition, and the refusal is the point.** The aligned load
returned 0 with cause 5 (access fault), so translation was not working, and that run's X1 cause of 4
was taken under a broken mapping — reported as-is it would have said *"misaligned IS delivered under
translation"*, which it does not show. The cause was `MPRV`+`MPP=S` making the effective privilege S
for data accesses, so **PMP began applying where it had not**, and the `p` environment configures
none because it never leaves M-mode. One NAPOT entry covering everything fixes it. Recorded rather
than quietly patched: a reader building a similar test will hit the same thing.
