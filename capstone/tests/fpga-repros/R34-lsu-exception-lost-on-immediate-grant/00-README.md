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
missing trap. The evidence that the corpus is unaffected is empirical, not "nobody noticed": every measurement
run was matched against an oracle that wrong data would have changed — SQLite `speedtest1` at sizes 1/20/100
reproduced the native result on both arms (§7l–§7p), the seven SQLLogicTest files (10,807 records) were identical
to native on silicon (2026-09-07), E1's 21 cells were identical across three repetitions (§7r), and R1's 420
records are deterministic to the cycle (§7t). The weaker half of the argument: the rv64imac toolchain rarely
emits misaligned accesses, and none of the ported allocators does.

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
the waveform extract (`TRACE_FAST=1`, read with `sim/readvcd.py`). The audited artifact set is the run of
2026-09-15 13:51:55: `.log.iss` sha256 `e9495d5e2ff104ce…`, RVFI `.log` `5670a53503d9c6af…` (a live
`verif/sim/out_*` directory is overwritten by every run; capture the hashes when you read).
