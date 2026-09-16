# The M-mode arm of the LSU gate question, run at the desk: the block's exceptions are lost on an immediately granted access (R-34)

2026-09-15, 13:00–14:00, the board lane, while the M2 chase boots ran on the board. Registry entry:
**R-34**; the folder that is the report: `capstone/tests/fpga-repros/R34-lsu-exception-lost-on-immediate-grant/`.

## The question as it arrived

The paper lane asked for one board arm: run the `obn` probe (`tests/runtime-qemu/silicon-ladder/obn_kernel.h`,
a plain store through an integer copy of a capability) from M-mode with capmode already set, to separate "the
LSU's plain-access checker is unreachable from a domain" (the privilege gate,
`15-09-2026_lsu-capmode-gate-why-domains-cannot-satisfy-it.md`) from "the checker is broken". Pre-registered by
both the RTL lane's note and the paper lane: it traps, cause 24. The paper lane added two design constraints
worth keeping: sample `mstatus` at the access (the gate reads `mprv ? mpp : priv`), and drive a second clause so a
negative is informative.

## What the desk read changed before anything ran

1. **Cause 24 cannot trap on this core.** `riscv_pkg.sv:348` has `DEBUG_REQUEST = 24`; `csr_regfile.sv:2008`
   keeps that value off the trap path and `:2197` takes it into debug mode. R-24 had recorded this for the
   execute path; the LSU block emits the raw spec numbers (`load_store_unit.sv:991-1009`), so its untagged-base
   clause, if live, halts the core into the debug ROM. The pre-registration was unmeasurable as written.
2. **The board already runs the arm continuously.** The monitor's rdtime emulation (`sbi_capstone.S`,
   `_handle_non_ecall`: `add t5, sp, t5; sd a0, 16(t5)`) stores through an integer-op result — untagged, since
   the metadata bank is written on every GPR write with `cap_result.valid ? metadata : 0`
   (`issue_read_operands.sv:1841-1848`, `commit_stage.sv:279`) — in M-mode with capmode sticky since the init
   `CAPENTER`, at every Linux clock read. If the clause fired there the board would halt at the first `time` read.
3. So the arm went where it is expressible: a directed RTL test (the `rtl-sim` skill), not a monitor change.

## What ran (all in M-mode after `CAPENTER`; handler in registers only)

`lsu-mmode-gate.S`, 14 s per run, three runs of the same file as arms were added: gate witnesses (a CSCRATCH
round trip returns the capability, so capmode is set; `mstatus` reads `MPRV = 0`); an illegal-instruction
control (cause 2, caught, resumed — the only trap of every run); an in-bounds load through a tagged NONLIN RW
capability (the sentinel); a load through a WRITE-ONLY capability (no trap, the sentinel); a load at exactly
`bound_end` (no trap, the word past the buffer); a misaligned `lw` (no trap, shifted bytes); a store through a
READ-ONLY capability (no trap, it landed); a misaligned `sw` (no trap, it landed and corrupted the neighbour);
a load through an untagged base (no trap, no halt, the sentinel). Then the waveform (`TRACE_FAST=1`) showed the
gate inputs at the LSU exactly as required (`capmode_i = 1`, `ld_st_priv_lvl_i = M`) and the block RUNNING
(its revnode tracking updated at the first tagged load) — and it showed the loss: every exception input is
valid for the one request cycle, the entry is popped that cycle on the cache's immediate grant, and the MMU
asserts the request valid one cycle later with the exception recomputed from an empty `lsu_ctrl`
(`cva6_mmu.sv:514` forwards `misaligned_ex_i` unregistered; `:513` registers only the valid; the load unit
pops in IDLE without consulting `ex_i.valid`, `load_unit.sv:423-424`, and emits only in SEND_TAG, `:718`).
The claim-auditor read the same waveform independently and refuted my first wording ("does not fire"):
`cap_exception.valid` rises 21 times in the run with causes 24/27/28 and is delivered ONCE — the first of two
back-to-back untagged loads after the stores, whose successor held the signal across the boundary; not, as I had
first written, a load the cache did not grant at once. That one delivery entered the debug ROM (cause 24 is
`DEBUG_REQUEST`; no debug request was pending) and came back by `dret`; the same pair then re-ran as two separate
pulses and completed silently. Upstream `23355d29f` (#2528) is where the MMU's `misaligned_ex_q` register went
away. Then the stock `rv64mi-p-ma_addr` was run with capmode never set: it FAILS (TESTNUM 10, the `ld` crossing an
8-byte word, returns wrong data; the in-word cases pass on the cache's shifted bytes) — the loss predates the
capability check and is the LSU's exception delivery itself.

## What is retracted, and by whom

* The RTL lane's "**the block is live for the trusted monitor's own accesses and dead for untrusted domain
  code**" is superseded: the gate keeps domains off the block, and R-34 loses the block's exceptions everywhere
  else. The paper lane had endorsed that framing and retracted it on reading this result; this lane had written
  it into the S1S2 manifest's scope note earlier the same day and has corrected the note.
* My first wording, "the block is dead when reached", was refuted by the audit before it was pushed: the block
  fires; delivery is lost. Same observable, different defect, and the difference is what the RTL lane needs.
* The pre-registration "traps, cause 24" was wrong in the dangerous direction — a live clause could never
  have produced it — and was caught before a boot was spent. The paper lane's own note names the pattern:
  a cause number in an expected-value field is a source-derived constant and needs the commit it was read at.

## What it means for the manuscript's safety matrix

The four plain-data-access rows are unsupported on this configuration for two independent reasons (the gate
and R-34), and enabling the gate for domains would not have enforced them. Capability accesses are the DYN
unit's path with no privilege gate and no LSU exception, so the enforced rows (REVOKE's fault, the type reads)
stand. Misaligned plain accesses on this core are decided by pipeline timing: silent shifted data (wrong across an
8-byte word) unless a second exception is presented one cycle later, and then a trap the monitor cannot service (`EXCX`).

## Process

The auditor could not re-examine the run the claim cited: I re-ran the simulation into the same `out_*`
directory twice while the audit read it. A claim built on a live simulation directory needs its artifact hashes
captured at the moment of reading (the folder now carries them); the rule already exists for board results
(cite by image hash) and applies to simulation logs the same way.

## Open

The translation-on path (S-mode, Linux) is argued from source, not driven; the board has not run the test;
stores were driven for the permission and misaligned clauses only. The fix is the RTL lane's call and is
filed apart from R-24 on purpose: once the exception is carried with the request, the untagged clause will
enter debug mode on every M-mode plain access through an integer register, which is the R-24 decision.
