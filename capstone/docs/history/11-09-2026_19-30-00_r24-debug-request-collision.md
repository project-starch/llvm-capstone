# R-24 is refuted by its own gate: the spec's capability cause base collides with this core's debug cause

**2026-09-11, RTL lane.** R-24 proposed moving the two execute-path exception encoders from base 24
to base 23, so that the FLU and DYN paths would report the spec's capability cause numbers and stop
disagreeing with the load/store unit, which already emits them raw. The change was written on
2026-09-10 as `69658cf16` and explicitly **not gated**: its own message named the test updates, the
lint gate and the full sweep as owed. Gating it is what produced this note.

**The fix cannot ship as written.** Under base 23, `UNEXPECTED_OPERAND` — ordinal 1, and by a wide
margin the most common capability exception in the directed suite — comes out as `mcause` **24**. In
this core 24 is already spoken for:

```
core/include/riscv_pkg.sv:348
  localparam logic [XLEN-1:0] DEBUG_REQUEST = 24;  // Debug request
```

and the trap logic branches on exactly that value. The path to `mtvec` is gated to exclude it, and a
separate branch takes it into debug mode:

```
csr_regfile.sv:2019
  if ((CVA6Cfg.DebugEn && !debug_mode_q && ex_i.cause != riscv::DEBUG_REQUEST
       && ex_i.valid) || (!CVA6Cfg.DebugEn && ex_i.valid)) begin

csr_regfile.sv:2208
  if (ex_i.valid && ex_i.cause == riscv::DEBUG_REQUEST) begin
    ... debug_mode_d = 1'b1; set_debug_pc_o = 1'b1;
```

So after R-24 an ordinary capability operand error **does not reach the trap handler at all**. It
puts the core into debug mode and jumps to the debug ROM.

## Measured, with a one-variable control

| tree | encoder base | tests assert | result |
|---|---|---|---|
| `1bfff7776` (pre-R-24) | 24 | 25 | `cincoffset-linear-clear` **PASS**, 686 cycles, 2 exceptions; `excode-base-audit` **PASS**, 617 cycles, 2 exceptions |
| `r24-excode-base` | 23 | 24 | both, and six more, **HANG** at 400013 cycles |

400013 is the `+time_out=` value, so those are the harness's SUCCESS-at-timeout and not passes. The
trace settles the mechanism rather than leaving it inferred: the test's own trap handler at
`0x80000220` **never retires a single instruction**, while the core spins in the low debug-ROM
addresses reading `dscratch0`. The handler is never entered because the exception was never
delivered to `mtvec`.

## A second, independent defect in the same commit

The `riscv_pkg.sv` localparam block the commit edited is now internally inconsistent, and worse than
before the edit — which matters because the stated purpose of touching it was to stop the constants
misleading the next reader:

```
UNEXPECTED_OPERAND_TYPE = 25      left alone
INVALID_CAPABLITY       = 25      moved down -- now a DUPLICATE of the line above
ILLEGAL_OPERAND_VALUE   = 30      left alone; should be 29 under the new base
```

Four of the six were moved; two were missed, and one of the misses now collides with one of the
changes.

## RETRACTION

`69658cf16`'s message records that `host-sweep.sh` returned `TIMEOUT 400013` for all nine tests it
was given on that worktree, and calls those nine rows **"a harness failure, not readings"**. They
were readings. That is this defect, seen and then attributed to the instrument.

The standing rule in `CLAUDE.md` is that a surprising *clean* result should make you suspect the
instrument. This is the same rule running the other way, and it is worth stating in that form: a
surprising *failure* invites the same suspicion, and the suspicion has to be **discharged by a
control**, not settled by assumption. The control here was one run of the same test on a pre-R-24
tree. It costs about ten minutes and it is what turned nine "harness failures" into a refutation.

## What this leaves

R-24 now needs a **decision, not a test pass**, because the spec's capability cause numbering and
this core's debug cause want the same number. Neither of the obvious routes is a lane's call:
renumber the debug cause, or accept that the execute path cannot sit on the spec base in this core
and close the LSU/execute disagreement the other way instead.

Two things narrow the decision usefully:

* **The problem is only at 24.** Every other capability cause under base 23 lands in 25-29 and
  collides with nothing. It is the single lowest ordinal that is unusable.
* **R-24 has no firmware half.** The monitor's trap entry dispatches only on the interrupt bit and
  supervisor-ecall; it reports `mcause` and never branches on 24-30. So unlike R-30/R-31 this one
  would not have to ship RTL-and-firmware together — which also means nothing in the monitor would
  have caught the collision either.

The test updates R-24 owed were done anyway and are committed at `c7b616b6e` on `r24-excode-base`,
anchored at `backup/r24-tests-collision-2026-09-11`, because the arithmetic is correct for any fix
that puts the execute path on the spec base and the survey behind it is the expensive part: 31 cause
assertions across 17 files, each verified against its expected current value before being changed,
plus 15 comment expectations and a rewritten header for `excode-base-audit`. Five comments recording
numbers actually **observed** on pre-R-24 silicon were annotated rather than changed, since editing
them would falsify a record.

**Every one of the 32 assertions is raised by the execute path and none by the load/store unit** —
cleaner than expected. `LDC` and `STC` never reach the LSU's raw-numbered block at all: the decoder
gives them `fu = CAPSTONE_DYN` (`decoder.sv:1303`, `:1309`) while that block is gated on
`lsu_ctrl.fu inside {LOAD, STORE}` (`load_store_unit.sv:950`).

**Still owed if R-24 is ever revived.** Six cause-asserting tests exist only on the newer main line
and are absent from this branch, so a forward merge must also decrement
`ldc-consumer-stale-rs1.S:88`, `ldc-consumer-stale-rs1-miss.S:83`, `stc-ldc-dirty.S:98`,
`stc-ldc-miss.S:94` and `stc-then-ldc-same-reg.S:83` (25 → 24) and `seal-minsize-boundary.S:86`
(27 → 26). `s06sec-amo-no-resurrect.S` was edited but appears in **no** testlist, so nothing
exercises it. The full 89-test sweep has not been run, and running it before the collision is
resolved would measure the collision rather than the change.

The lint gate passes at the committed baseline (LATCH 52, MULTIDRIVEN 3, ALWCOMBORDER 0, COMBDLY 0,
UNOPTFLAT 40, BLKSEQ 2, UNDRIVEN 25, UNUSEDSIGNAL 717, ANVIL_UNOPTFLAT 0). That was never in doubt
for two changed literals and says nothing about the defect above — which is the point worth carrying
forward: **lint and audit are necessary and not sufficient, and here even a passing lint sat beside
a change that breaks exception delivery outright.**
