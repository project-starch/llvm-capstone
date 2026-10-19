# S-26: the simulated context-entry startup cost is the testbench memory model, and the gap to silicon closes at zero added latency

*RTL lane, 2026-10-19. Answers the board lane's question of this evening — "something in the simulated entry path
costs ~600-1,000+ cycles that silicon does not, and that is yours to find" — and corrects an attribution of mine
that was computed at the wrong memory delay. Continues
`19-10-2026_15-30-00_quantum-zero-budget-s26-rtl-half.md`, whose ladder ran only at `S12_MEM_DELAY=12`.*

## What was measured

Nineteen arms on `bf3db3175` (the RTL the vp2 bitstream was built from), the same arm source throughout, with the
context's quantum budget set by a compile-time define so one model build covers a whole ladder. Three memory
delays. The reading is the context's loop counter: 0 = retired nothing, 10 = the loop completed; the escape kind
follows it (1 = the quantum expired, 3 = the context's own ecall).

| budget | `S12_MEM_DELAY=0` | `=12` | `=40` |
|---|---|---|---|
| 100 | nothing | nothing | nothing |
| 150 | **counter 4** | nothing | nothing |
| 200 | counter 10, kind 1 | nothing | nothing |
| 300 | counter 10, kind 3 | nothing | nothing |
| 400 | counter 10, kind 3 | nothing | nothing |
| 1,000 | counter 10, kind 3 | nothing | nothing |
| 1,200 | counter 10, kind 3 | nothing | — |
| 1,500 | counter 10, kind 3 | nothing | counter 10, kind 1 |
| 2,000 | counter 10, kind 3 | counter 10, kind 1 | counter 10, kind 3 |

Every arm SUCCESS, `--sv_seed 1`, `+define+S12_MEM_DELAY=<d>+SUP_TRACE`, one model build per delay with the model
removed from inside the container first and the rebuild asserted (`Building Model` = 1 per run), since cva6.py
reuses a compiled model across define changes. The budget arms are `va-satpstale.S` with
`-DFENCE_BEFORE -DRESUME_PC -DQUANTUM_ESC -DQUANTUM_N=<n>` and the standard isa-macro include set; the testlist
entries live in a worktree rather than on a branch, so the options are written out here to keep the result
reproducible without it.

## What it says

**The whole sim-to-silicon gap is the testbench's memory model.** The budget at which the context first retires an
instruction is **100-150 cycles at zero added latency**, against a silicon figure of 51-57 quantum cycles per step
measured by the board lane (three arms, two completing within 6 cycles of each other). That is a factor of two to
three — the size cold caches and a real DRAM against a zero-latency model would predict — not the order of
magnitude that was left over when the sim figure was quoted at delay 12.

**So no further mechanism needs to be found**, and the earlier framing ("something else in the simulated entry
path costs ~600-1,000+ cycles") rested on comparing a delay-12 threshold with silicon. My own contribution to that
error was giving a memory attribution computed at a 40-cycle latency for arms that ran at 12; the fix was to stop
attributing and measure the intercept.

**The delay knob is not monotone, and that is now seen twice.** Delay 12 is consistently *slower* than delay 40:
222,648 cycles against 163,702 for this arm, and 8,548 against 6,516 for `s23-kill-window` the same evening. The
define sets `FIXED_DELAY_INPUT` and `FIXED_DELAY_OUTPUT` on the DRAM path
(`corev_apu/tb/ariane_testharness.sv:447-457`); something in that handshake interacts badly at 12. The
non-monotonicity is a testbench artefact and is not chased here, but it has one consequence that must be carried:
**"the simulated threshold" is not a single number and has to be quoted with its delay.** The recorded 1,500-2,000
is the delay-12 figure and is the worst of the three.

**The ladder measures a different quantity from the silicon E** — the budget at which the first instruction
retires, against the quantum charged per step — so the right statement is "at zero added memory latency the
simulated startup cost falls to the same order as the silicon figure", not "the simulation agrees with E".

## What is unchanged

The RTL half of S-26 stands as recorded: a context entering with a zero or sub-threshold budget retires nothing
and escapes kind 1 with cause 0, the caller resuming at CSRUNV+4, and the quantum's cause reads 0 in the frame
because register id 71 masks it (`core/csr_regfile.sv:497`, tval at `:499`). The simulated 1,500 and the board's
sub-1,000 were never in conflict and needed no extra mechanism to reconcile.

Also unchanged, and worth repeating because this session nearly spent a session on it: a simulated per-access cost
is a testbench parameter and transfers to silicon in pattern only, never in cycles.
