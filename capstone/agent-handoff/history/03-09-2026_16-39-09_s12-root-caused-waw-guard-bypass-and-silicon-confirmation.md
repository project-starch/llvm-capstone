# S-12 root-caused: the WAW guard is released by a stalled capability store

Date: 2026-09-03. Status: mechanism established in RTL simulation and confirmed on silicon.
Candidate fix validated against the reproducer; NOT synthesised, so not ready for a board.

## The defect

A capability store decodes its scoreboard destination to its own store-data register
(`decoder.sv:1313`, `rd := instr.rtype.rs2`; a plain integer store has `rd = x0`). This is
INTENTIONAL and must not be removed: `capstone_dyn_unit.anvil:458-462` clears rs2 to `cnull` when
the stored capability is linear-family, so a capability store has to null its source register.

When such a store stalls on a full store buffer, `commit_stage.sv` asserts `we_gpr_o[0]` (:323)
and then clears **only** `commit_ack_o[0]` (:346). `we_gpr` is never retracted, so
`we_gpr = 1 / waddr = rX / ack = 0` persists for the whole stall while the entry stays live and
unretired. That signal reaches the issue stage (`cva6.sv:1993 -> 1728`), where the WAW guard is
cleared by `we_gpr_i[c] && waddr_i[c] == rd` (`issue_read_operands.sv:1637`) — a clause whose own
comment says it tests that the register "will be written in this cycle by the commit stage",
which is exactly what is false during the stall.

So a younger `ldc rX` issues while the older store still claims `rX`. Forwarding candidacy needs
`still_issued & sbe.valid` (`issue_read_operands.sv:719-726`): the written-back store entry
qualifies, the unproduced load does not. The consumer is handed the store's result, which for a
null source is `{cursor 0, cap_type 0}` — and the FLU rejects that as UNEXPECTED_OPERAND with
`tval = 0`. That is the S-12 signature exactly.

## Evidence

**Simulation**, `+define+S12_MEM_DELAY=40`. The reproducer had existed for a day and had never run
under memory latency, so its store buffer could never fill and its `hazard = 0` was a test that
never created its own triggering condition.

| arm | delay | flu-issues | ldc-pending-cycles | hazard | in-loop traps |
|---|---|---|---|---|---|
| `stc-ldc-sbpressure` | 0 | 529 | 1800 | 0 | 0 |
| `stc-ldc-sbpressure` | 40 | 529 | 64279 | **254** | **254** |
| `stc-ldc-sbpressure-norel` | 40 | 529 | 64256 | 0 | 0 |

Rows 2 and 3 are the comparison: identical FLU issue counts, LDC-unproduced windows agreeing to
0.04%, opposite outcomes. Do NOT use `escape` as the precondition — `hazard` is counted inside its
if-body, making it a strict superset of the outcome, and it is pinned at once-per-iteration by the
loop shape.

**What the register holds.** `stc-ldc-sbpressure-a4` adds `CAPPRINT(a4)` as the first instruction
of the trap handler. At all 254 traps a4 held a real capability — cursor `0x80004000`, revnode 2,
type 2, perm 7 — identical every time. The load lands; only the consumer is misfed. The one trap
where a4 is genuinely null (the ARM P control, cycle 591) produces no capability print, so the
instrument can give both answers.

**Silicon.** A SQLite domain built with an in-domain trap handler that folds the answer into its
report word, boarded behind a known-good control, per-draw sha verified inside the boot:

    draw 4   obs=0xE643D221   marker 0xE, mcause 25, _start+0xF4884

Pre-registered before the boot: `0xF643D221` would have meant the load returned the null;
`0xE643D221` means a4 is non-zero. The offset moved `0xF4874 -> 0xF4884`, exactly the 16 bytes of
the four added handler instructions. Draws 1-3 returned an ordinary result (`obs=0x9E11`); this
build trapped 1 of 4 where its 3/3 baseline differs by those 16 bytes, and the defect is
documented as perturbation-sensitive.

## What this refutes

The repro folder recorded "(b) wrong-address forward" as the better-supported of two
memory-ordering accounts, on the grounds that it "predicts `{cursor 0, NOT_CAP}` EXACTLY, which is
what is observed". The prediction was right and the inference from it was wrong: `tval` reports the
operand **as the execution unit ingested it**, not what the load returned, and those differ
precisely when the value is forwarded. Both simulation and silicon now show the load landing.

The folder's own objection to (b) was already on the page — the store and load addresses in that
window are 176 bytes apart — and the register-relation 2x2 it ships never fitted an address-keyed
mechanism.

## Why it surfaces at one address

Counted over the built SQLite domain (331,808 instructions): the bare `stc rX` / `ldc rX` alias
appears 2832 times (12.8% of stores) and is harmless, because with no immediate capability
consumer nothing type-checks the stale operand. The full exploitable triple — store, load, and a
capability consumer reading the register — appears **68 times**, and exactly one of those is on
`a4`: the S-12 site.

## Candidate fix

Require the acknowledgement in both WAW-clearing clauses. Reproducer: 254 in-loop traps -> 0,
hazard 254 -> 0, duplicate-live-rd 64285 cycles -> 0. `commit_ack_i` needed no new top-level
wiring.

A first attempt patching only the `rs1` clause changed nothing, byte-identically. A probe added in
the same commit says why: `rd-match-while-unacked = 10419` against `rs1-match-while-unacked = 6`.
Without it, "the fix does nothing" and "the fix was never exercised" are indistinguishable.

**Readiness.** `rtl-lint-gate.sh` PASS (LATCH 52, MULTIDRIVEN 3, ALWCOMBORDER 0, COMBDLY 0,
UNOPTFLAT 39, BLKSEQ 2, UNDRIVEN 25, UNUSEDSIGNAL 719, ANVIL_UNOPTFLAT 0). Synthesis NOT run.
`issue_read_operands.sv` sits INSIDE the standing combinational-loop cone at `scoreboard.sv:129`
(example path: `scoreboard.sv:167` -> `issue_instr_o` -> `issue_read_operands.sv:645` ->
`rs1_fwd_req` -> `rr_arb_tree`), and the edit adds a term crossing a module boundary into a stall
that feeds `issue_ack_o` back to `issue_pointer`. Lint cannot see that; only synthesis can.

## Open

* The frequency gap: simulation fires 254/254 deterministically on a 737-cycle period, the board is
  ~54% per draw. Not claimed either way.
* Whether the other 67 triples are reachable, and whether any has ever fired.
