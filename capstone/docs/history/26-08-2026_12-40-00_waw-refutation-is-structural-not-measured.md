# The duplicate-live-rd refutation is right, and its own evidence is the weak part

**Date:** 2026-08-26
**Verdict:** the mechanism is **refuted STRUCTURALLY**. Re-justify `4c0def314`; do not downgrade
it, and do not write more tests against it.

## The claim under examination

`4c0def314` ("Refute the wrong-producer-forwarding lead by measuring its precondition") ruled out
a proposed S-12 mechanism: the forwarding arbiter selects among scoreboard entries by raw slot
index rather than by age, so if two live entries ever claimed the same `rd`, a consumer could be
handed the older producer's result. For `movc a4,zero` then `ldc a4,...` that is
{cursor 0, NOT_CAP} — the board signature exactly.

It refuted this by adding a sim-only checker counting cycles where two live entries share a
non-zero `rd`, and measuring zero.

## Why that measurement is weak

Its own commit message says the test "deliberately fills the scoreboard with twenty-four
long-latency divisions" and then reports "**peak occupancy never exceeds 2 entries of 8**". The
fill did not work. The precondition needs two live entries; occupancy peaked at exactly two, so
the test sat on the boundary of being able to observe anything at all. It also declares itself
scalar-only, leaving a capability-pair residual open.

That is the same shape as the void race test recorded earlier today: a clean number from a run
that barely created its condition.

## Why the conclusion survives anyway — and this is the durable part

**The precondition is structurally impossible, so no test could ever have observed it.**

    issue_read_operands.sv:1478   if (!stall_raw[i] && !stall_waw[i] && !stall_waw_rs1[i])
                                      issue_ack[i] = 1'b1;
    issue_read_operands.sv:1418   stall_waw = '1;                    // fail-closed default
    issue_read_operands.sv:1427-31 cleared ONLY if rd_clobber_gpr[rd] == NONE

`rd_clobber_gpr[rd]` is non-NONE whenever any live scoreboard entry claims that `rd`, so an
instruction whose `rd` is already claimed is never acked. Two live entries sharing a non-zero
`rd` cannot arise — independent of occupancy, instruction mix, or whether the producer is scalar
or a capability op. That last point closes the scalar-only residual too: the stall is register-file
machinery and is blind to the producer's type.

**The one escape closes as well.** `stall_waw` is additionally lifted when the commit stage writes
that `rd` in the same cycle (`issue_read_operands.sv:1440-1446`) — the only window in which a
duplicate could momentarily exist. It does not, because `scoreboard.sv:273-278` clears the old
entry's `issued` in the **same `mem_n` update** that the new instruction's ack sets its own slot.
Both land in `mem_q` together; at the next cycle only the new entry is live.

## The consequence worth acting on

**The checker in `4c0def314` counts a condition the hardware cannot enter.** So:

- a positive control for it is **unsatisfiable**, and its absence is not a detector defect;
- a future zero from it is **not evidence of anything** and must not be cited as one;
- the capability-pair re-run that two lanes independently wanted is **unnecessary** — it would
  have produced another structurally-guaranteed zero and looked like confirmation.

This is a distinct failure mode from the usual one. The standing rule is "a detector that has
never fired is unproven". Here the detector is unprovable *and* the conclusion is still correct,
because the impossibility is visible in the RTL. Measurement was the wrong instrument for a
structural question, and reading the issue logic answered in minutes what a sim campaign could
not have settled at all.

## What is NOT claimed

Nothing about the S-12 fault itself. Operand-delivery failure still stands as a description: a4
at the halt equals the slot's cursor exactly, 6 wedges of 6. What died is one proposed *mechanism*
for it. Separately, the four-instruction production shape (movc/stc/ldc/cincoffsetimm at
production spacing) was measured on silicon and returns clean, so the shape alone is not
sufficient either — the trigger needs context those instructions do not carry.

## Related

- `4c0def314` — the refutation being re-justified. Note it is **not** an ancestor of `84ed6eafb`,
  so the checker is absent from the shipping tree.
- [[26-08-2026_11-40-00_race-test-void-control-did-not-fire]] — same session, the weak-evidence
  shape this shares.
