# S-12 operand-mux recorder: GO / NO-GO, written BEFORE any synthesis

**This document exists to be checked before a bitstream is built, not after.** The precedent it
guards against is concrete: the S-12 untagged-LDC recorder cost 6+ hours of synthesis for a reading
already derivable from `tval = 0` plus a decode sitting in its own repro folder. The prediction had
been written down; it simply was not used as a go/no-go input. **If the table below does not show
the live hypotheses producing DIFFERENT readings, the build does not go.**

## The live hypotheses, and whether each is still alive

| # | hypothesis | status | why it is still live (or not) |
|---|---|---|---|
| **H1** | the consumer is forwarded the **STC's** scoreboard entry, whose `cap_result` for a null capability is `create_cnull()` | **LIVE but WEAKER (2026-09-03)** | `decoder.sv:1313` decodes STC `rd := rs2`, so the store IS a producer; predicts `{cursor 0, NOT_CAP}` = `mcause 25` + `tval 0` exactly; predicts the three-register coincidence the board measured |
| **H2** | writeback-port displacement (S-07/A-1): the LDC retires through a scalar port, metadata zeroed | **WEAK but UNMEASURED** | forwards the REAL cursor, so predicts non-zero `tval`; we measure `tval = 0` in every wedge. Its synthesised detector could not be read — both apertures returned the same byte. Disfavoured by argument, excluded by nothing |
| **H3** | missed RAW hazard: the consumer reads the **register file**, getting the pre-`ldc` value left by `movc a4, zero` | **STRONGLY DISFAVOURED (2026-09-03)** | `{cursor 0, NOT_CAP}` bit-for-bit; the tree's `b-NOFORWARD` counter reads 0 but CANNOT be positive-controlled (the event it counts would itself be a reproduction), so its zero excludes nothing |
| **H4** | the operand path manufactures zero on both halves somewhere else | **LIVE** | nothing measured distinguishes it; it is the residual |

**Not live, and must not be re-tested by a bitstream:** the register relation (proven necessary),
register identity/ABI class (D3/D4), producer distance (D1/D2), the stored value (null throughout),
layout (NOP with byte-identical symbol table wedges 4/4), pure pipeline delay (same NOP control),
domain context (12,288 executions), store pressure (24,576), slot provenance (matched pair).

## UPDATE 2026-09-03 — read this before using the table above

The simulated memory had **zero latency**: the AXI delayer was instantiated with a hardwired 0, so
no reconstruction ever produced a slow load (7.0 cycles per load warm, 9.5 with conflict misses).
The outcome counter needs a live-but-UNPRODUCED LDC when its consumer issues, so that window barely
existed and every earlier zero was partly a statement about the testbench. With
`S12_MEM_DELAY=40` and a test carrying BOTH an STC and a genuinely missing load
(`stc-ldc-miss.S`), the load reaches **86 cycles** and the escape counter 4191 — and the outcome
counter is **still 0** across 1024 attempts.

**What that does to the hypotheses:**

* **H3 is strongly disfavoured.** At 86-cycle loads, unproduced LDCs are everywhere; if a consumer
  could issue against one without forwarding, it had 1024 chances. The RAW machinery held.
* **H1 is weaker but not dead.** Bare metal is not a capability domain, and NOTHING — simulation or
  silicon — has ever reproduced S-12 outside SQLite itself, so a bare-metal negative does not
  separate "the mechanism is wrong" from "the harness cannot create what SQLite creates".
* **H2 and H4 are untouched**, since neither was ever testable this way.

**Consequence for the build decision.** The go criterion (readings mutually distinct) still holds,
but the EXPECTED VALUE has dropped: the recorder would now be spending 90 minutes plus a reflash
largely to separate H1, H2 and H4, with H1 weakened. **Recommendation: do not build until the fence
workaround has been generalised across the corpus.** That work is cheap, is the actual project
goal, and may change what anyone wants to know about the mechanism.

## What the recorder must capture, at the cycle the FLU raises UNEXPECTED_OPERAND

1. `forward_rs1` for the raising port — was rs1 forwarded at all, or read from the register file?
2. the **winning forwarding source**: which scoreboard slot or writeback port the arbiter selected.
3. that slot's `sbe.op` — **STC or LDC**.
4. `operand_a` and the tag bit of `cap_metadata_a` as ingested.

## The predicted readings — THIS IS THE GO/NO-GO TEST

| | `forward_rs1` | winning source | its `op` | tag / cursor |
|---|---|---|---|---|
| **H1** | 1 | a scoreboard slot | **STC** | 0 / 0 |
| **H2** | 1 | a **scalar WB port** | LDC | 0 / **real cursor** |
| **H3** | **0** (regfile) | — | — | 0 / 0 |
| **H4** | 1 | a scoreboard slot | **LDC** | 0 / 0 |

**The four readings are mutually distinct in at least one field.** H3 is separated by `forward_rs1`
alone; H1 and H4 by the winning entry's opcode; H2 by both the source class and a non-zero cursor.
**GO condition met** — but only for a recorder that captures all four fields. A recorder capturing
fewer does NOT meet it: capturing only the tag, for instance, reads identically under H1, H3 and H4
and would repeat the previous waste exactly.

## Silent-failure guard, which is mandatory

An observation-only change that never fires leaves us worse off than before, because an absent
signal reads as a clean result. The recorder must therefore carry **two counters, precondition and
outcome, exactly as the existing `S12-ESC` instrumentation does**:

* **precondition** — any FLU consumer issuing whose rs1 is the `rd` of a live uncommitted capability
  op. This is the positive control and must be non-zero in any run that executes the window.
* **outcome** — the fault-cycle capture above.

A zero outcome is admissible ONLY if the precondition is non-zero. Without that pairing the
recorder cannot distinguish "did not happen" from "did not work", which is the single most
expensive mistake available here.

## Everything that must ride along in the SAME bitstream

Discovering a second question afterwards costs another full cycle, so:

* the four fields above, latched on the FIRST raise and held (not last-writer-wins);
* the precondition and outcome counters;
* the `trans_id` of the winning entry, to tie the capture to a specific instruction;
* a bit recording whether the LDC had written back when the consumer issued;
* **repair of the debug-mux aperture readback**, since `sw=204` and `sw=208` returned the same byte
  on a live core — a recorder that cannot be read is worth nothing, and this is the failure that
  wasted the previous mux attempt.

## Do NOT build if any of these is true

* the recorder captures fewer than the four fields (readings stop being distinct);
* it lacks the precondition counter (a zero becomes uninterpretable);
* the mux readback is not repaired in the same build (the data cannot be retrieved);
* a simulation has meanwhile been made to fire — a waveform is strictly better and free;
* any live hypothesis has been eliminated by cheaper means since this was written. **Re-read this
  table before building.**

## Cost and standing

~90 minutes of synthesis plus a reflash, and the reflash is the project lead's call. The RTL change
feeds observation-only logic; per the standing rule it must still pass `rtl-lint-gate.sh` and go to
synthesis before it goes anywhere else, and it must not add a term to any cone on the `UNOPTFLAT`
list.
