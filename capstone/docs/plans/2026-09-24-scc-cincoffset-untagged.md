# SCC and CINCOFFSET on an untagged value: trap, or compute like CHERI?

*2026-09-24. A decision for the project lead, with the spec's owners and the hardware side, and meant
to be taken **together with Q-04** (`2026-09-24-q04-movc-integer-source.md`). Both are RTL changes
of the same kind ("an untagged value is data"), and taken together they ride one synthesis cycle.*

## The question

`CINCOFFSET`, `CINCOFFSETIMM` and `SCC` raise *Unexpected operand type (24)* when `x[rs1]` is not a
capability (`capstone-academic-spec`, `cap-man-insn.adoc:65-66`, `:97-98`, `:131-132`). CHERI's
equivalents take any value: on an untagged operand they compute the new address and return an
untagged value, and the tag, which was never there, stays clear. Should Capstone do the same?

| | untagged `rs1` today | where |
|---|---|---|
| spec | exception 24 | `cap-man-insn.adoc:65-66`, `:97-98`, `:131-132` |
| RTL | exception 24 | `capstone_flu_unit.anvil`: `CINCOFFSET` :29, `CINCOFFSETIMM` :57, `SCC` :92 |
| QEMU `cincoffset`/`cincoffsetimm` | exception 24 (a diagnostic knob, `CAPSTONE_CINC_UNTAGGED_SURVIVE`, logs and continues) | `op_helper.c` |
| QEMU `scc` | **`assert`: the whole emulator aborts**; exception 24 with capstone-qemu#8 | `op_helper.c` `helper_csscc` |

## What it costs today: little

- The suite was run once with `CAPSTONE_CINC_UNTAGGED_SURVIVE=1`, which reports every untagged
  `cincoffset` instead of trapping at the first. Forty-nine libc-test tests produced **one** site:
  musl's `__fputwc_unlocked`, arithmetic on a null `FILE` cursor. It is replaced in the port's
  runtime (`fputwc_null_safe.c`, commit `1247922c33d5`).
- Earlier, LICM hoisted `&p->field` above a `p != NULL` guard. Generic LLVM was changed so that such a
  GEP is not speculated when its base may be null, at +0.13 % instructions on SQLite (C-19 in the
  archive).
- The compiler emits `SCC` only on the stack pointer and for the `capstone_cap_scc` intrinsic. No
  untagged use of either is known.

So no program we build needs the change now.

## Why decide it now anyway

1. **intcap needs it.** Under a capability-carrying `uintptr_t`, `u + 8` on a `uintptr_t` that holds
   an integer is a `cincoffset` of an untagged value. With the trap, the compiler must branch on the
   tag at every such operation, and a register copy cannot be guarded at all
   (`plans/intcap-uintptr-model.md` on branch `compiler/intcap-design`, WP1).
2. **Batching.** It changes the same unit as Q-04's one line (the FLU). Deciding it later costs a
   second synthesis cycle and a second reflash.
3. **The spec has already moved this way once.** `LCC`'s type query was made total on 2026-08-12
   (`cap-man-insn.adoc:204-220`): it answers `7` for a non-capability instead of raising. That was
   the same judgement, that asking about an untagged value is not an error.

## What is lost

The early trap. Integer or null pointer arithmetic fails today at the arithmetic, and with the change
it fails at the first dereference: the result is untagged and every access through it still traps.
Both findings above came from that early trap. **No authority is created either way.** An untagged
operand yields an untagged result, and a tagged operand is handled exactly as today (types 3 and 4
still raise 26, and MOVC's rules apply). The early trap is worth something for debugging, but it is not a
security property, and the QEMU knob above keeps a way to find such sites.

## Recommendation

**If intcap is the direction** (its plan recommends CHERI's model): adopt the CHERI rule for all
three, now, and put it in the same bitstream as Q-04. **If it is not:** keep the trap and revisit
with intcap. Q-04 does not depend on this. This one depends on Q-04: the instructions are defined
through `MOVC rd, rs1`, and an untagged `rs1` must be left alone, which is Q-04's (b).

**Regardless of the decision:** QEMU's `scc` must raise 24 instead of aborting, as the RTL does
today. That is a QEMU fix with no ISA question in it, the same one `cincoffset` already received. It
is capstone-qemu#8.

## What to request, if adopted

**Spec**, for each of the three: drop *"`x[rs1]` is not a capability"* from exception 24 (keep
*"`x[rs2]` is not an integer"*). For CINCOFFSET, prefix the semantics with:

```
. If `x[rs1]` is not a capability, write `x[rs1] + val` to `x[rd]` (an integer) and stop.
```

For CINCOFFSETIMM the same with `imm`; for SCC, `write val to x[rd]`.

**RTL**, in the three functions: the `NOT_CAP` arm returns an integer result instead of raising.
That is metadata `NOT_CAP` (as `create_cnull`), cursor `rs1.cursor + val` for the two increments and
`val` for SCC, with `rs1` unchanged. The adder already exists for the capability arm. This is more
than Q-04's one condition, a result arm in each of three functions, and the RTL gates apply:
- lint, including the combinational-loop list (the FLU is Anvil, and the baseline has
  `ANVIL_UNOPTFLAT 0`);
- a directed test per instruction with an untagged `rs1`;
- synthesis;
- a board pass.

**The probe, and the prediction for the board**, written before synthesis:
`capstone/tests/runtime-qemu/untagged-cap-arith/`.
- Case 0 runs `cincoffset` and `scc` on a real capability and stores through both results: the
  control, and the proof that the hand-encoded instructions are the right ones.
- Cases 1 and 2 run them on the integer `0x5000`.
- On the current bitstream, cases 1 and 2 raise cause 24. With the change they print `0x5008` and
  `0x6000` (`EXPECT=cheri`).
- A domain that traps ends its guest session, so it takes two boots, or two slots at the end of a
  batched board load, next to the stage-50 MOVC probe.

Under QEMU, measured 2026-09-24: with capstone-qemu#8, case 0 is ok and cases 1 and 2 raise 24
(exit 0). Without it, case 2 aborts the emulator on the `assert`. With `EXPECT=cheri` the probe fails,
as it must while the trap is the rule.

## What this does not decide

- Nothing about tagged operands. The type checks (26) and bounds behaviour stay as they are.
- Not whether intcap is built. That is its plan's own decision.
