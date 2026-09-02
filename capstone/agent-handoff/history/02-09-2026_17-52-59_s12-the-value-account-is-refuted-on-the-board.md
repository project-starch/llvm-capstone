# S-12: the stale-VALUE account is refuted on the board, and the cure is somewhere in five instructions

## The ladder

Three arms, all built from the pinned base (`69fe70b7…`) by patching the real faulting binary, all
passing the QEMU functional gate with `SLT-SUMMARY` byte-identical to the base, all run in the same
slot of a two-domain image with the same known-good control in slot 0.

| arm | `[32]` | `[33]` | `a4` into the load | nulls stored from | result |
|---|---|---|---|---|---|
| **anchor** (unmodified base) | `movc a4, zero` | `stc a4, 0x0(a5)` | 0 | `a4` | **3 wedges / 3 valid** |
| **control** | `movc a4, zero` | `stc t0, 0x0(a5)` | 0 | `t0` | 0 wedges / 4 valid |
| **sentinel** | `li a4, 0x5a5` | `stc t0, 0x0(a5)` | `0x5a5` | `t0` | 0 wedges / 4 valid |

All three anchor wedges carry the canonical signature and nothing else: `mepc 0x828f4814` → VA
`0x104814`, `tval 0`, trap log `0x99` = mcause 25, domain entered, per-run sha correct, control
slot returning in 6 s. Every wedge in the campaign fell in the anchor's three draws and none in the
eight modified-arm draws: one-sided exact p = 0.0061.

A fourth anchor draw was lost, not excluded for cause. A duplicate runner -- spawned because a
chained waiter's parent was killed while its child survived -- truncated `anchor-1.log` while that
draw was mid-boot, so a real 8-minute board session produced no readable data. It scores VOID
rather than clean, which is the classifier working, but the boot is gone.

## What is refuted

**The stale-register-VALUE account.** The RTL admits a path where a consumer of an in-flight `ldc`
issues without the RAW hazard being detected, reads `a4` from the register file, and finds what
`movc a4, zero` left there two instructions earlier — `{cursor 0, NOT_CAP}`, which is the observed
operand bit for bit, `tval = 0` included. The sentinel arm was built to make that self-labelling:
it leaves `a4` holding `0x5a5` instead of zero, so the account predicts the SAME wedge rate with
`tval = 0x5a5`, a value nothing else in the image can produce.

The sentinel did not wedge at all, and it is identical to the control in **exactly one
instruction**. So the value `a4` carries into the load does not decide the fault. That prediction
was written down before the boot and it failed.

Two simulations agree, and they are the first non-void ones on this question. `s12-flu-raw.S` had
asked it and reported `ldc-pending-cycles = 0` with `flu-issues = 0` — its loads hit, so the window
never existed and its zero said nothing. `ldc-consumer-stale-rs1.S` runs the silicon triple 1024
times and does create it:

    warm   ldc-pending 7175 / 13880 cyc   b-consumers 1024   b-NOFORWARD 0   HAZARDS 0
    miss   ldc-pending 9712 / 27433 cyc   b-consumers 1024   b-NOFORWARD 0   HAZARDS 0
    +stc   ldc-pending 7176 / 25130 cyc   b-consumers 1024   b-NOFORWARD 0   HAZARDS 0

`b-NOFORWARD` counts precisely the account's precondition: an FLU op acked whose rs1 is a live
uncommitted LDC's rd, with no forward, so it reads the register file. Zero across all three, with
the guard proven live in each (ARM P trapped mcause 25 at cycle 328, the only exception in every
run) and the loop proven to have executed (the three encodings retire exactly 1024 times each).

## What is NOT established, and it is the important half

**The cure cannot be attributed yet.** Anchor and control differ in FIVE instructions —
`[26] [27] [28] [30] [33]`, the whole `t0` substitution — measured from the disassembly, not
assumed. A difference between them is caused by the substitution as a whole. It may be the
store-register match at `[33]`, the correlate this folder has retracted twice; it may equally be
any of the other four. **Do not attribute it to `[33]`.** Arms differing in more than one respect
measure whichever difference was not intended.

The arm that can attribute is built and gated: base with `[28] sw a4,0x0(a5)` → `movc t0, zero`
and `[33] stc a4` → `stc t0`, leaving `[26] [27] [30] [32]` on `a4`. Two instructions instead of
five, and the only property that moves is which register supplies the null the store writes. `[28]`
is available as the scratch slot because the functional gate showed removing its zero-init is
behaviour-preserving — its slot `s0-0x10c` is re-stored at `[755]`. sha `a6e853e25888958c`,
`SLT-SUMMARY` identical to the base.

## Method notes worth keeping

**Every arm today was the first hash-attested measurement of its kind.** Of 134 logs carrying a
latched trap `mepc`, 31 latch the canonical site and exactly ONE carried a per-run `.dom` sha256.
The base's own wedge behaviour as the dd2_join arm was attested on two draws, both from 2026-08-31.
The familiar "~54%" came from runs attributed by name and geometry, and geometry cannot distinguish
the base from a NOP-patch of it.

**The QEMU functional gate is what made the ladder readable.** Of the four cuts the static gates
left admissible, two (`{33}` and `{32,33}`) break the program and halt on a capability fault in a
different function — the same one the EJF cut faulted in on silicon. Those were the plan's priority
board candidates; boarding them would have bought unreadable verdicts at three draws each.

**A boot whose control slot fails carries no verdict**, and a draw whose image cannot be identified
by hash is VOID rather than clean. Both are enforced by the classifier now, and both were
negative-tested, because a stalled or mis-staged draw counted as clean inflates exactly the cure
this ladder exists to detect.
