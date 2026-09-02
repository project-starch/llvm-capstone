# RETRACTED THE SAME EVENING — the title of this note is WRONG

**"The stale-VALUE account is refuted" is WITHDRAWN.** The sentinel arm cannot refute it, and this
file's own tooling said so before the boot. `probes/s12-sentinel.py` pre-registered:

> `no fault` — weakest outcome. The perturbation cured it; uninformative alone, and must NOT be
> read as support for anything.

That is the branch that occurred. The rescue argument below — that SENT and CTRL differ by exactly
one instruction — does not work: **a matched pair licenses attributing a DIFFERENCE, and there is
no difference.** Both are 0/4.

Worse, the two halves of this note contradict each other. Section (B) accepts that the `t0`
substitution suppresses the wedge. The sentinel **contains that substitution**. If it closes the
window, the value account — which predicts a fault only when the window opens — predicts exactly
0/4 for the sentinel. The arm inherited a cure and never created the condition it was built to
label: a directed test coming back clean without ever creating the triggering condition, which is a
named failure shape in CLAUDE.md.

The only contrast bearing on the value account against a wedging baseline is SENT vs ANCHOR, and
that is a **six**-instruction difference — the exact confound this note correctly refuses to accept
for ANCHOR vs CTRL, applied to itself.

Evidence points the other way, in fact. `anchor-2.log`, from the WEDGING arm, with the mepc guard
satisfied at the canonical site:

    [wedge] a4(x14)=0x82be4cd0  a0(x10)=0x82b9f410
            => a4 holds a NON-ZERO cursor: the load DID write it, so the consumer read something
               else -- the STALE-OPERAND ACCOUNT IS CONFIRMED

**Correct status: the value account is OPEN.** What today's board work established is only what
section (B) says.

**Also corrected here:** `anchor-1` was NOT "a real 8-minute board session whose log was destroyed".
The whole 2905-byte file is a preflight block and a `flock` refusal — `another board session holds
/tmp/capstone/.board.lock`. The draw that wrote it never reached the board. The exclusion is
legitimate and cannot select for wedges, but "a lost draw" and "not a draw at all" are different
claims and only one is true.

**And the simulation weight below is overstated.** `b-NOFORWARD` has never been shown able to
increment: `b-occupancy=1024` proves the WINDOW existed, not that the comparator at
`issue_read_operands.sv:1296-1304` can fire. Occupancy is a positive control for the window, not
for the detector. Those zeros are sim-structural evidence, not board evidence, on an FPGA-only
fault.

**The experiment that would settle it:** the base with ONLY `[32] movc a4,zero` -> `li a4,0x5a5`,
leaving `[33] stc a4, 0x0(a5)` and all of `[26] [27] [28] [30]` untouched — one instruction from an
arm now attested at 5/5 wedges (3 today plus s12t-1/s12t-2 on 2026-08-31). It must be QEMU-gated
first: `[33]` would then store an untagged `0x5a5` where the program stored a null capability, which
is plausibly the same downstream break the `{33}` and `{32,33}` cuts already hit. If it cannot be
gated, the honest record stays "value account OPEN".

---

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

---

## UPDATE, same evening — the cure narrows to TWO instructions

| arm | changes from base | `[33]` stores | wedges |
|---|---|---|---|
| **anchor** | none | `a4` | **3 / 3** |
| **tight** | `[28]`, `[33]` | `t0` | 0 / 4 |
| **control** | `[26] [27] [28] [30] [33]` | `t0` | 0 / 4 |
| **sentinel** | the above plus `[32]` | `t0` | 0 / 4 |

Every wedge falls in the anchor's three draws and none in the twelve modified draws: one-sided
exact p = 2.2e-03, or 1.6e-04 counting the two hash-attested base wedges of 2026-08-31, which ran
the same binary under the sentinel's own filename *and* initramfs size.

**`[26]`, `[27]`, `[30]` and `[32]` are therefore NOT required for the fault.** All four are present
in the tight arm, which does not wedge. The cure lives in `[28] sw a4, 0x0(a5)` and/or
`[33] stc a4, 0x0(a5)`.

Every draw is validated identically: domain entered, per-run `.dom` sha256 matched, known-good
control slot returned, `SLT-SUMMARY` byte-identical to the base. Draws still running, draws that
never reached the board, and draws whose control failed are reported as three distinct non-results
and excluded from both numerator and denominator.

`--only28` is on the board to finish it: `[28]` changed, `[33]` left storing `a4`, differing from
the tight arm in `[33]` **alone**.

---

## FINAL, same evening — it is `[33]`, and specifically its SOURCE REGISTER

| arm | `[28] sw a4` | `[33] stc` source | wedges |
|---|---|---|---|
| **anchor** (unmodified) | present | **`a4`** | **3 / 3** |
| **`[28]`-only** | removed | **`a4`** | **3 / 4** |
| **tight** | removed | `t0` | 0 / 4 |
| **control** | removed | `t0` | 0 / 4 |
| **sentinel** | removed | `t0` | 0 / 4 |

**`[28]` is NOT required.** Anchor and `[28]`-only differ in `[28]` alone and both wedge, 3/3 and
3/4.

**`[33]` IS required.** `[28]`-only and tight differ in `[33]` **alone** — `stc a4, 0x0(a5)` against
`stc t0, 0x0(a5)`, the same null value into the same slot from a different register — and one
wedges 3/4 while the other is clean 0/4.

    direct, differing in [33] alone     3/4 vs 0/4     p = 0.071
    pooled by [33]'s source register    6/7 vs 0/12    p = 2.6e-04
    anchor against all t0-store arms    3/3 vs 0/12    p = 2.2e-03

The direct pair is suggestive rather than conclusive by itself. The pooling is what carries it, and
it is justified BY THE LADDER rather than assumed: the tight arm shows `[26] [27] [30] [32]` are not
required, the anchor/`[28]`-only pair shows `[28]` is not required, so the only property still
varying across the pooled arms is `[33]`'s source register.

**What this localises S-12 to.** Not a value, not an address, not the null store's presence: the
faulting configuration needs the capability STORE at `[33]` to read `a4` — the same architectural
register the LOAD at `[34]` immediately writes. Substituting an equally-null `t0` for that operand,
leaving the stored value, the store address and every other instruction identical, removes the
fault. This is the register-match correlate that has been retracted twice in this folder, now
established by substitution inside a wedging baseline rather than by correlation across
differently-built arms — which is exactly why the earlier attempts failed.

**What it does NOT establish.** Why the pairing matters. `stc-then-ldc-same-reg.S` reproduces the
shape in bare-metal simulation — `stc a4` then `ldc a4` then the consumer, 1024 times — and shows
nothing: `b-consumers 1024`, `b-NOFORWARD 0`, `HAZARDS 0`. That gap is the next thing to attack,
and the first move is a positive control for `b-NOFORWARD`, which **has never been shown able to
increment**. Until it can, those zeros bound nothing.

The value account also remains OPEN, for the structural reason recorded above: `[33]` stores `a4`
and `[34]` overwrites it with no slot between, so a4's value entering the load cannot be varied
without changing what is stored.
