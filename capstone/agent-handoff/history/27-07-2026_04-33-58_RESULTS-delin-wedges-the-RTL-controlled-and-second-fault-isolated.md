# Board results #67a–#67f: `delin` in domain code WEDGES the RTL (size-matched control), and a second independent fault is isolated to the seeding loop

**Date:** 2026-07-27
**Lane:** C (primary)
**Cost:** one board session, six boots, one rung each. Board powered off + unlocked.
**Supersedes:** the "next probe" half of
`27-07-2026_00-58-47_RESULTS-65-falsified-66-localizes-hang-to-core_init_matrix.md`
**Related:** `27-07-2026_02-45-07_core_init_matrix-codegen-audit-three-framings-refuted.md`

---

## Headline

**A `delin` executed in *domain code* on a capability loaded from the gp cap-table wedges the
CVA6 RTL.** Proven by a size-matched, position-matched, register-plumbing-matched control: the
same image with `addi x0,x0,0` in the `delin`'s place **returns correctly**. This is a genuine
silicon divergence — every one of these builds is QEMU-correct through the identical board
controller — and it is **not** a compiler bug.

Removing the `delin` is **safe and correct** (the derivation it was protecting works without it,
and QEMU still computes the right answer), but it does **not** make `coremark_matrix` run:
a **second, independent fault** remains, now isolated to the seeding loop or later.

---

## Results

Every build below was QEMU-validated through `run-ladder-perf-qemu.sh` — the *same* controller
the board uses — before it was allowed on hardware. All are `-O0` at a 32 KiB code window,
i.e. the exact configuration of #66 and of the mode-7 probe.

| probe | delta from the row above | board |
|---|---|---|
| mode 7 (#63) | entry path only, compute branched over | RETURNS |
| **#67a** | + the dimension `while` loop | **RETURNS 9** ✓ (1678 cyc, 295 instr) |
| **#67c** | + **`delin`** — one instruction | **HANGS** |
| **#67e** | #67c with `addi x0,x0,0` **instead of** `delin` (size-matched control) | **RETURNS 9** ✓ (1812 cyc, 302 instr) |
| **#67f** | `B = A + N*N` derivation, **`delin` removed** | **RETURNS 9** ✓ (1854 cyc, 307 instr) |
| **#67b** | `delin` + derivation | **HANGS** |
| **#67d** | the **full** benchmark, `delin` removed | **HANGS** |
| #66 | full `core_init_matrix` (with `delin`) | HANGS |

---

## 1. The `delin` is the fault, and layout is controlled out

`#67c` differs from `#67a` by **exactly one instruction**. That alone would have been unsafe to
act on: the 2026-07-26 controlled A/B showed that adding **four** instructions to `domain_main`
flipped a passing rung from correct to wrong, so this machine is documented to be sensitive to
code-layout perturbation at that scale.

So `#67e` holds everything constant and changes only the instruction *encoding*:

```c
/* #67c   */ __asm__ volatile(".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(A));  /* delin  */
/* #67e   */ __asm__ volatile(".insn i 0x13, 0x0, x0,  0(x0)"      : "+r"(A));  /* addi 0 */
```

Both are 4 bytes, at the same position, with identical `"+r"(A)` register plumbing. Verified by
diffing the emitted `core_init_matrix`: the two functions differ in exactly one line.

**#67c HANGS, #67e RETURNS.** Layout, size, register allocation and surrounding code are
eliminated. The `delin` opcode itself is what wedges the board.

## 2. It is *not* "the RTL doesn't implement `delin`"

Every domain executes `delin` several times **in the glue** — `BUILD_GP_CAPTABLE` does
`split(gp,…)` → `delin(gp)`, then per global `split(t2,…)` → `delin(t2)` → `stc(t2, gp, i*16)`,
and finally `delin(sp)` — and passing rungs like `beebs_prime` and `rv8_primes` reach their
results. So `delin` executes fine there.

The difference is **what it is applied to**:

| site | operand provenance | outcome |
|---|---|---|
| glue | capability **fresh from `split`** (register-resident, never stored) | works |
| domain code | capability **loaded by `ldc` from the cap-table** (a memory round-trip) | **wedges** |

The glue `delin`s each cap-table entry *before* `stc`-ing it, so on a machine that preserves the
capability **type** through a memory round-trip, the entry the domain later loads is **already
NONLIN** — and the domain's `delin` is then a NONLIN→NONLIN operation.

**This is exactly the case our QEMU fork was explicitly patched to tolerate.** `capstone-qemu`
`f4d416c265` (2026-06-04) added an early return to `helper_csdelin` for already-NONLIN
capabilities, with the comment *"treat delin as idempotent rather than faulting"* — i.e. before
that patch **our own model faulted on it**, and we taught it not to. The RTL has no such patch.

That is the same QEMU-permissive / RTL-enforces shape as the two prior silicon findings:
`C_GEN_CAP` (a QEMU-only op the RTL decodes as `default: ;`), and the `scc`-derived-load wedge
of 2026-07-23.

**Caveat, stated honestly:** instrumented QEMU reports the domain-side `delin` operand as **LIN**,
not NONLIN (zero redundant delins in a full run — see the 02-45-07 note, where this was used to
*refute* the redundancy theory). So QEMU and the glue disagree about the type of a capability
after `stc`→`ldc`. The refutation in that note was made **using the very model now under
suspicion**, and should be read as "QEMU says LIN", not as "the capability is LIN". Which of the
two readings is right — type not preserved through memory in QEMU, or `delin` diverging on the
RTL for some other reason — is the open question, and it is a question **for the board owner**,
not something we can settle from this side.

## 3. Removing the `delin` is safe — but does not fix the rung

- `#67f` (derivation, no `delin`) **RETURNS 9**: the `cincoffset rd, rs1, rs2` with `rd != rs1`
  that the `delin` exists to protect works fine without it. The kernel comment's fear that this
  would *consume* a linear `A` does not materialise on hardware.
- QEMU with the `delin` removed still computes the correct full-benchmark answer **14343**.
- But `#67d` — the **full** benchmark with the `delin` removed — still **HANGS**.

⇒ There are **at least two independent faults** in this rung. Fault 1 is the `delin` (settled).
Fault 2 is downstream of the derivation, i.e. in the **N×N seeding loop or later**.

**Fault 2 revives the one static candidate that survived the codegen audit:** `coremark_matrix`
is the only rung in the corpus performing **narrow (`sh`/`sb`/`lh`/`lb`) accesses through the
gp-delivered block capability** — 4 narrow stores + 9 narrow loads, versus zero in all three
passing rungs. The seeding loop is exactly where those `sh` stores live (`MATDAT` is `ee_s16`).
It is still a candidate, not a mechanism; the next cheap probe is a phase bisect *inside* the
seeding loop, or a variant with `MATDAT` widened to 32-bit.

---

## What this changes

- **A minimal, self-contained silicon repro now exists** — two 4-byte instructions, one image,
  one boot each, both QEMU-correct. This is precisely the "hand it to the board owner with a
  minimal repro" outcome the plan called *paper-acceptable*: a documented hardware limitation,
  not an unexplained divergence. Keep the message **short and human**.
- **`matmult_int` is untouched by this.** It contains **no `delin` at all**, so fault 1 cannot
  explain it. Continue to treat the two hanging rungs as possibly two different bugs.
- **The paper position is unchanged and safe:** 3 measured rungs + the
  `ref/fpga-silicon-measurements-for-paper.md` §5 caveats. `coremark_matrix` remains blocked,
  but is now blocked for a *named and controlled* reason rather than an unexplained one.

## Build knobs added

In `silicon-ladder/coremark_matrix_kernel.h` (all `#ifdef`, default off, oracle-forwarded):
`LADDER_CM_STOP_AFTER_WHILE` (#67a), `LADDER_CM_STOP_AFTER_DELIN` (#67c),
`LADDER_CM_STOP_AFTER_DERIVE` (#67b), `LADDER_CM_NO_DELIN` (#67d/#67f),
`LADDER_CM_DELIN_NOP_CONTROL` (#67e). Plus an uncommitted 32 KiB linker script
`gp-free-domain/link-gpfree-32k.ld` (the `0x1000` → `0x8000` globals offset), which #66 and
mode 7 also required and which had not been checked in.

## Board etiquette

Six boots, one rung each (one rung per boot — a warm reset cannot re-enter OpenSBI). Board
**powered off and unlocked** after every run via the driver's `finally`. No bitstream re-flash
was attempted or needed. `FPGA_URL` was passed inline per run and never persisted or logged.
