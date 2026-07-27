# `core_init_matrix` codegen audit: three framings refuted board-free, and a warning for probe #67

**Date:** 2026-07-27
**Lane:** C (primary)
**Cost:** zero board boots. Static codegen comparison across all rungs + two QEMU legs.
**Follows:** `27-07-2026_00-58-47_RESULTS-65-falsified-66-localizes-hang-to-core_init_matrix.md`

The assigned question was: *does our codegen keep a live capability across the seeding loop,
where the RTL's shrink-off / store-hazard workaround assumes otherwise?* The answer is **no** —
and pursuing it refuted three separate framings without spending a board boot. Negative results
are the reliable direction here; this note records them so nobody re-derives them.

---

## Method

Emitted the compiler's own assembly (`-S`) for every rung at the opt level it is built at,
in the exact silicon config (`-capstone-gp-captable`, shrink off, `-fno-jump-tables`, `+m`).
Using `-S` rather than `llvm-objdump` sidesteps the M-extension disassembler trap entirely —
the compiler prints real mnemonics for `mul`/`rem`.

Then taint-tracked, per function, which registers hold a capability derived from `ldc rd, N(gp)`
(the cap-table = the block/global capability), and classified every access through one.

---

## What `core_init_matrix` actually does at −O0

The premise of the question does not hold. At −O0 the function keeps **no** live capability
across the loop; it does the opposite. Per element of the 9×9 seeding loop it emits:

```
ldc  a6, 0(a3)      # a3 = s0-144  -> RELOAD B from its stack slot
cincoffset a6, a6, a7                # fresh dynamic derivation
sh   a2, 0(a6)                       # narrow store
...
ldc  a2, 0(a2)      # a2 = s0-128  -> RELOAD A from its stack slot
cincoffset a2, a2, a3
sh   a0, 0(a2)                       # narrow store
```

Two `ldc` reloads + two dynamic `cincoffset` + two `sh` per element, 81 elements. Both `A`
(`s0−128`) and `B` (`s0−144`) are stack-resident capabilities re-materialised every iteration.
All capability stack slots are 16-aligned; nothing is misaligned.

**Block layout matters for probe #67 (see below):**
`BB4_3/4` = the dimension `while` loop → `BB4_5` = **the `delin` + `B = A + N*N` derivation** →
`BB4_6..12` = the seeding loops.

---

## Refuted #1 — "an extra capability load/store in a loop is the trigger"

`rv8_primes` is a decisive control. Its innermost sieve loop is:

```
ldc  a0, 0(gp)          # reload the block cap from the cap-table, every iteration
cincoffset a1, a0, a1   # dynamic index, rd != rs1
ld   a0, 0(a1)
or   a0, a0, a2
sd   a0, 0(a1)          # store through the derived cap, every iteration
```

That is a capability reload **and** a store through a dynamically-derived capability in the
hottest loop of the benchmark — and `rv8_primes` is **silicon-correct** (17,283,292 cyc, 1.050×).

⇒ The 2026-07-26 framing "an extra capability store in a loop is the confirmed miscompute
trigger" is **too strong as stated**. Whatever the trigger is, it is not that on its own.

## Refuted #2 — "the block cap gets round-tripped through memory"

Attractive, because the 23-07 silicon fix was explicitly characterised as working with
"**NO gp memory round-trip**". But `beebs_prime` — a **passing** rung — spills its block
capability to memory and reloads it (2 `stc` spills / 2 `ldc` reloads).

⇒ Memory round-tripping of a block capability is **not** a rung-level discriminator.

## Refuted #3 — "a redundant `delin` (NONLIN→NONLIN) faults on the RTL"

This one looked strong enough to be worth a real experiment:

- `coremark_matrix` is the **only** rung in the entire corpus that executes a `delin` in
  *domain code* (`CAPSTONE_DELIN(A)` in the kernel header). Every other rung: zero.
- That `delin` sits **inside `core_init_matrix`** — exactly where #66 localised the hang.
- Our QEMU fork was **explicitly patched** (`capstone-qemu` `f4d416c265`, 2026-06-04) to make
  `helper_csdelin` *idempotent* for already-NONLIN capabilities "rather than faulting" — i.e.
  our own model used to fault on this, and we taught it not to. **The RTL has no such patch.**
  That is the documented QEMU-permissive / RTL-enforces signature (cf. `C_GEN_CAP`, the
  scc-derived load wedge).
- And the cap-table glue `delin`s every entry (`split(t2,…)` → `delin(t2)` → `stc(t2, gp, i*16)`),
  which *suggested* the entry is already NONLIN when the kernel delins it again.

**Test (board-free):** instrumented `helper_csdelin`'s NONLIN early-return to print every
redundant delin, rebuilt QEMU, ran the `coremark_matrix` ladder leg.

**Result: ZERO redundant delins in the whole run** (and the run passed, oracle 14343). The
capability is genuinely **LIN** at the `delin(A)` site, so that `delin` is *necessary*, not
spurious, and the kernel header's comment ("on silicon the block cap is LINEAR") is **correct**.

⇒ Refuted. The instrumentation was reverted afterwards.

---

## What survives: one candidate, for `coremark_matrix` only

The single feature present in `coremark_matrix` and absent from **all three passing rungs**:

| rung | narrow (`sh`/`sb`/`lh`/`lb`) access through the block cap |
|---|---|
| `beebs_prime`, `rv8_primes`, `beebs_recursion` (PASS) | **0** — word-or-wider only |
| `matmult_int` (HANG) | **0** |
| `coremark_matrix` −Os (HANG) | **4 stores + 9 loads** |

Every passing rung touches its block capability only with `ld`/`sd`/`lw`/`sw`; `coremark_matrix`
is alone in doing sub-word (`sh`) stores through it — which is precisely `B[i*N+j] = val` and
`A[i*N+j] = val` in the seeding loop, `MATDAT` being `ee_s16`.

**State this as a candidate, not a mechanism.** It is a static correlation, which is exactly the
shape of the two hypotheses that died on 26-07 and 27-07. It also **cannot** be the shared cause,
because `matmult_int` has zero narrow accesses — consistent with the standing warning that the
two hanging rungs may be two different bugs.

---

## ⚠ Actionable for probe #67 — it does not split what it says it splits

#67 is specified as "return `N` *before* the seeding loop", pre-registered as a clean 2-way
separation of (a) the dimension `while` loop from (b) the N×N seeding loop.

**It is a 3-way, because the `delin` + `B = A + N*N` derivation (`BB4_5`) sits between them.**
On the HANG branch #67 would leave three candidates, not two:

| #67 outcome | remaining candidates |
|---|---|
| HANGS | the `while` loop **OR** the `delin`/`B`-derivation block |
| RETURNS `N`=9 | the seeding loop |

Refuted #3 makes the `delin` itself unlikely, but the same block also contains the
`cincoffset a0, a0, a2` that derives `B` — an `rd != rs1` derivation off the block cap, which is
the operation the kernel comment says would *consume* a linear capability.

**Suggestion for B (cheap, same boot count):** put the split point *before* the `delin`, or make
#67 3-way. Either gives a clean answer; as specified it may burn a boot and still leave two.

---

## Housekeeping

A `git checkout` intended to drop the temporary diagnostic reverted the whole file and discarded
the uncommitted local `CAPSTONE_GP_FABRICATE` / `CAPSTONE_GP_STANDIN` patch in
`capstone-qemu` `op_helper.c` (the four gp-fabrication guards + the `cscall` monitor stand-in).
It was reconstructed and **validated end-to-end**, not just eyeballed:

- `gp-free-domain/build-and-run.sh` with `CAPSTONE_GP_STANDIN=1` →
  `retval = 554745961`, `__CAPSTONE_GPFREE_DOMAIN_PASSED__`, `static: cjalr=0 cincoffset-gp=0 scc-gp=3`
  — the exact documented expected result, exercising the reconstructed stand-in path.
- Legacy default (fabrication ON) → `coremark_matrix` ladder leg passes, oracle 14343.
- Diff size back to **39 insertions**, matching the original.

**Lesson worth keeping: never `git checkout --` a submodule source file.** Submodule source in
this project is deliberately uncommitted local experiment, so `checkout` is a destructive
operation with no undo. Revert a temporary edit with a targeted `Edit`, or stash first.
