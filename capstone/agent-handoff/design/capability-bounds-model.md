# Capstone capability bounds model — precision, `SHRINK`, and naming

*Status: ground-truth for the granularity contribution (C1). Sourced from the
spec (`capstone-spec/parts/{cap-man-insn,prog-model}.adoc`), the QEMU model
(`capstone-qemu/target/riscv/{cap.h,cap_compress.c,op_helper.c}`), and the LLVM
backend (`CapstoneInstrInfo.td`, `CapstoneISelDAGToDAG.cpp`,
`IntrinsicsCapstone.td`). This resolves the "representability" question and the
`SHRINK` vs `CSetBounds` naming inconsistency in the other design docs.*

## TL;DR

- Capstone uses a **CHERI-128-style compressed-bounds capability**: bounds are
  **precise (full 64-bit base/cursor/end) while resident in a register**, and
  **compressed into 128 bits when stored to memory** (floating-point-style
  mantissa + shared exponent).
- **Representability rounding is real and necessary** — the GPT note's
  "near-ideal *modulo representability*" hedge is **correct**, not an imported
  CHERI-ism. Precise bounds for objects **< 4 KiB**; larger objects round to a
  power-of-two-aligned grain (base down, top up), so authority may round
  **outward** by up to one grain.
- The bounds-narrowing primitive is **`SHRINK`** (QEMU `csshrink`, spec §shrink).
  **`CSetBounds`/`scbnds` do not exist in Capstone** — that is CHERI naming;
  purge it from the docs in favour of `SHRINK`.
- The ISA also has **`SHRINKTO`** (narrow to `[cursor, cursor+imm)`, ideal for
  object/`malloc` materialization) and **`SPLIT`** (true capability split into
  two adjacent halves). **Neither is wired into LLVM** — only `SHRINK` is. This
  also corrects the PI-discussion claim that "capability splitting does not
  exist": the *primitive* exists (`SPLIT`); the compiler just never emits it.

## 1. Representation: precise in register, compressed in memory

`capstone-qemu/target/riscv/cap.h` — the in-CPU "fat" capability carries exact
64-bit fields and bounds checks are exact:

```c
struct CapBoundsFat { capaddr_t cursor, base, end; };   // full 64-bit each
static inline bool cap_in_bounds(capboundsfat_t* b, capaddr_t base, capaddr_t size) {
    return b->base <= base && base + size <= b->end;     // exact check
}
```

`cap_compress.c` packs a fat cap into 128 bits on store (`cursor` + a 64-bit
`other` holding `bE:3 b:11` base mantissa, `tE:3 t:9` top mantissa, `iE:1`,
`ty:3`, `perms:3`, `revnode_id:31`) and `cap_uncompress` reconstructs it on load.
This is the standard CHERI-128 floating-point bounds scheme.

> **Spec basis** (`prog-model.adoc`): implementations *may* compress fields as
> long as (1) a register↔memory round-trip preserves the value and (2) a
> compressed result is *never more powerful* than on an uncompressed machine.
> The practical consequence: a bounds-setting op must yield a **representable**
> value, and where it can't, it rounds **conservatively**. (Note: current QEMU
> keeps fat bounds exact in-register and only rounds inside `cap_compress`, so a
> SHRINK'd cap can *widen* by up to one grain after a store/reload for large
> objects — the CHERI "imprecise bounds" caveat. Track this when reasoning about
> spilled/stored narrowed caps.)

## 2. Precision threshold (from `cap_compress`)

Let `len = end - base`, `hb = ` index of the highest set bit of `len`,
`E = max(hb - 12, 0)`.

- **`len < 4096` (E = 0, `iE = 0`):** **byte-exact** bounds (14-bit base / 12-bit
  top mantissa, no rounding).
- **`len ≥ 4096` (E ≥ 1):** base/top mantissas shift right by `E`, base's low
  3 bits cleared (**base rounds down**) and top incremented if any low bits
  (**top rounds up**). Grain = `2^(E+3)` bytes; e.g. a ~1 MiB object rounds to a
  2 KiB grain.

**Paper-precise claim:** *compiler-generated object bounds are byte-exact for
objects under 4 KiB; for larger objects, base/end are aligned to a
`2^(E+3)`-byte grain (`E = ⌈log2 len⌉ − 12`), rounding authority outward by at
most one grain.* This is the honest form of "near-ideal granularity."

## 3. `SHRINK` semantics (the narrowing primitive)

Spec §shrink / `helper_csshrink` (`op_helper.c:729`): `rd = shrink(rd, base, end)`
(in LLVM: `SHRINK $rd, $rs1, $rs2` with `$rd = $cap_in`).

- **Monotone:** raises `Illegal operand value` unless `base ≥ rd.base` and
  `end ≤ rd.end` (and `base < end`). You can only narrow, never widen — exactly
  the property the granularity/provenance argument needs.
- Sets `rd.base = base`, `rd.end = end`, clamps `cursor` into the new range.
- Requires `rd` to be a **capability** of type LIN/NONLIN/UNINIT; `rs1/rs2`
  **integers**. No rounding in the helper itself (fat bounds are exact;
  representability bites only at the later store — see §1).

**LLVM wiring:** `int_capstone_cap_shrink(ptr, i128 base, i128 end)`
(`IntrinsicsCapstone.td:90`) → `selectShrink` → `Capstone::SHRINK`
(`CapstoneISelDAGToDAG.cpp:1383`). It is **only reachable via the intrinsic** —
never emitted automatically at object materialization. Closing that gap is the
C1 (granularity) work item.

## 4. Sibling primitives that exist in the ISA but not in LLVM

| ISA / QEMU | What it does | In LLVM? |
|------------|--------------|----------|
| `SHRINK` / `csshrink` | narrow to `[base, end)` (monotone) | **yes** (`int_capstone_cap_shrink`) |
| `SHRINKTO` / `csshrinkto` | narrow to `[cursor, cursor+imm)` — natural fit for object/`malloc` sizing | **no** |
| `SPLIT` / `cssplit` | split a cap into `[base, mid)` + `[mid, end)` | **no** |

For the global/heap slice, `SHRINK` is sufficient. `SHRINKTO` would be a cleaner
lowering for "narrow to `sizeof(obj)` at the cursor" and is worth wiring up if
the materialization code already knows the size as an immediate.

## 5. Consequences for the plan

- **C1 granularity claim is well-founded and quantifiable** — exact < 4 KiB,
  bounded outward rounding above. Use the §2 statement verbatim in the paper.
- **Emit `SHRINK` (not `CSetBounds`).** Optionally wire `SHRINKTO` for the
  size-immediate case.
- **Spill/store widening caveat (§1)** is a real, citable subtlety for the
  spilled-capability question (PI Q1) and for any stored narrowed cap.
- **`SPLIT` exists** — update `pi-discussion-capability-granularity-provenance.md`
  Q7: the hardware splitting primitive is present; the compiler emits none.
