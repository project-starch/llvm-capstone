# SQLite gap 5 — `cscincoffset` untagged-base fault (backend operand order)

*Status: ROOT-CAUSED 2026-07-01. Not yet fixed. Surfaced after the untagged
`ldc`/`stc` QEMU fix + tag-preserving `memcpy` cleared gaps 3–4. This is a
compiler (backend ISel) bug, not a memory/tag-loss bug — the `memcpy` is
exonerated by the `tagged_cap_memcpy_aligned` authority probe.*

## Symptom

SQLite (`run-sqlite-memory.sh`, domain built at **`-O0`**) aborts QEMU with:

```
target/riscv/op_helper.c: helper_cscincoffset: Assertion `rs1_v->tag' failed.
```

`cscincoffset rd, rs1, rs2` computes `rd = rs1 + offset(rs2)` and requires the
**base** `rs1` to be a tagged capability.

## Evidence (temporary instrumentation of `helper_cscincoffset`, since reverted)

```
[CSCINCOFFSET-DEBUG] untagged base: pc=10201be90 rd=10 rs1=10 rs2=11 \
    rs1.scalar=60 rs1.scalar_hi=0 rs2.tag=1 rs2.val=102152f50
```

So the emitted instruction is `cscincoffset x10, x10, x11` where:
- `rs1` (x10, the base) is **untagged**, value `0x60 = 96` — a genuine small
  integer, **not** a de-tagged pointer (a stripped heap pointer would be a large
  address like the `0x102152f50` we see in rs2).
- `rs2` (x11, the offset) is the **real tagged capability** (`0x102152f50`).

The capability and the integer are in the **wrong roles**: this is `96 + <cap>`
(`int + ptr`, commutative) lowered with the integer as the capability base.

## Root cause

`llvm/lib/Target/Capstone/CapstoneISelDAGToDAG.cpp`:

- `ISD::ADD` i128 case (~:1787): the operand swap that puts a constant in the
  offset position only fires for a compile-time `ConstantSDNode`:
  ```cpp
  if (isa<ConstantSDNode>(Base) && !isa<ConstantSDNode>(Offset))
    std::swap(Base, Offset);
  ```
  A **runtime** integer (an i64 widened to the i128 carrier via
  ZERO/SIGN/ANY_EXTEND) is not a `ConstantSDNode`, so it is not swapped.
- `selectCIncOffset` (:1217) then emits `CIncOffset`/`cscincoffset` with
  `operand(0)` as the base. When `operand(0)` is the integer carrier and
  `operand(1)` is the capability (i.e. source `int + ptr`), the integer becomes
  the base `rs1` → untagged base → the assertion.

The DAG cannot currently tell which i128 operand carries a capability vs an
integer, so it trusts operand order. For `ptr + int` order it happens to be
correct; for `int + ptr` it is wrong.

## Fix direction (next step, not yet done)

In the i128 `ISD::ADD` path (and/or `selectCIncOffset`), pick the **capability**
operand as the base regardless of operand order. A workable heuristic: if `Base`
is an integer-carrier i128 (an ANY/ZERO/SIGN_EXTEND of a scalar ≤ XLEN, i.e. the
same shape `matchExtendedXLenOffset` already recognises for the *offset*) and
`Offset` is not, `std::swap(Base, Offset)` before selection. This generalises the
existing constant-only swap to runtime integer carriers.

- Add a lit test: `add i128 (zext i64 %n to i128), %cap` must select
  `cscincoffset cap, n` (capability as base), for both operand orders.
- Add an authority probe: an `int + ptr` deref that faults today and works after
  the fix (analogue of the tagged-cap probes).
- Re-run `run-sqlite-memory.sh` to confirm gap 5 clears; then the next SQLite
  fault (if any) becomes gap 6.

## Separate, do not conflate

The RV8 `aes` `-O1` run in the (contaminated) stack-shrink matrix hit the **same
assertion**, but SQLite reproduces it at **`-O0`**, so this is not an
optimization-only artifact. The other contaminated `-O1` failures
(`Cannot select: i128 = xor/or`) are a **distinct** i128-logic-op ISel gap and
should be tracked separately.
