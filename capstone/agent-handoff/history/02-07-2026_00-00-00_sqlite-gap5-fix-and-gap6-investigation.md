# SQLite gap 5 — `cscincoffset` untagged-base fault (backend operand order)

*Status: FIXED 2026-07-01 (commit pending). Root-caused then fixed in the same
session. Surfaced after the untagged `ldc`/`stc` QEMU fix + tag-preserving
`memcpy` cleared gaps 3–4. This was a compiler (backend ISel) bug, not a
memory/tag-loss bug — the `memcpy` is exonerated by the `tagged_cap_memcpy_aligned`
authority probe. The fix cleared the `cscincoffset` assertion; SQLite now runs
past it and hits a distinct, deeper untagged-pointer gap (**gap 6**, see below).*

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

## Fix (implemented)

The real chokepoint is `selectCIncOffset` (CapstoneISelDAGToDAG.cpp): **both** the
raw `ISD::ADD` i128 ISel case (`Select`, ~:1815) and the `CapstoneISD::CIncOffset`
custom-node case (~:3685) funnel through it, and it reads `Node->getOperand(0/1)`
directly — so the pre-existing constant-only swap in the `ISD::ADD` case never
reached it. `ISD::ADD` i128 is `Custom`-lowered (`CapstoneTargetLowering::LowerADD`
already canonicalizes operand order using `isCapstoneIntegerOffset` /
`isCapstoneCapabilityValue`), but adds that reach ISel **after** legalization
(post-legalize combines) bypass that swap and arrive raw.

Fix: at the top of `selectCIncOffset`, apply the *same* predicate-based swap
`LowerADD` uses, so the tagged capability is always the base:

```cpp
if (((isCapstoneIntegerOffset(Base) && !isCapstoneIntegerOffset(Offset)) ||
     (isCapstoneCapabilityValue(Offset) && !isCapstoneCapabilityValue(Base))) &&
    !isa<FrameIndexSDNode>(Offset))
  std::swap(Base, Offset);
```

- The two predicates were made non-`static` and declared in `CapstoneISelLowering.h`
  so both canonicalization sites share one classifier (no drift). The FrameIndex
  guard preserves the dedicated frame-index base-materialization path above.
- The swap is conservative and idempotent: it fires only on definite
  integer-carrier-base / capability-offset shapes, so a correctly-ordered
  `ptr + int` is provably never swapped, and a node `LowerADD` already
  canonicalized is left unchanged.

### Validation

- Capstone lit suite **34/34** (added `cap-cincoffset-base.ll`, which locks in
  that a capability arriving as a raw i128 load is classified as the base).
- `run-sqlite-memory.sh`: the `helper_cscincoffset: rs1_v->tag` assertion is
  **gone**. SQLite executes past gap 5.

## Gap 6 (surfaced by the fix) — untagged `Table*` into `sqlite3DeleteTable`

With gap 5 cleared, `run-sqlite-memory.sh` now faults with a different mechanism:

```
[CAPSTONE] Cap mem access requires capability: pc = 10200079c, rs1 = x11, imm = 84
```

Domain base this run = `0x101ff6000`, so the fault is at domain vaddr `0x1079c`
in `sqlite3DeleteTable`:

```
10774: stc a1, 0x0(a0)   ; arg1 (pTable) spilled to [s0-0x40] with stc (tag-preserving)
...
107a0: ldc a1, 0x0(a0)   ; reloaded with ldc
107a4: lw  a0, 0x54(a1)  ; FAULT — a1 (pTable) is untagged
```

`pTable` is stored/reloaded with the tag-preserving `stc`/`ldc` pair (proven by
the `spill_reachability` authority probe), so it was **already untagged when the
caller passed it**. This is a distinct, deeper provenance gap upstream, **not**
the cscincoffset operand-order bug and **not** a spill defect.

### Investigation (2026-07-01, QEMU instrumentation, since reverted)

Temporary QEMU detectors (printing the untagged value + a frame anchor, a
"scalar store of a tagged capability register" detector, and a value-filtered
"scalar load from a capability slot" detector) established:

- **The untagged pointer is `0x102247f50`.** By symbol table that vaddr lies
  inside **`sqlite_heap`** (`nm`: `sqlite_heap` `0x166f50` .. `__capstone_gct_start`
  `0x266f50`) — SQLite's static **MEMSYS5** arena (the domain configures it via
  `sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, …, 64)`, build flag
  `-DSQLITE_ENABLE_MEMSYS5=1`). So `pTable` is a MEMSYS5-allocated object.
- **The allocator returns a *tagged* pointer.** `memsys5MallocUnsafe`'s return
  path is `ldc a0, 0x10(a2)` (zPool, tagged) then `cincoffset a0, a0, <off>` —
  a tagged capability. So the allocation is fine; the tag is lost **downstream**.
- **Not a scalar store.** Zero domain hits for "scalar store of a tagged
  capability register" before the fault — the tagged pointer is never `sd`'d.
- **Plain varargs is *not* the culprit.** A minimal `va_arg(ap, void*)` compiles
  to tag-preserving `stc`/`ldc` at `-O0`, so passing `sqlite_heap` through
  `sqlite3_config`'s `...` keeps the tag.
- **Signature = 128-bit scalar copy of a capability.** The faulting register's
  full 128-bit value is `0x3bcd5c5568 : 0x102247f50`; the **high word looks like
  compressed capability bounds**, not zero/garbage. So *both* 64-bit halves of a
  real capability were preserved while the tag was dropped — the fingerprint of a
  capability-containing aggregate copied with scalar 64-bit ops (`ld`/`sd` pairs
  or an inlined struct copy), **not** a single scalar pointer load.

### Reframed root cause + next step (gap 6)

Gap 6 is a **capability-containing aggregate/struct copied via scalar 64-bit
memory ops**, dropping the tag — the same class as gaps 2–3 but a case the
16-byte-aligned tag-preserving `memcpy` does **not** cover (an inlined struct
copy, a `memcpy` the compiler expanded to `2×i64` load/store, or a pointer field
at a non-16-aligned struct offset). Next step: map the copy site to a source
line (via `-g`/`llvm-symbolizer` on the domain, or a value-filtered lo+hi load
detector) and fix it — either make the backend lower capability-containing
aggregate copies with `ldc`/`stc`, or ensure such copies route through the
tag-preserving `memcpy`. A minimal authority probe (a struct with a pointer
field copied by assignment, then dereferenced) should reproduce it.

## Separate, do not conflate

The RV8 `aes` `-O1` run in the (contaminated) stack-shrink matrix hit the **same
assertion**, but SQLite reproduced it at **`-O0`**, so it was not an
optimization-only artifact. The other contaminated `-O1` failures
(`Cannot select: i128 = xor/or`) are a **distinct** i128-logic-op ISel gap and
should be tracked separately.
