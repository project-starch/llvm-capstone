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
- **Faulting 128-bit value = `0x3bcd5c5568 : 0x102247f50`.** The high word is
  shaped like compressed capability bounds, not zero/garbage — so both halves of a
  real capability are present while the tag is dropped. (Initially read as a
  "128-bit scalar copy of an aggregate," but that hypothesis was **tested and
  rejected** — see below.)

### Hypotheses tested and rejected

- **Simple aggregate/struct copy** — REJECTED. Isolated `-O0` codegen of struct
  assignment (pointer at offset 0 *and* mid-struct offset 8), by-value return, and
  `__builtin_memcpy` of a pointer-containing struct all lower to **tag-preserving
  `ldc`/`stc`** 16-byte pairs. Plain aggregate copies do not lose the tag.
- **Pointer↔integer round-trip in SQLite source** — REJECTED. The tag map has a
  live entry (a capability *was* `stc`'d) at the slots that later read untagged,
  so the pointer **was tagged** at some point; it is not an inttoptr-of-integer.

### Candidate mechanism (gap 6) — loss is UPSTREAM of DeleteTable

A value-filtered detector — "a scalar store/load touching a 16-byte granule that
currently holds a capability whose low word is `0x102247f50`" — fired **12
`TAG-ST-CLR` events** at scalar-store offsets `+0xc / +0xe`. That first suggested
"a live `pTable` copy shares a 16-byte granule with a scalar." **Static
disassembly + symbolization (`-g` build, 2026-07-02) refuted the DeleteTable-frame
version of that story and relocated the loss upstream.**

**Symbolization of the three sampled pcs (image base was `0x101ff6000`):**

| runtime pc | image off | symbol / source |
|---|---|---|
| `0x1020adfb0` | `0xb7fb0` | `sqlite3-capstone.c:158462` — trigger creation (`sqlite3AuthCheck` call) |
| `0x1020d6ccc` | `0xe0ccc` | `sqlite3-capstone.c:154848` — WITH-clause walker (`pParse->pWith = …`) |
| `0x1021077e0` | `0x1117e0` | `sqlite3ExprCollSeqMatch` prologue (`:113174`) |

Three **unrelated** functions, three different stack addresses — i.e. the
value-only filter caught **incidental** granule-aliasing on a pervasive value, not
the causal clear (exactly the caveat that was flagged).

**Disassembly of `sqlite3DeleteTable` (`0x10754`) proves its own slot is clean:**

```
10770: cincoffsetimm a0, s0, -0x40   ; pTable slot = [s0-0x40]
10774: stc a1, 0x0(a0)              ; store arg pTable — tag-preserving
10778: ld  a0, 0x0(a0)              ; scalar LOAD of low word (if(!pTable)) — loads don't clear
1079c: cincoffsetimm a0, s0, -0x40
107a0: ldc a1, 0x0(a0)             ; reload pTable
107a4: lw  a0, 0x54(a1)            ; FAULT (--pTable->nTabRef); a1 untagged
```

`stc` → scalar-*load* → `ldc`: the frame slot never gets a tag-clearing *store*.
So **`pTable` arrives in `a1` already untagged** — the tag was lost **before** the
call. `sqlite3UnlinkAndDeleteTable` retrieves the `Table*` from the schema hash
(`sqlite3HashInsert(&pDb->pSchema->tblHash, zTabName, 0)`), so the pointer is
parked in a **`HashElem` in `sqlite_heap`** between CREATE and DROP. The candidate
mechanism (a scalar sharing the stored capability's 16-byte granule) is still
viable, but it must be located in that **HashElem storage**, not any stack frame.

### Next step

Run a **storage-slot-keyed** trace (not value-keyed): follow the specific object
`0x102247f50` from `memsys5MallocUnsafe` to the `HashElem.data` slot it is stored
in, then report the pc of any tag-clearing *store* to **that HashElem granule**
before the DROP retrieval. That pins the causal site (vs. the incidental
value-aliasing hits above). Then symbolize it and inspect the MIR. Only after the
causal store is confirmed does the fix (substantial backend work) get a proposal
doc. Keep the general finding — 16-byte tag granularity makes storage *layout* a
provenance-correctness property — in `design/research-decisions-log.md`.

## Separate, do not conflate

The RV8 `aes` `-O1` run in the (contaminated) stack-shrink matrix hit the **same
assertion**, but SQLite reproduced it at **`-O0`**, so it was not an
optimization-only artifact. The other contaminated `-O1` failures
(`Cannot select: i128 = xor/or`) are a **distinct** i128-logic-op ISel gap and
should be tracked separately.
