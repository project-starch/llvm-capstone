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

### CONFIRMED root cause (2026-07-02, storage-slot-keyed trace)

A storage-slot-keyed trace (QEMU hooks on the tag-map add/remove + cap-load
paths, keyed by the pointer value **and** by the granules that value is stored
into; since reverted, submodule clean) settled it. All numbers below are from the
**helper arguments** (tag flag, address, size) — *not* from `env->pc`, which this
QEMU syncs lazily so it is useless for pinning the storing instruction (it
attributed a **stack**-address byte store to `pcache1FetchStage2`, a heap
page-cache routine — impossible; `ra` is equally stale).

Reliable findings:

- **Every `stc` of the `Table*` is tagged** (89 stores, all `tag=1`). The tag is
  never lost at a capability store.
- The value is read back **untagged** at exactly **two** granules, both on the
  **stack**: `0x1023ffa80` (first) and `0x1023f39e0` (DeleteTable's own
  `[s0-0x40]`, the fault).
- **Transition granule = `0x1023ffa80`:** stored **tagged** (`seq80 tag=1`), then
  its tag is cleared by **byte-wise scalar stores (`size=1`)**, then read back
  **untagged**. That is the root event.
- DeleteTable's slot `0x1023f39e0` receives the value **already untagged**
  (`stc tag=0`) and reads it untagged → fault. So the loss is upstream of
  DeleteTable and propagates in as an untagged register (consistent with the
  clean-slot disassembly above).

**Mechanism (confirmed):** a live tagged capability resident in a **16-byte stack
granule** has its tag stripped by a **byte-wise memory copy** (a
`memcpy`/`memmove`/small struct-copy/`memset` lowered to `sb`/`sh`/`sw` scalar
stores) that overwrites the granule. This is the **storage-aliasing /
16-byte-tag-granularity** class — the *sub-16-byte scalar-copy* path that bypasses
the tag-preserving `ldc`/`stc` middle (the same class as the documented
`tagged_cap_memcpy_misaligned` limitation), now shown to bite an on-stack
`Table*`. It is **not** the heap `HashElem` (that earlier hypothesis is
superseded), not the allocator, not a value-motion ABI bug.

### EXACT site pinned (2026-07-02) — the domain `memcpy` byte-copy path

The lazy `env->pc` / stale `ra` could not name the store, so a **translate-time
pc** was threaded through: `gen_helper_remove_cap_mem_map` was widened to take
`ctx->base.pc_next` (the current-instruction guest pc, exact and non-perturbing;
`cpu_restore_state()` was tried first but it *diverged the run*, so it was
discarded). With the exact pc — and the **correct load base `0x101ff0000`**
(derived by matching a byte-store pc to `memcpy`'s `sb`; the base I had used before
was off by `0x6000`, which is why earlier symbolizations were incoherent) — the
tag-clearing store is:

- `exact_pc 0x102142c44` = **`memcpy + 0x1fc`** (`sb a0, 0x0(a1)`), the byte-copy
  loop of the domain's freestanding `memcpy`; a sibling clear at `0x102143000`
  is the analogous `memmove`/`memset` byte loop.
- The 16 clears span offsets `0..f` of granule `0x1023ffa80` — a **full 16-byte
  byte-by-byte copy** of the region holding the `Table*`.
- The `memcpy` caller (from the `ra` capture, rebased) is in the
  **`sqlite3NestedParse`** region — schema/parse machinery.

Crucially, this `memcpy` **has** a tag-preserving `ldc`/`stc` fast path (22 such
ops in its body), but this copy took the **byte path** — which happens when the
source and destination are **relatively misaligned mod 16**, so the 16-byte
`ldc`/`stc` cannot be used. The byte copy of a 16-byte-aligned tagged capability
strips its tag. This is exactly the documented **`tagged_cap_memcpy_misaligned`**
limitation, now shown to break real SQLite.

### Fix direction

Primarily a **runtime-library** fix (smaller than a backend ISel change): make the
domain `memcpy`/`memmove` preserve tags even when src/dst are relatively
misaligned — e.g. detect any 16-byte-aligned capability granule inside the copied
range and move it with `ldc`/`stc` (querying/repairing the tag) rather than
byte-wise. Alternatively/additionally, guarantee that capability-bearing SQLite
structs are 16-aligned so the fast path always applies. Connects to gap 2 and the
sub-capability aggregate-copy fix. Next: write the proposal doc, then implement +
add an authority probe reproducing the relatively-misaligned cap-copy. The general
finding (16-byte tag granularity makes storage *layout* a provenance-correctness
property) is in `design/research-decisions-log.md`.

## Step 0 diagnostic (2026-07-03) — Case A confirmed; exact culprit = `sqlite3NestedParse` `saveBuf`

Ran the proposal's step-0 disambiguation (Case A vs B). Method: temporary QEMU
instrumentation (since reverted, submodule clean) recording, for every byte load
(`size==1`), the source address + whether its granule is tagged (`gap6_last_load_*`),
then a **pc-gated** log at the culprit store `sb` (`0x102142c44` = `memcpy+0x1fc`)
correlating each tag-strip with its immediately-preceding `lbu` source and printing
`memcpy`'s caller (`ra`, rebased). Findings:

- **Exactly one primary loss** across the whole run:
  `dst=0x1023f1f1c dst%16=12 (misaligned) | src=0x1023ffa10 src%16=0 src_tagged=1`.
  A genuine **tagged, 16-aligned source** capability is byte-copied to a
  **relatively-misaligned destination** (offset 12). → **Case A** (not Case B).
- All other strips (~15, `src_tagged=0`) are **secondary**: they copy the already
  untagged value back onto aligned tagged granules, clobbering their tags. Their
  callers symbolize to `sqlite3_str_append` / `tokenExpr` / `sqlite3NestedParse`
  (benign stale-tag stack reuse for strings/tokens).
- Primary caller `ra=0x1020c57c8` = **`sqlite3NestedParse+0x174`**. Disasm of the
  call site: `dst = s0-0x174` (offset 12 mod 16), `n = 0x100 = 256`, source = the
  16-aligned Parse struct tail.

Source confirms it exactly (amalgamation `sqlite3-capstone.c`):
```c
char saveBuf[PARSE_TAIL_SZ];                        /* bare char[] → NOT 16-aligned */
memcpy(saveBuf, PARSE_TAIL(pParse), PARSE_TAIL_SZ); /* aligned tail → misaligned buf: strips */
memset(PARSE_TAIL(pParse), 0, PARSE_TAIL_SZ);
...
memcpy(PARSE_TAIL(pParse), saveBuf, PARSE_TAIL_SZ); /* untagged buf → aligned tail: clobbers */
```
`PARSE_TAIL(pParse)` is 16-aligned (empirically `src%16=0`); `saveBuf` is a plain
`char[]` the compiler placed at a 12-mod-16 slot. The Parse tail holds capability
pointers (the `Table*` that later faults in `sqlite3DeleteTable`), saved into the
misaligned buffer and restored untagged.

**Consequence for the fix.** Because the two ends have a **constant** relative
misalignment (12), **no `memcpy` cleverness (Option 2) can preserve the tag** — a
capability physically cannot be tagged at destination offset 12. The only correct
fix for Case A is **Option 1: eliminate the misalignment** — 16-align `saveBuf`
(e.g. `_Alignas(16)` / `__attribute__((aligned(16)))`) via a build-time `sed`
patch, so `da == sa == 0` and `memcpy`'s existing `ldc`/`stc` fast path carries the
tags. Option 2 remains valuable only as general hardening for *mixed*-alignment
copies (sub-ranges that re-align), which this case does not exhibit. This sharpens
the paper point: **a `char[]` used to save/restore pointer-bearing memory is a
latent tag-stripping bug on a capability machine unless 16-aligned.**

Next: implement Option 1 (align `saveBuf`), add authority probe
`tagged_cap_memcpy_relmisaligned`, and validate SQLite past gap 6 + BEEBS 82/82 /
RV8 7/7 / CoreMark green.

### Gap 6 FIXED (2026-07-03) — 16-align `saveBuf` (Option 1)

Implemented Option 1. `build-sqlite-capstone.sh` now `sed`-patches
`char saveBuf[PARSE_TAIL_SZ];` → `char saveBuf[PARSE_TAIL_SZ] __attribute__((aligned(16)));`
(with a verification `grep`). With the buffer 16-aligned, both the save
(`memcpy(saveBuf, PARSE_TAIL(pParse), …)`) and restore copies are aligned→aligned,
so `memcpy`'s existing `ldc`/`stc` fast path carries the Parse-tail capability tags
and nothing is byte-copied. **Result: the `sqlite3DeleteTable` untagged-`Table*`
fault is gone** — SQLite now runs past gap 6.

- **Scope / no regression risk:** the fix touches only the SQLite amalgamation
  (`sqlite3-capstone.c`); the shared freestanding `memcpy` and QEMU are unchanged
  (submodule clean, built from committed source), so BEEBS/RV8/CoreMark compile
  byte-identical code and are unaffected. (Option 2 — a memcpy change — was *not*
  needed: a constant relative misalignment is unpreservable by any copy loop, so
  layout is the only correct fix.)
- **Authority probe added:** `tagged_cap_saverestore_aligned_buf` — saves a tagged
  capability into a 16-aligned `char[]` and restores it (the exact gap-6 shape),
  asserting the tag round-trips (oracle `ok`, retval `0x22AC0001`). It is the
  positive counterpart to the existing `tagged_cap_memcpy_misaligned` (unaligned
  destination → tag-fault). Full authority suite green.

**New blocker surfaced — "gap 7":** past gap 6, SQLite now hits a QEMU assertion
`helper_cscincoffset: Assertion 'rs1_v->tag' failed` (`op_helper.c:597`) — a
`cscincoffset` with an **untagged capability base** (`rs1 != gp`). Same helper as
gap 5 but a distinct site; a hard assert rather than a clean fault. Root-causing is
the next step (needs the guest pc of the offending `cscincoffset`).

## Separate, do not conflate

The RV8 `aes` `-O1` run in the (contaminated) stack-shrink matrix hit the **same
assertion**, but SQLite reproduced it at **`-O0`**, so it was not an
optimization-only artifact. The other contaminated `-O1` failures
(`Cannot select: i128 = xor/or`) are a **distinct** i128-logic-op ISel gap and
should be tracked separately.
