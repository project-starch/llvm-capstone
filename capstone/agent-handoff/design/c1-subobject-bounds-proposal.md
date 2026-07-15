# C1 subobject-bounds narrowing — proposal (design, awaiting review)

*Status: **v1 IMPLEMENTED + VALIDATED (2026-07-09), PI-approved.** The design
below is the full proposal; v1 shipped a deliberately narrowed slice of it (see
the v1-scope note under §2 and the rollout in §6). Author: B-lane
(compiler/codegen + emulator lane), branch `capstone-bootstrap-b`, 2026-07-08.*

> **v1 implementation note (2026-07-09).** Shipped: the `-fcapstone-subobject-bounds`
> flag (default off, Capstone-only), a frontend `CGExpr.cpp` hook
> (`maybeNarrowSubobjectBounds`) that narrows **array-typed** fields only, with
> refusals for unions, flexible/incomplete arrays, last-member (trailing) arrays,
> and incomplete types. Rationale for arrays-only: it flips the headline gap
> (`subobject_overread`) with **zero container_of exposure** (arrays are never
> container_of subjects), so the offsetof-pattern refusal + `no_subobject_bounds`
> opt-out attribute (§2.2, §5 Q1) and embedded-record/scalar-field narrowing move
> to **increment 2**, where they are actually load-bearing. Validated: clang lit
> `capstone-subobject-bounds.c` (on/off) + Capstone clang lit 7/7 + backend lit
> 36/36; runtime authority `subobjfield_*` 5/5 (`overrun`→bounds-fault;
> `inbounds`/`union_active`/`flexarray`→ok; un-flagged `subobject_overread` still
> no-trap-today). Files: `clang/{include/clang/Basic/LangOptions.def,
> include/clang/Driver/Options.td, lib/CodeGen/CGExpr.cpp,
> test/CodeGen/capstone-subobject-bounds.c}`,
> `capstone/tests/capstone-authority/{domains/subobjfield_*.c, oracle.tsv,
> build-authority-suite.sh}`.

*Evidence base (trusted over any summary): `design/capability-bounds-model.md`
(the `SHRINK` primitive + exact-bounds caveat), `design/c1-coverage-matrix-and-overhead.md`
(what object-granularity narrowing does/doesn't cover), the object-narrowing code
in `llvm/lib/Target/Capstone/CapstoneISelDAGToDAG.cpp` (`selectLGA`,
`narrowToFrameObjectBounds`, `materializeFrameIndexAddrBase`, `selectShrink`), and
the authority suite `capstone/tests/capstone-authority/` (`oracle.tsv` +
`domains/subobject_overread.c`).*

---

## 1. The gap and why it matters

### 1.1 What is narrowed today (object granularity)

C1 narrows a capability to a whole **object** at the point the object's base
capability is materialized:

- **Globals** — `selectLGA` (`CapstoneISelDAGToDAG.cpp:1388`) materializes
  `&g = cincoffset(gp, pcrel(g))`, delinearizes it, then (when
  `-capstone-shrink-globals`, **default on**) emits
  `SHRINK &g, cursor=&g, end=&g+sizeof(g)`.
- **Stack** — `narrowToFrameObjectBounds` (`:3921`), shared by the bare
  `ISD::FrameIndex` path and `materializeFrameIndexAddrBase` (`:3946`, the
  interior-pointer / load-store-base path), emits
  `SHRINK cap, cursor, cursor+MFI.getObjectSize(FI)` when
  `-capstone-shrink-stack` (**default on** as of 2026-07-03).
- **Heap** — source-level: allocators call `__builtin_capstone_cap_shrink`
  (clang builtin `cap_shrink`, `BuiltinsCapstone.td:179`) on the returned
  pointer.

All three narrow to `[base, base + sizeof(**whole object**))`.

### 1.2 Where object granularity stops and subobject would begin

The narrowing happens **once, at object materialization**. A field access such as
`object.first[index]` lowers, in SelectionDAG, to pointer arithmetic
(`cincoffset(object_cap, field_offset + index*stride)`) **off the already-narrowed
whole-object capability**. The GEP that projects "field `first` of struct
`object`" is flattened into an integer offset; the field structure is gone by
ISel. So the derived pointer keeps the **whole-object** bound, and any access that
stays inside the object — even if it leaves the *field* — does not fault.

This is pinned by the authority probe `domains/subobject_overread.c`:

```c
struct adjacent_fields { unsigned char first[8]; unsigned char second[8]; };
static struct adjacent_fields object = {{0}, {0xA5}};
void domain_main(unsigned *res, unsigned func) {
  volatile unsigned index = 8;                 // deliberately off the end of first[]
  *res = 0x22070000u | (unsigned)object.first[index];
}
```

`object` is a 16-byte global; `selectLGA` narrows it to `[&object, &object+16)`.
`object.first` is at offset 0, so `object.first[8]` is `&object + 8` — the first
byte of `second[]`, **still inside** the 16-byte object bound. Result: no fault,
`retval = 0x220700A5` (the `0xA5` marker from `second[0]`).
`oracle.tsv:44` records this as `subobject_overread  no-trap-today  570884261`,
and `README.md §7` states: *"Whole-object global SHRINK does not provide
field-level spatial isolation."* That `no-trap-today` line **is** the subobject
gap.

### 1.3 Why it matters for the paper

The C1 coverage matrix (`c1-coverage-matrix-and-overhead.md §1`) lists
"Subobject / struct field — **No** — authority struct-field over-read =
no-trap-today" as the head of the residual gap set. The 2026-06-29 audit's
strategic note is that **object bounds re-derive CHERI**; a working, measured
**subobject** story is one of the few places C1 can claim to go *past* the
CHERI-object baseline (CHERI itself only offers subobject bounds as an opt-in,
famously fragile, mode). Closing this — even conservatively and opt-in — turns a
"no-trap-today" line into a demonstrated bounds-fault and lets the paper state a
field-granularity result with honest, measured caveats.

**Non-goal:** this proposal does **not** claim a spatial-safety theorem. Broad
`gp`/`sp` roots, RWX permissions, and function capabilities remain out of scope
(unchanged from the object-granularity honesty statement).

---

## 2. Proposed policy (conservative, opt-in, default OFF)

**One-paragraph statement.** When the address of a **statically-typed subobject**
is *materialized as a capability that escapes* (address-of a field, array-to-
pointer decay of a field array, or a reference bound to a field), and only then,
narrow that capability monotonically to `[&field, &field + sizeof(typeof field))`
on top of the existing whole-object bound — but **refuse** to narrow in the cases
where a narrowed field capability would break valid, common intra-object idioms:
unions, flexible / trailing array members, arrays that are the last member,
`container_of`/`offsetof`-style negative walks, incomplete/zero-sized field types,
and the degenerate case where the field already spans the whole object. The whole
transform is behind a new **default-off** flag `-fcapstone-subobject-bounds` (with
a matching backend `-mllvm -capstone-shrink-subobject` guard), exactly the staged
rollout `-capstone-shrink-globals`/`-capstone-shrink-stack` used — off first,
validated across the corpus, flipped on only after a clean matrix.

### 2.1 When to narrow

Narrow the projected capability to the field type's size when **all** hold:

1. The projection is a **field access whose result is a pointer/lvalue that can
   escape**: `&s.f`, decay of `s.arr` (a field array used as a pointer), binding a
   C++ reference to `s.f`. (A direct scalar load/store `s.x` that never yields a
   capability to other code needs no narrowing — it already can't over-read past
   the loaded scalar.)
2. The field has a **complete, sized type** with `sizeof(field) > 0`.
3. The field is **not** the last member if that member is an array (see FAM /
   trailing-array refusal below).
4. The container is a **struct/class**, not a union.
5. The narrowed bound is a **strict subset** of the current object bound (offset+
   size ≤ object size; it always is for a real field, but assert it — this keeps
   `SHRINK` monotone and never a no-op-that-costs-instructions).

For a field array `s.arr[i]`, narrowing to `sizeof(s.arr)` (the whole array field)
gives the desired behaviour: **intra-array indexing is in-bounds; walking off the
array field's end into the next field faults.** This is precisely the
`subobject_overread` case we want to flip.

### 2.2 When to refuse (and why)

| Case | Why narrowing is wrong | Action |
|---|---|---|
| **`container_of` / `offsetof` back-walk** | Code takes `&s.f` and does pointer arithmetic *backward* (`(struct S*)((char*)p - offsetof(S,f))`) to recover the container. A field-narrowed cap makes the back-walk leave `[&f, &f+sz)` → spurious bounds-fault. Canonical in kernel/intrusive-list code. | Do **not** narrow when the address-of result flows into subtractive pointer arithmetic; conservatively, allow a whole-program/function opt-out attribute and keep default off. Detect the syntactic `offsetof`-difference pattern in CodeGen and skip. |
| **Unions** | Members overlap; the active member may be larger than the one whose lvalue is spelled. Narrowing to one member's size can cut off a valid access to a larger active member. | Refuse (narrow to the **whole union** at most, i.e. no sub-union narrowing). |
| **Flexible array member** (`struct { int n; char data[]; }`) | Declared size of `data[]` is 0/incomplete; real accesses run past it into the allocation. Narrowing to declared size breaks every use. | Refuse for the FAM and for the containing struct's FAM projection. |
| **Trailing array `[0]`/`[1]` (pre-C99 FAM idiom)** | Same as FAM — the last-member array is deliberately over-indexed. | Refuse when the array is the **last member** of the struct. |
| **Arrays inside structs (non-trailing)** | Legal C only indexes *within* the array; over-indexing into the next field is UB. Intra-array is what we keep in-bounds. | **Narrow** to `sizeof(arr)` — this is the wanted behaviour, not a refusal. |
| **Pointer arithmetic that walks between fields** | `&s.a` then `+k` to reach `s.b` is UB in ISO C but appears in some hand-packed idioms. This is exactly the `subobject_overread` UB we intend to trap. | **Narrow** (this is the feature). Documented as a behaviour change; the opt-out attribute covers code that relies on it. |
| **Incomplete / zero-sized field type** | No size to narrow to. | Refuse. |
| **Bit-fields** | No addressable capability is produced. | N/A (never narrowed). |
| **Field == whole object** (single-field struct, offset 0, equal size) | Narrowing is identical to the object bound — pure cost, no benefit. | Skip (no-op elision). |

### 2.3 Default

**Default OFF**, behind `-fcapstone-subobject-bounds`. Rationale: (a) it is the
one C1 increment that can *break correct programs* (container_of), so it must not
silently change behaviour; (b) it mirrors the proven staged rollout of the other
two shrink flags; (c) the honest paper story is "opt-in field-granularity,
validated across corpus X, default-off pending a container_of-safe default policy"
— which is *more* defensible than CHERI's historically-fragile subobject mode.

---

## 3. Where it hooks in, and how it composes with object `SHRINK`

### 3.1 Hook site: the **frontend** (Clang CodeGen), not the backend

This is the load-bearing design decision. **SelectionDAG cannot do subobject
bounds reliably**: by ISel the field-projection GEP is already flattened into
`cincoffset(object_cap, integer_offset)`, indistinguishable from ordinary in-object
pointer arithmetic. The object-narrowing sites work precisely because a *whole
object*'s base is a distinguished node (a `GlobalAddress` in `selectLGA`, a
`FrameIndex` in `narrowToFrameObjectBounds`). A field has no such distinguished
node.

The **field structure is only available in Clang CodeGen**, at the `MemberExpr` /
array-subscript-of-field lowering. This is also where CHERI implements
`-cheri-bounds=subobject`. Concretely, the hook is in `clang/lib/CodeGen/CGExpr.cpp`:

- `CodeGenFunction::EmitLValueForField` (and the `MemberExpr` path in
  `EmitMemberExpr`) — for `&s.f` and field lvalues that escape.
- `EmitArraySubscriptExpr` — for `s.arr[i]` where `s.arr` is a field array
  (narrow the array-base to `sizeof(arr)` before the index add).

At the hook, when §2.1 says "narrow", emit a call to the existing
`int_capstone_cap_shrink` intrinsic (via the `cap_shrink` builtin machinery, so no
new intrinsic is needed) on the projected pointer:
`p' = cap_shrink(p, addr(p), addr(p) + sizeof(field))`. The `SHRINK` monotonicity
check in QEMU (`helper_csshrink`) guarantees this is only ever a narrowing.

> Alternative considered and rejected: carry a "this GEP is a field projection +
> its field size" marker on the IR GEP (metadata or a dedicated intrinsic) and
> narrow in the backend to keep all C1 logic in one place. Rejected because the
> *decision* (which projections are safe to narrow — the §2.2 refusals) inherently
> needs the AST type info, so the frontend must make the call anyway; adding a
> backend leg only splits the logic. Emitting the `cap_shrink` intrinsic in the
> frontend keeps the whole policy where the type information lives, and the backend
> already lowers that intrinsic (`selectShrink`).

### 3.2 Composition with object narrowing

`SHRINK` is **monotone** (`capability-bounds-model.md §3`: it raises
`Illegal operand value` unless `base ≥ rd.base ∧ end ≤ rd.end`). So the two
narrowings **compose cleanly and order-independently**:

```
segment (gp/sp)  ⊇  object  [obj, obj+sizeof(obj))   ← selectLGA / frame narrowing (backend)
                 ⊇  field   [f,   f+sizeof(field))    ← subobject shrink (frontend), THIS proposal
```

Because a real field satisfies `obj ≤ f` and `f+sizeof(field) ≤ obj+sizeof(obj)`,
the frontend `cap_shrink` on the projected pointer is always a legal further
narrowing of whatever object bound the backend later stamps on the base. No change
to `selectLGA`/`narrowToFrameObjectBounds` is required; they run as-is on the base,
and the extra frontend shrink tightens the derived pointer. (Heap: same — the
allocator's object `cap_shrink` and a subsequent field `cap_shrink` compose.)

### 3.3 Bounds-exactness note

Per `capability-bounds-model.md`, this QEMU keeps exact fat bounds in a side
table, so field bounds are **exact at all sizes** here. In a faithful 128-bit-only
implementation, sub-4 KiB fields (the overwhelming majority) are byte-exact; the
rounding rule only bites for ≥4 KiB fields (rare). State it as "exact in the
current model" — do **not** cite rounding as measured evidence.

---

## 4. Test + authority plan

### 4.1 Flip the existing gap probe

`subobject_overread` currently expects `no-trap-today` (retval `0x220700A5`). Under
`-fcapstone-subobject-bounds` it must become **`bounds-fault`** (`first[8]` leaves
`[&first, &first+8)`), while remaining `no-trap-today` with the flag off. Because
the authority runner keys expectations per domain, add a build-variant mechanism
(like the `stack_*` probes already use `-capstone-shrink-stack=true`): either a
new `subobject_*` case in `build-authority-suite.sh` that passes
`-fcapstone-subobject-bounds`, and/or a second oracle column for the flag-on
expectation.

### 4.2 New authority probes (`capstone/tests/capstone-authority/domains/`)

| Probe | Flag on | Purpose |
|---|---|---|
| `subobject_inbounds` | `ok` | `object.first[3]` — in-field access still works |
| `subobject_array_overrun` | `bounds-fault` | field array walked off its end into the next field (the feature) |
| `subobject_union_active` | `ok` | write via the larger active union member after taking a smaller member's address — must **not** fault (refusal correctness) |
| `subobject_flexarray` | `ok` | access `data[k]` of a `struct{int n; char data[];}` — FAM refusal keeps it in-bounds |
| `subobject_trailing_array` | `ok` | last-member `[1]` array over-index — trailing-array refusal |
| `container_of_roundtrip` | `ok` | `&s.f` → back-walk to container → read another field — must **not** fault (container_of refusal) |
| `subobject_whole_object` | `ok` | single-field struct — no-op elision, identical to object bound |

Each ships source + the expected `oracle.tsv` line, matching the suite's existing
"source + asm + QEMU trap/no-trap vs oracle" shape.

### 4.3 lit (frontend)

`clang/test/CodeGen/Capstone/cap-subobject-bounds.c` (new): with
`-fcapstone-subobject-bounds`, CHECK that a `cap_shrink`/`int.capstone.cap.shrink`
is emitted for `&s.f` and for a field-array decay, and CHECK-NOT for a union
member, a FAM/trailing array, a `container_of` difference, and a whole-object
single field. An off-arm (`-fno-...` / absent) CHECK-NOTs any shrink. Mirror the
on/off A-B structure of `cap-shrink-globals.ll` / `cap-shrink-stack.ll`.

### 4.4 Regression gate

Subobject narrowing is **frontend + default-off**, so the object-granularity
backend gates are unaffected while off. For the flag-on validation before any
default flip:

```bash
source capstone/tests/capstone-test-env.sh
"$CAPSTONE_LLVM_LIT" -sv clang/test/CodeGen/Capstone llvm/test/CodeGen/Capstone
bash capstone/tests/capstone-authority/run-authority-suite.sh        # incl. new probes
# corpus false-positive sweep — the real risk is spurious traps, not miscompiles:
#   rebuild CoreMark + run-all-beebs (+ RV8) WITH -fcapstone-subobject-bounds,
#   expect zero new bounds-faults; any fault is a container_of-class false positive
bash capstone/tests/runtime-qemu/run-coremark.sh
bash capstone/benchmarks/beebs/run-all-beebs.sh
```

(These builds are the point at which the ~10 GB LLVM build is actually needed —
**explicitly out of scope for this design phase**; they run only after sign-off.)

---

## 5. Open questions / risks for the reviewer

1. **`container_of` safety is the whole ballgame.** The benchmark corpus is mostly
   self-contained numeric code and may not exercise `container_of`; that would make
   a corpus sweep *look* clean while masking the real-world hazard. **Q:** do we
   want (a) a syntactic CodeGen refusal for the `offsetof`-difference pattern, (b)
   an opt-out function/type attribute (`__attribute__((no_subobject_bounds))`), (c)
   a whole-file pragma, or some combination? CHERI shipped all three and still had
   friction. Recommend (a)+(b) minimum.
2. **Escape analysis scope.** Narrowing only *escaping* field capabilities (vs
   every field lvalue) avoids pointless shrinks on direct scalar `s.x`. **Q:** is a
   simple syntactic rule (address-of, array decay, reference bind) sufficient, or
   do we need dataflow? Recommend syntactic to start (matches CHERI).
3. **Union / FAM policy = refuse vs narrow-to-max.** Proposal refuses (safest).
   **Q:** acceptable to leave unions entirely un-narrowed for v1?
4. **Nested subobjects** (`s.a.b.c`). Narrow to the innermost field, or the
   outermost escaping projection? Monotonicity makes innermost strictly safe.
   **Q:** confirm innermost is the intended granularity.
5. **Overhead.** Each narrowed projection adds an `lcc`+`add`+`shrink` (~the same
   ~15.6 B/site as globals, but at **use** sites, which are far more numerous than
   global-materialization sites). Field-dense code could see materially higher
   code-size and dynamic-instruction cost than object narrowing. **Q:** is a
   per-projection cost acceptable, or should we cache/narrow-once-per-pointer?
   (No runtime-cycle measurement is possible on this functional QEMU — same caveat
   as the object overhead table.)
6. **Fault delivery.** A subobject bounds-fault hits the same in-domain
   cap-fault path that currently halts the domain cleanly (`current-state.md`
   2026-07-03 note); no new runtime work, but worth confirming the halt line is
   acceptable evidence for the paper's probes.
7. **`SHRINKTO`.** For the "narrow to `sizeof(field)` at the cursor" shape,
   `SHRINKTO` (`csshrinkto`, unwired in LLVM per `capability-bounds-model.md §4`)
   is a cleaner single-op lowering than `lcc`+`add`+`SHRINK`. Wiring it is optional
   and orthogonal; note it as a possible follow-up optimization, not a v1 blocker.

---

## 6. Incremental rollout

1. **Design sign-off** (this doc) — reviewer picks the container_of policy (Q1) and
   confirms Q3/Q4. **← we are here; STOP for review.**
2. **Frontend + intrinsic emission**, default off. Land `EmitLValueForField` /
   array-decay hook emitting `cap_shrink` under `-fcapstone-subobject-bounds`, with
   the §2.2 refusals. Add lit `cap-subobject-bounds.c`.
3. **Authority probes** — add the §4.2 domains + oracle lines; flip
   `subobject_overread` under the flag. Prove field-granularity traps and that all
   refusal cases stay `ok`.
4. **Corpus false-positive sweep** — rebuild CoreMark/BEEBS/RV8 with the flag on;
   drive out any container_of-class spurious traps; record results (the point where
   the big build is finally required).
5. **Overhead measurement** — code-size delta per corpus (same method as
   `c1-coverage-matrix-and-overhead.md §2`), reported honestly (code size only).
6. **Default-on decision** — only after (4) is clean and (Q1) has a shipped
   container_of-safe default; otherwise stay opt-in and say so in the paper.

---

*This document lives in `design/` because it is an architecture decision (a new
narrowing policy + its backend/frontend hook), per CLAUDE.md's rule that design
decisions go here while bug-fix/root-cause trails go in `history/`. It references
`design/capability-bounds-model.md` (the `SHRINK` primitive and exact-bounds
caveat) and the authority suite (`capstone/tests/capstone-authority/`) as its
evidence base. No code has been written and no build has been run.*
