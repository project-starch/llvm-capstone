# Decisions waiting on the lead — the CPython interpreter port, 2026-09-23

Four questions only the lead can answer. Each has the evidence, what happens either way, and
what it costs. The work they gate is in [cpython-interpreter-port.md](cpython-interpreter-port.md);
the IDs (A1, B1, R1, ...) are that plan's. Read in order: **item 1 decides whether the others
matter.**

Nothing below is blocked from being measured; what is blocked is choosing a direction and
asking another lane to build for it.

## 1. Is a full CPython domain in scope, given the size wall? (R1)

**Evidence.** The first link of everything that compiles is 8.37 MiB (`PT_LOAD` memsz) without
29 objects, among them `unicodeobject`, `typeobject` and `ceval`. Scaled by the measured
capstone/native size ratio (1.30), the whole static image is **~11.5 MiB, an estimate**, before
any heap. A domain is one contiguous kernel allocation, capped at **4 MiB** by `MAX_ORDER`
(`ports/musl-capstone/README.md`). `-Os` was measured to save 5.4 %. The mruby port's tests stop
at image size too (commit `10322a2fb795`, by its subject). Results:
`ports/cpython/interpreter/results/link-2026-09-23-d030df93d4a4.txt`.

**Options.**

| | what it takes | what it gives |
|---|---|---|
| a. raise the domain limit (larger contiguous allocation, or a domain of several regions) | kernel module + runtime + loader work; the loader today sizes one segment | CPython, mruby and any large port become possible |
| b. a reduced CPython (drop `unicodedata`, codecs, most stdlib modules; frozen minimal startup) | port work only; how small it can get is not measured | still unlikely under 4 MiB: `.text` of what compiles is 5.0 MiB |
| c. stop at the survey and link as evidence, and use MicroPython (already running in a domain) as the Python workload | nothing | the six compiler findings stand either way |

**Recommendation.** (a) if large application ports are a goal; otherwise (c). (b) alone does
not close a 3x gap on present numbers.

## 2. A capability-carrying integer, or capability builtins? (B1 vs B2)

**Evidence.** On capstone64 `uintptr_t` is 8 bytes and a pointer 16 (configure: `SIZEOF_VOID_P
16`, `SIZEOF_UINTPTR_T 8`); a pointer that goes through an integer comes back untagged. clang's
`-Wcapstone-pointer-roundtrip` counts **62 such sites in 15 CPython files** among what compiles.
The one that matters first is the GC: `_gc_next`/`_gc_prev` are `uintptr_t` with flag bits in
the low bits, on every GC-tracked object (`Include/internal/pycore_gc.h`, `Python/gc.c`).
`BuiltinsCapstone.td` has no capability builtins today; the runtime writes raw `.insn`
encodings in inline asm.

**Options.**

| | B1: `uintptr_t` carries a capability (as CHERI's `__uintcap_t`) | B2: capability builtins (get/set address, bounds, tag) |
|---|---|---|
| CPython | most of the 62 sites work as written | each site is rewritten by hand; GC first |
| compiler | a new integer type whose arithmetic keeps the tag; larger | a handful of builtins; smaller |
| musl and every other port | **ABI change**: `sizeof(uintptr_t)` becomes 16 for all of them; the musl arch layer already works around the 8-byte type (`syscall_arch.h`, `pthread_arch.h`, `lite_malloc`) | none |
| precedent | CheriBSD's C ABI, where `uintptr_t` is `__uintcap_t` (whether CPython itself was ported there was not checked) | none here |

**Recommendation.** B2 first: it is small, useful to every port (the runtime's own inline asm
becomes C), and does not commit the ABI. Revisit B1 if the rewrite count grows past what a patch
series can carry.

## 3. Where C-51 and C-54 sit in the compiler lane's queue (A1, A2)

**Evidence.** C-51 (`llvm.ptrmask` on a capability; every 8/16-bit atomic) stops 26 of 253
objects and is the difference between the survey and a complete link. C-54 (atomics on a
pointer become `__atomic_*_16` calls) leaves 3 undefined symbols now and more once C-51 is
fixed. `d5b5f11cae8f` names both as outside its scope. C-32 is OPEN in the registry and was the
compiler lane's live blocker at its 2026-09-17 handover (`2026-09-17-compiler-lane-handover.md` §2).

**Question.** After C-32, or before? A2 has an interim the port can do alone -- the runtime
defines `__atomic_*_16` as plain capability loads/stores/compares, correct on a one-hart domain
only -- so the compiler lane's part could be A1 alone.

**Recommendation.** A1 after C-32; A2's interim in the port, the real lowering later.

## 4. Does the silicon have floating-point hardware? (C3)

**Evidence.** Domains are built soft-float (`docs/design/capstone-softfloat-libm.md`); every
`float`/`double` operation in CPython becomes a compiler-rt call. Not measured: its share of the
1.30x size ratio or of run time.

**Question.** Is an F/D extension present or planned on the CVA6 configuration? If not, this
item closes and soft-float is the design.
