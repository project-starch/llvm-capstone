# Software Capstone on x86-64

Status: plan, nothing built. Branch `plan/software-capstone-x86`.

## Goal

Run C programs natively on x86-64 under the memory-safety model of the virtual Capstone
profile, enforced by compiler instrumentation and a C runtime instead of hardware:

- spatial safety in the style of SoftBound: every pointer carries bounds and rights in
  metadata kept apart from the pointer;
- temporal safety through Sublet lifetimes instead of CETS key/lock: every capability names
  a node `{index, generation}` in a lifetime forest, and ending a node ends its subtree.

The model is the **non-linear subset** of the virtual profile
([isa.md](../design/virtual-capstone/isa.md): "Sublet child lifetimes", "Authority model",
"Overlap", the generations ruling, "Access and instruction policy"). Linear capabilities are
out of scope by decision.

The result is a precise checker for instrumented C and a software baseline for the hardware,
not a capability system: the metadata is ordinary memory, so code outside the instrumentation
is outside the guarantee, and there is no domain isolation and no PCC.

## Design

**D1. Thin pointers, separate metadata.** The ordinary x86-64 frontend with native layouts.
The purecap frontend (address space 200, 128-bit pointers) would need an x86 capability
lowering and new library ABIs and would still need every check below.

**D2. Metadata.** `{base, end, index, generation, rights, tag}`, beside the SSA value in
registers and in a shadow entry keyed by the storage address in memory, never by the
pointer's value. Pointer-bearing aggregates and ABI-lowered pointers are marked by the
frontend and instrumented before any optimization that erases them; a construct the pass
cannot follow is refused at compile, link or run time, never left unchecked.

**D3. Lifetime forest.** One process-local table of native records. The algorithm comes from
QEMU's `target/riscv/cap_rev_tree.c` (derive, subtree walk, preflight before mutation), with a
64-bit generation: increment on reuse, retire a saturated slot, fail closed on exhaustion,
compare the full id everywhere.

**D4. Operations.** Access checks, `CDERIVE`, `CREVOKE`, `SHRINK`, `TIGHTEN`, removing MANAGE
and non-linear `SPLIT` follow isa.md's operand checks, fault order and all-or-nothing
mutation. Bounds follow the ruled compact profile: `CDERIVE` refuses inexact bounds, other
bounds-forming results round through the capability encoder and decoder (QEMU's
`cap_compress.c`, or a version checked against it), and pointer arithmetic clears the tag
when the source's compressed bounds decode differently at the new cursor (R-60). Exact bounds
is a separately labelled stronger variant. Indirect calls are allowed only to registered
instrumented entries or audited wrappers.

**D5. Tags in memory.** As in isa.md, every non-pointer write clears the metadata it overlaps,
even when the bytes do not change. A per-granule "may hold metadata" bitmap lets a store skip
the shadow when every granule it touches is clear. Integer-to-pointer casts yield untagged
pointers. `memcpy`/`memmove` check both ranges, snapshot the source metadata first, and
carry it only for whole pointers.

**D6. Calls.** A metadata shadow stack carries argument and result metadata,
aggregate fields and variadic arguments included, so native signatures stay unchanged for
direct, indirect and variadic calls. Calls to uninstrumented code go through wrappers that
state what they read, write and return and where they move pointer bytes; callbacks enter
through thunks, and `qsort` moves each element's metadata with its bytes. Unknown boundaries,
unadapted `longjmp` and tail calls are refused.

**D7. Stack, globals, heap.** Matching the pinned compiler: address-taken stack objects and sized
globals get object bounds (`-capstone-shrink-stack`, `-capstone-shrink-globals`) under the lifetime of
their mapping, with no per-frame or per-scope revocation. The heap follows the virtual
runtime (`capstone/runtime/virtual/heap.c`): `malloc` issues a `CDERIVE` child of
`capstone_repr_round(n ? n : 1)` bytes under its group's lifetime, `free` `CREVOKE`s it, and a
successful realloc rotates the lifetime. First over a host-allocator adapter that
returns representable bases and backing for the rounded size, then musl mallocng.
The non-linear functions of `capstone/runtime/include/capstone/capability.h` keep their
signatures, so ports that call them compile unchanged.

Code: the pass in `llvm/lib/Transforms/Instrumentation/`, the runtime in
`capstone/runtime/software/`, contract tests in `capstone/runtime/tests/software-capstone/`.

## Milestones

Each milestone ends at a gate that must pass before the next starts.

| | Scope | Gate |
|---|---|---|
| M0 | Runtime alone: forest, checks, D4 operations as C calls | Runtime and the pinned host-compiled `cap_rev_tree.c` against an independent forest model (QEMU's `tests/virtual-capstone-model/model.py`, extended for generations and reuse); directed saturation, atomicity and rights tests; mutation controls (no subtree walk, no generation compare, no parentage check) must fail it |
| M1 | Pass for heap pointers, with minimal malloc/free wrappers and complete D5 for the admitted instructions | Non-linear cases of `capstone/runtime/tests/sublet-lifetimes` at `-O0` and `-O2`: overflow, access after free, same-address reuse, missing MANAGE each caught where expected, and the safe controls pass |
| M2 | Coverage: casts, `memcpy`, stack, globals, D6 calls | Lit tests and executable contracts at `-O0`/`-O2`; each refused construct has a test that it is refused |
| M3 | libc wrappers, then musl mallocng | Exact output on named single-threaded C ports; injected spatial and temporal faults caught |
| M4 | Agreement with Capstone | Same sources and inputs as the virtual profile on a pinned QEMU, compiler and runtime (default profile, as `runtime/virtual/run-staged.py` runs it), on Juliet heap cases and one bug corpus: all safe controls pass, zero mismatches beyond a list of differences declared before the run. QEMU at dev's pin `a9936e70` keeps `SHRINK` bounds exact until the capability is stored, so either pin a QEMU that rounds at writeback or declare those cases |
| M5 | Cost | Only after M4: access checks, shadow, call transport and revocation measured separately, against an uninstrumented build with the same allocator |

## Out of scope

Linear capabilities, threads, C++, PCC and domain isolation, signals, `dlopen`, JIT,
hand-written assembly.

## Prior art

SoftBoundCETS (public code for LLVM 3.4), its LLVM 12 revision
("SoftBound+CETS Revisited") and MemInstrument (LLVM 12) are references for the pass and the
libc wrappers. None targets LLVM 22 or opaque pointers.
