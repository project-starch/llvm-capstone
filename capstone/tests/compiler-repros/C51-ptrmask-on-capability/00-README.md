# C-51 — `llvm.ptrmask` on a capability crashes isel; every 8- and 16-bit atomic reaches it

**A COMPILER bug.** Found 2026-09-23 by the CPython compile survey
(`capstone/ports/cpython/interpreter/`): CPython 3.13's `PyMutex` is one byte locked by
compare-exchange, and 26 of 253 objects stop on it. Sibling issues from the same survey: C-50
(`../C50-assignment-tracking-index-width/`, a debug-info assert under `-g`) and C-52
(`../C52-greedy-regalloc-segfault/`, a register allocator segfault).

## Reproducer

Five lines of IR, no atomics involved (`src/ptrmask.ll`):

    define ptr addrspace(200) @g(ptr addrspace(200) %p) addrspace(200) {
      %a = call ptr addrspace(200) @llvm.ptrmask.p200.i64(ptr addrspace(200) %p, i64 -4)
      ret ptr addrspace(200) %a
    }

    SelectionDAG.cpp:1826: SelectionDAG::getShiftAmountConstant(...):
      Assertion `VT.isInteger() && "Shift amount is not an integer type!"' failed.
      ... SelectionDAGBuilder::visitIntrinsicCall

And the C a program actually contains (`src/atomics.c`, `src/pymutex.c`), at `-O1` with `+a`:

| | u8 | u16 | u32 | u64 |
|---|---|---|---|---|
| compare-exchange | **crash** | **crash** | ok | ok |
| fetch-add | **crash** | **crash** | ok | ok |
| exchange | **crash** | **crash** | ok | ok |

`./run.sh` prints `ptrmask.ll` alone, that matrix and CPython's `PyMutex_Lock` shape, then a
verdict (exit 1 = PRESENT). The 32- and 64-bit cells are its controls: if one of them fails the
verdict is CONTROL FAILED, not a count. `CLANG=` selects the compiler.

| compiler | `ptrmask.ll` | matrix |
|---|---|---|
| `d030df93d4a4` (dev) | crash | u8/u16 crash, u32/u64 ok: PRESENT |
| `d5b5f11cae8f` | crash | same: PRESENT |
| `f7b50f081ca4` | crash | CONTROL FAILED -- predates `d5b5f11cae8f`, which added capability-address 32/64-bit atomics |

## Mechanism

A sub-word atomic has no instruction of its own, so AtomicExpand rewrites it as a masked
operation on the aligned 32-bit word, and aligns the address with
`llvm.ptrmask.p200.i64(%p, -4)` (visible with `llc -print-after=atomic-expand`). The 32- and
64-bit forms need no alignment and never emit `ptrmask`.

`SelectionDAGBuilder::visitIntrinsicCall`, `case Intrinsic::ptrmask`, compares the mask (`i64`,
the index width) with the pointer's MEMORY width (128). Because the mask is narrower it takes the
branch written for AMDGPU buffer descriptors, which pads the mask with ones by building
`SHL` on the POINTER's value type -- `c128` here, which is not an integer -- and
`getShiftAmountConstant` asserts. Even past the assert, that branch would AND the whole capability
with a mask; on a capability, masking has to act on the address and leave the metadata alone.
That is a lowering design question for the compiler lane, not a missing guard.

The commit that added capability-address atomics says this edge in its message: "Subword and
capability-valued atomics are outside this change" (`d5b5f11cae8f`). This entry is that edge
with a consumer, and the finding that the defect is `ptrmask` itself, so any other producer of
`llvm.ptrmask` on a capability hits it too. Measured for one: `src/align-down.c`,
`__builtin_align_down(p, 16)` on a `char *`, lowers to `llvm.ptrmask.p200.i64` and crashes the
same way, while the same builtin on an `unsigned long` compiles.

## Workaround

None at the source level that keeps the semantics: CPython's lock must be atomic. Widening
`PyMutex` to 32 bits is a CPython ABI change and would have to be argued as a port patch.
