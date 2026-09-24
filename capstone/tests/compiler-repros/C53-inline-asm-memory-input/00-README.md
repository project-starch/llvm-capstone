# C-53 — an inline-asm `"m"` INPUT operand crashes isel; `"=m"` outputs compile

**A COMPILER bug.** Found 2026-09-23 by CPython's `configure`, whose x87 and mc68881 FPU checks
use one (`capstone/ports/cpython/interpreter/`). CPython on capstone64 never needs those checks
to say yes, so this blocks nothing there today; `prepare-cpython-capstone.sh` records the two
crashed checks as reviewed. Siblings from the same port: C-50, C-51, C-52, C-54.

## Reproducer

    void f(unsigned int *p) { __asm__ __volatile__("lw zero, %0" : : "m"(*p)); }

    SelectionDAGBuilder.cpp:10392: visitInlineAsm:
      Assertion `InOperandVal.getValueType() == TLI.getPointerTy(DAG.getDataLayout())
                 && "Memory operands expect pointer values"' failed.

`./run.sh` compiles five shapes at `-O0` and `-O1`; the two `"=m"` OUTPUT shapes are its
controls and must compile. `CLANG=` selects the compiler.

| shape | `-O0` | `-O1` |
|---|---|---|
| `"=m"(*p)` output through a pointer | ok | ok |
| `"=m"(local)` output | ok | ok |
| `"m"(*p)` input through a pointer | **assert** | **assert** |
| `"m"(local)` input | **assert** | **assert** |
| `"m"(global)` input | **assert** | **assert** |

The same on `d030df93d4a4` (dev), `d5b5f11cae8f` and `f7b50f081ca4`.

## Mechanism, read at `d030df93d4a4` (not confirmed by a fix)

For a memory INPUT operand, `visitInlineAsm` asserts that the operand's value type is
`TLI.getPointerTy(DL)` -- the pointer type of address space 0, `i64` here. The operand is an
`addrspace(200)` capability, `c128`. Comparing against the pointer type of the operand's own
address space is the obvious reading of the intent; whether the rest of the memory-operand path
then handles a capability is not known.

## Impact

Any C that passes memory INTO inline assembly with `"m"`, which is how inline-asm code usually
reads memory. No port here does so today (a grep of CPython found only `Python/pymath.c`, whose
x87 code is never built for this target).
