# C-55 — two cascaded selects on a capability with a null operand put `$c0` into a PHI

**Fixed** by `b3fdbbe081e1`, the commit before this folder on its branch. Measured 2026-09-23: `./run.sh` reports ABSENT with that compiler and PRESENT with `dev`'s (`1a08706b6344`, whose compiler sources are those of `d030df93d4a4`). Regression test: `llvm/test/CodeGen/Capstone/cascaded-select-null-cap.ll`. Everything below is the investigation as it was recorded before the fix.

**A COMPILER bug, and a residual of a fixed one**: `d5b5de228b38` ("Fix null capability select
lowering", 2026-03-26) fixed exactly this shape for a single select and left the cascaded path
beside it unchanged. Found 2026-09-23 when the CPython compile survey ran at `-Os`
(`capstone/ports/cpython/interpreter/`, `--opt=-Os`): `Objects/dictobject.c`, which compiles at
`-O3`, stops on it. Siblings from the same port: C-50 to C-54.

## Reproducer

Nine lines of IR (`src/two-selects-null.ll`):

    %x = select i1 %c, ptr addrspace(200) %a, ptr addrspace(200) null
    %y = select i1 %c, ptr addrspace(200) null, ptr addrspace(200) %x
    ret ptr addrspace(200) %y

    -O1..-O3: LiveVariables.cpp:114: Assertion `Reg.isVirtual() && "getVarInfo: not a virtual register!"'
    -O0:      PHIElimination.cpp:552: Assertion `SrcReg.isVirtual() && "Machine PHI Operands must all be virtual registers!"'

`src/dict___contains__.reduced.ll` is what `llvm-reduce` made of CPython's `dict___contains__`
at `-Os` (predicate `src/interesting.sh`: llc aborts on the LiveVariables assertion, checked
positive on the original and negative on a trivial module); it is the same two selects.

`./run.sh` compiles three controls and the two failing files at `-O0` to `-O3`, with
`-Xclang -disable-llvm-passes`, because the IR optimizer would otherwise fold two selects on one
condition into one and the crash would go. Controls: one select with null, two selects without
null, and the same two selects on `i64`; all must compile.

| file | `-O0` | `-O1`..`-O3` |
|---|---|---|
| one select with null (control) | ok | ok |
| two selects, no null (control) | ok | ok |
| two selects on `i64` with 0 (control) | ok | ok |
| **two selects with null** | PHIElimination assert | LiveVariables assert |
| CPython's `dict___contains__`, reduced | PHIElimination assert | LiveVariables assert |

The same on `d030df93d4a4` (dev), `d5b5f11cae8f` and `f7b50f081ca4`.

## Mechanism

**Seen in the machine code.** Straight after `finalize-isel` the two selects are one PHI whose
incoming values include the physical null-capability register:

    %4:gpcr = PHI $c0, %bb.0, %0:gpcr, %bb.1, $c0, %bb.2

A PHI's operands must be virtual registers; the two assertions are the two passes that check.

**Read in the source at `d030df93d4a4`, not confirmed by a fix.** `emitSelectPseudo`
(`llvm/lib/Target/Capstone/CapstoneISelLowering.cpp`) hands a select whose successor consumes its
result to `EmitLoweredCascadedSelect` (~`:23308`). The general path below it passes each PHI
source through `materializeSelectPHISource` (~`:23397`), which COPYs a physical register into a
virtual one; that is `d5b5de228b38`'s fix. `EmitLoweredCascadedSelect` builds its PHI from the raw
operand registers (~`:23254`) and never received it. The obvious fix is the same materialization
there.

## Impact

One CPython object, and only at `-Os`; at `-O3` the optimizer produces a different shape. Any
C that selects between a capability and NULL twice on one condition can reach it.
