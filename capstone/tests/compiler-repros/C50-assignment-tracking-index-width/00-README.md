# C-50 — with `-g` at `-O1`+, Assignment Tracking asserts on any local whose address escapes

**Fixed** by `f8b140caa818`, the commit before this folder on its branch. Measured 2026-09-23: `./run.sh` reports ABSENT with that compiler and PRESENT with `dev`'s (`1a08706b6344`, whose compiler sources are those of `d030df93d4a4`). Regression test: `clang/test/CodeGen/capstone-assignment-tracking-index-width.c`. `candidate-fix.diff` is the changed line; the commit adds only a comment to it. Everything below is the investigation as it was recorded before the fix.

**A COMPILER bug, in generic LLVM code, reachable on capstone64 because a pointer is 128 bits
wide and its index is 64.** Found 2026-09-23 by the CPython compile survey
(`capstone/ports/cpython/interpreter/`), where it alone failed 146 of 253 objects: CPython builds
with `-g -O3`, and no port here had built with `-g` and optimisation before. Sibling issues found
by the same survey: C-51 (`llvm.ptrmask` on a capability, reached by every 8- and 16-bit atomic,
`../C51-ptrmask-on-capability/`)
and C-52 (a Greedy register allocator segfault, `../C52-greedy-regalloc-segfault/`).

## Reproducer

    void g(char *);
    void f(void) { char buf[32]; buf[5] = 1; g(buf); }

    clang -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m -ffreestanding -g -O1 -c

    Value.cpp:731: ... stripAndAccumulateConstantOffsets(...):
      Assertion `BitWidth == DL.getIndexTypeSizeInBits(getType()) &&
                 "The offset bit width does not match the DL specification."' failed.
    Running pass 'Assignment Tracking Analysis' on function '@f'

`./run.sh` prints three controls and a verdict (exit 1 = PRESENT); `./run.sh workaround` shows
the flag that avoids it. `CLANG=` selects the compiler. Neither needs a board or QEMU.

| arm | result |
|---|---|
| `escaping-local.c -O1` (no `-g`) | compiles |
| `escaping-local.c -g -O0` | compiles |
| `promoted-local.c -g -O1` (the local is removed by SROA) | compiles |
| `escaping-local.c -g -O1` | **asserts** |
| `escaping-local.c -g -O1 -Xclang -fexperimental-assignment-tracking=disabled` | compiles |

Present on `d030df93d4a4` (dev), `d5b5f11cae8f` and `f7b50f081ca4`: not a regression.

## Where, and how that was established

`llvm/lib/CodeGen/AssignmentTrackingAnalysis.cpp:271`, in `walkToAllocaAndPrependOffsetDeref`:

    APInt OffsetInBytes(DL.getTypeSizeInBits(Start->getType()), false);
    Value *End = Start->stripAndAccumulateInBoundsConstantOffsets(DL, OffsetInBytes);

The accumulator is sized with the POINTER's bit width, 128 for an `addrspace(200)` capability,
and `stripAndAccumulate...` asserts that it has the INDEX width, 64. The same pass sizes its other
accumulator correctly (`:2093`, `Layout.getIndexTypeSizeInBits(MemOp->getType())`), as does
`llvm/lib/IR/DebugInfo.cpp:1993`. Wherever a pointer's width equals its index width the two
expressions agree, which is why the line is harmless on the common targets.

**Established by a matched pair, not read off the source.** `candidate-fix.diff` changes that
one line to `getIndexTypeSizeInBits`. A clang built from `d030df93d4a4` plus that line only:
`escaping-local.c` and `escaping-struct.c` (a struct holding a pointer, which asserts the same
way unpatched) compile at `-g -O1`, and CPython's `Objects/bytesobject.c` compiles with its own
flags (`-g -O3`) without the workaround. The
unpatched build of the same tree, rebuilt afterwards, is byte-identical to the original
(`clang-22` SHA-256 `acb66996…` before and after).

**NOT established:** that the fix is complete (it has not run lit or the QEMU suites), and that
the resulting DWARF location expressions are right for a capability-addressed local. The fix is
a compiler change and belongs to the compiler lane; until it lands, CPython's `capstone-cc`
passes `-Xclang -fexperimental-assignment-tracking=disabled`, which turns off this one
debug-info analysis and keeps `-g`.
