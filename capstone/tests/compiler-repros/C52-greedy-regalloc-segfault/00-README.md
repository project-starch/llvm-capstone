# C-52 — the Greedy register allocator segfaults in `SplitEditor` on CPython's `compiler_visit_stmt`

**A COMPILER bug.** Found 2026-09-23 by the CPython compile survey
(`capstone/ports/cpython/interpreter/`): `Python/compile.c` is the one object it stops, and that
object is the bytecode compiler, so an interpreter cannot link without it. Sibling issues from
the same survey: C-50 (`../C50-assignment-tracking-index-width/`) and C-51
(`../C51-ptrmask-on-capability/`).

## Reproducer

`src/compiler_visit_stmt.reduced.ll`: CPython 3.13.7's `compiler_visit_stmt` at `-O1`, reduced by
`llvm-reduce` from 2.2 MB of IR to one function of 208 instructions in 53 blocks. The file is 64 KB
because the function still reaches `_PyRuntime`, whose type drags in 903 struct definitions; their
names were shortened to `%T<n>` afterwards and the result re-checked.

    ./run.sh      # control with the basic allocator, then the default; VERDICT, exit 1 = PRESENT

    Running pass 'Greedy Register Allocator' on function '@compiler_visit_stmt'
    SIGSEGV in llvm::SplitEditor::rematWillIncreaseRestriction
      <- SplitEditor::defFromParent <- enterIntvBefore <- splitSingleBlock
      <- RAGreedy::tryBlockSplit <- trySplit <- selectOrSplitImpl

`src/interesting.sh` is the predicate the reduction used: `llc -O1` must die by SIGSEGV inside
that pass, so the reduction could not drift to a different crash or to a clean exit. It was
checked in both directions before use: interesting on the original IR, not on a trivial module.

| arm | `d030df93d4a4` (dev) | `d5b5f11cae8f` |
|---|---|---|
| `llc -O1` (Greedy) | SIGSEGV | SIGSEGV |
| `llc -O1 -regalloc=basic` | compiles | compiles |
| `llc -O0` (Fast) | compiles | compiles |
| clang on `compile.c` at `-O1 -g`, `-O2 -g`, `-O3 -g`, `-O3` | SIGSEGV in all four, same pass and function | SIGSEGV at `-O3 -g` (the only arm run there) |

**Instrument warning.** `clang -O1` on the `.ll` re-runs the IR optimizer, which reshapes the
function, and then it COMPILES. `run.sh` passes `-Xclang -disable-llvm-passes` for that reason; a
run without it reports ABSENT on a compiler that still has the defect.

## What is and is not established

**Established.** The crash is in Greedy's live-range splitting, at the rematerialization check,
at every `-O` level that selects Greedy; the basic and fast allocators compile the same input.

**Not established: the cause.** `d5b5f11cae8f` does not contain C-32's rematerializable
inttoptr bridge (`46c53b7b6ae2`) and crashes the same way, so that change is not required for
it; nothing more than that is known about a link to C-32. `rematWillIncreaseRestriction`
(`llvm/lib/CodeGen/SplitKit.cpp:591`, read at `d030df93d4a4`) dereferences the def's operand 0,
its register-class constraint and `getLargestLegalSuperClass`; which of them is null here has
not been looked at.

**Workaround for the port.** Compile `Python/compile.c` alone at `-O0`, or with
`-mllvm -regalloc=basic`; both were measured to compile it. Neither is applied by the survey,
which reports the object as failed.
