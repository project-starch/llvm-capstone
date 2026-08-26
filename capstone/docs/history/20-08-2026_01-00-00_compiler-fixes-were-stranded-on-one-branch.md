# Three compiler fixes were stranded on one branch, and everything else was measured without them

**2026-08-20.** Not a bug in the compiler. A bug in where the compiler fixes lived.

## What was found

`musl-capstone-port` carried three fixes to the shared toolchain, each committed
there because that is where the failure was found:

| issue | file | what it fixes |
|---|---|---|
| C-21 | `clang/lib/AST/ExprConstant.cpp` | a negative integer constant cast to a pointer aborted the constant evaluator |
| C-25 | `llvm/lib/Target/Capstone/CapstoneISelLowering.cpp` | a pointer difference required both operands to be tagged, so `NULL - NULL` faulted |
| C-26 | `clang/lib/CodeGen/ABIInfoImpl.cpp` | `va_arg` of a by-reference struct fetched the reference with an 8-byte `ld` |

Two of the three were committed MIXED, compiler change and port change in one
commit. So they could not be cherry-picked without dragging the port with them,
and nobody looking at `capstone-llvm-fixes` would see they existed.

## What it cost

Every other branch built against a compiler that did not have them. Concretely,
in this session before the isolation:

- the musl survey was re-measured and a baseline edit prepared against a
  compiler missing C-21 — the numbers were wrong and the edit was reverted;
- the survey's negative control was investigated as if it had regressed, when
  in fact the branch simply had an older compiler;
- `jerryscript` and `micropython` had never had C-25 or C-26 at all, so any
  conclusion drawn on them about pointer differences or varargs rested on a
  toolchain two fixes behind the one that had already fixed them.

## What was done

All three now sit on `capstone-llvm-fixes`, each as its own commit, with the
port halves left behind on `musl-capstone-port` and their subjects reworded to
match what they actually contain (they were still titled "Fix C-21" / "Fix C-25"
while containing no fix). The stack was rebased so every leaf inherits them:

```
capstone-bootstrap
  capstone-llvm-fixes    C-21 . C-25 . C-26 . the ptr-diff lowerADD fix
    capstone-infra
      jerryscript . micropython . musl-capstone-port
```

Verified as a content diff rather than by inspection: each rebased branch
differs from its pre-rebase backup by exactly the seven files of those three
fixes and nothing else.

## The check that was missing

C-21 had no test. It was found by a musl build failing in a way that read as a
missing archive member, and the stack trace was in clang's frontend. A frontend
fix with no frontend test is exactly what lets it be re-broken silently, and
what let it sit on a side branch for weeks without anyone noticing the others
lacked it.

`clang/test/CodeGen/capstone-negative-int-to-pointer.c` now covers it, and it
was negative-tested the cheap way: run against the not-yet-rebuilt binary, it
fails on the `getActiveBits() <= 64` assertion; against the rebuilt one it
passes.

## Rule of thumb this leaves

A fix to `llvm/` or `clang/` committed on a workload branch is not "committed",
it is hidden. The workload branch is where the failure is FOUND; the compiler
branch is where the fix LIVES. When both change together, that is two commits,
not one.
