# C-21 — a select of two `__int128` constants cannot be selected

**A COMPILER bug, not a silicon defect.** Siblings: C-19 (cap-init block split) and
C-20 (`__builtin_ctz`) are in the neighbouring folders.

## Reproducer

    unsigned __int128 g(int c) { return c ? (unsigned __int128) -4 : (unsigned __int128) 0; }

    fatal error: error in backend: Cannot select:
      t32: i128 = CapstoneISD::SELECT_CC t40, Constant:i64<0>, seteq:ch,
                  Constant:i128<-4>, Constant:i128<0>

`./run.sh` reproduces at -O0, -O1 and -Os and prints a verdict.

## Why it matters, and why it was not seen before

On capstone64 `i128` is a legal type because it carries a capability, so an ordinary
ternary on `__int128` becomes a 128-bit select. The backend has a custom path for
selecting between two capabilities, added for the case where the COMPARE operand is a
constant. This is the other half: both **result arms** are constants, and nothing
lowers it.

Found while sizing the JerryScript port. Its `-O0` build does not happen to generate
this shape; `-Os` does, in `ecma_op_object_find_own`. That is worth stating plainly
because it means the shape is reachable from ordinary C and the optimisation level
only decides whether a given program hits it -- the minimal reproducer above fails at
`-O0` too.

## What it blocks

The JerryScript domain image is 2,965,680 bytes at `-O0`, against a hard ceiling of
1,376,256 (see `../../runtime-qemu/silicon-ladder/domdata-budget.py`). Lowering the
optimisation level is the obvious lever and this is what stands in the way: `-O1`,
`-Os` and `-Oz` all stop here.

So for that port the two blockers are coupled. It is not enough to note that `-Os`
would fit; `-Os` does not compile.
