# C-23 — an `__int128` whose high half carries information is computed on the low 64 bits only

**SILENTLY WRONG CODE, not a crash.** Same class as C-22 but without any of its
machinery: no `select`, no `sign_extend_inreg`, no DAGCombine rewrite. Two lines of
ordinary C are enough. C-22 should be read as one *instance* of this; C-23 is the
rule.

`bash run.sh` — exit 1 = present, 0 = absent, 2 = the detector itself failed.

## Reproducer

    void store_assembled(u128 *out, u64 a, u64 b) {
      *out = (u128)a | ((u128)b << 64);
    }

Arguments are `out`=`a0`, `a`=`a1`, `b`=`a2` (confirmed by the positive control in
`src/halves.c`, which returns `b` and does emit `movc a0, a2`). The whole body:

        cincoffsetimm sp, sp, -32
        stc  ra, 16(sp)
        stc  s0, 0(sp)
        movc s0, sp
        cincoffsetimm s0, s0, 32
        mv   a1, a1
        stc  a1, 0(a0)
        ...

`a2` never appears. The store writes `a` alone; `b` is discarded. `eq_full` in the
same file compares only the low halves, so `x == ((u128)a | ((u128)b << 64))` returns
true for every `b`.

## Why

On Capstone PureCap `i128` is the *capability carrier*, not a 128-bit integer: the
128 bits are {tag, permissions, bounds, cursor}, and only the low XLen bits are
meaningful as an integer. There is no register class that holds a plain 128-bit
integer. So the `lowerScalarI128*` helpers in `CapstoneISelLowering.cpp` all follow
the same strategy — truncate both operands to XLen, compute there, re-extend:

    // lowerScalarI128LogicalOnCapability
    SDValue Logical = DAG.getNode(Op.getOpcode(), DL, XLenVT, ToXLen(LHS), ToXLen(RHS));
    return DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i128, Logical);

That is exact whenever the high half is zero or redundant, and wrong otherwise. It is
never diagnosed.

## Relationship to the two loud failures

The same design also produces two *loud* errors, both of which block a `-O1`/`-Os`
build of JerryScript:

| shape | what happens |
|---|---|
| `~(u128)x`, i.e. `xor(zext(x), i128 -1)` | `Cannot select` — the result's high half must be all ones regardless of `x`, which the truncate-and-re-extend strategy cannot spell |
| `(u128)v >> 64` | `report_fatal_error` "cannot lower a 128-bit right shift by >= XLen" (`CapstoneISelLowering.cpp:8377`) |

These are the shapes where the low-64 approximation cannot even be *written down*, so
the backend refuses instead of lying. **The loud cases are the safe ones.** Every
shape that compiles today is a shape where the approximation happened to be
expressible — not one where it is known to be right.

## What a fix requires

Not a lowering pattern per shape. It needs `i128`-the-integer separated from
`i128`-the-capability-carrier, so that integer `i128` can be expanded to a pair of
XLen GPRs the way every other RISC-V target does it. Until then, any new `i128` shape
that appears in ported C is a coin flip between a build error and a wrong answer.

## Provenance

Found 19-08-2026 while estimating the compiler work needed to build JerryScript at
`-Os` (the `-O0` image is 2,965,680 bytes against a 1,376,256-byte domain ceiling).
The first probe of this — checking which shapes *compile* — reported `(u)a|((u)b<<64)`
as working. It compiles; it is also wrong. Shapes were then re-tested through memory
and through a full compare, which is what exposed the dropped half. A compile check is
not a correctness check.
