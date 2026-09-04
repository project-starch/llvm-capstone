# F-01 — a variable-index extract/insert on a wide vector crashes ISel: the index is zero-extended into the capability pointer type

**A COMPILER crash in shared LLVM code, found by llvm-stress on 2026-09-04 on every
seed (120 of 120 runs at -O0 and -O2).** Registry ID **C-39**, allocated by the capstone
session on 2026-09-04 (free in ISSUES.md and in git log --all).

## Reproducer

`reduced.ll` (8 lines, reduced by `capstone/tests/reduce.sh` from llvm-stress seed 1,
240 lines):

    define i64 @autogen_SD1(i32 %L) addrspace(200) {
      %E65 = extractelement <8 x i64> zeroinitializer, i32 %L
      ret i64 %E65
    }

    llc -mtriple=capstone64 -mattr=+m -O0 -o /dev/null reduced.ll
    llc: SelectionDAG.cpp:6606: getNode(...): Assertion `VT.isInteger() &&
         N1.getValueType().isInteger() && "Invalid ZERO_EXTEND!"' failed.

Same at -O2, and for `insertelement` with a variable index.  The lit pin is
`llvm/test/CodeGen/Capstone/fuzz-f01-vector-elt-pointer.ll` (XFAIL until the fix lands).

## Cause

A vector wider than the legal width is split by the type legalizer; a variable index
cannot be split statically, so `SplitVecRes_INSERT_VECTOR_ELT` (and the extract, and
the subvector paths in `LegalizeDAG.cpp`) spill the vector to a stack temporary and
address the element through `TargetLowering::getVectorSubVecPointer`, which starts with

    Index = DAG.getZExtOrTrunc(Index, dl, VecPtr.getValueType());

On this target the stack pointer is a capability, `MVT::c128`, not an integer, so the
zero-extension is refused by `getNode`.  The stack dump goes
`getVectorSubVecPointer <- getVectorElementPointer <- SplitVecRes_INSERT_VECTOR_ELT <-
DAGTypeLegalizer::run`; the only Capstone frame is `CapstoneDAGToDAGISel::runOnMachineFunction`,
i.e. the target's ISel driving generic legalization.

## Reachability from C

A GCC vector extension (`typedef long v8 __attribute__((vector_size(64)))`) subscripted
with a variable, or any front end producing `insertelement`/`extractelement` with a
non-constant index on a vector the target cannot hold in one register.  No project corpus
(CoreMark, BEEBS, RV8, SQLite, mruby, MicroPython) does that, which is why no suite ever
hit it; llvm-stress hits it on every module because every module it emits does.

## Fix

`TargetLowering::getVectorSubVecPointer`: compute the index in an integer type when the
pointer's value type is not one (AS0's pointer type stands in for the index width, as
`SelectionDAG::getMemBasePlusOffset` already does in this fork); `getMemBasePlusOffset`
then applies the integer offset to the capability.  Shared-LLVM patch: to be listed in
the Tier 4.7 manifest.

## Status

Pinned (XFAIL).  Fix written, awaiting the rebuild that the running Tier 2a twins must
not see mid-run, then the lit suite, the fuzz rerun, and a claim-auditor pass before the
commit says "fixed".
