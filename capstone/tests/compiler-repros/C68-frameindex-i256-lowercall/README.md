# C-68 — LowerCall asserts on a split scalar integer wider than 128 bits

**Status: OPEN, pre-existing, not reachable from C today.** Registry: C-68 (`docs/ref/ISSUES.md`).

## Reproduce

    llc -mtriple=capstone64 -O2 -o /dev/null repro.ll

    Assertion `N1.getValueType() == N2.getValueType() && N1.getValueType() == VT &&
               "Binary operator types must match!"' failed.
    #12 CapstoneTargetLowering::LowerCall(...) CapstoneISelLowering.cpp

Threshold is sharp: `i128` compiles, `i129` and wider assert.

## Cause

In `LowerCall`'s `CCValAssign::Indirect` path:

    SDValue SpillSlot = DAG.CreateStackTemporary(StoredSize, StackAlign);
    ...
    SDValue Address = DAG.getNode(ISD::ADD, DL, PtrVT, SpillSlot, PartOffset);

`CreateStackTemporary` goes through `TargetLowering::getFrameIndexTy`
(`getPointerTy(DL, DL.getAllocaAddrSpace())`), so `SpillSlot` is a **c128**. `PtrVT` in that scope
is `getPointerTy(DAG.getDataLayout())`, i.e. address space 0, which on this target is **i64**. The
two meet in an `ISD::ADD`. This target models capability arithmetic as `ISD::PTRADD` on `MVT::c128`,
not `ADD`. So the node is inconsistent where it is constructed, not merely at selection.

## Three things that decide its priority

1. **PRE-EXISTING.** The pre-C-50 compiler (`b7b31421e9fa`) asserts identically on this IR. The C-50
   fix neither causes nor masks it.
2. **NOT vector-gated.** It arrives through `CapstoneCallingConv.cpp`'s
   `ValVT.isScalarInteger() && (ArgFlags.isSplit() || !PendingLocs.empty())` — no vector anywhere.
   This matters: `fatal-scalable-stack.ll` records that RVV is non-functional on this target, and
   stopping at that prior art would have produced a wrong "unreachable" verdict.
3. **NOT reachable from C.** The only C spelling of a >128-bit scalar integer is `_BitInt(N)`, and
   clang rejects `N > 128` here ("signed _BitInt of bit sizes greater than 128 not supported"). It
   needs hand-written IR, another frontend, or a raise of that cap.

## Why it is not fixed here

The presumable fix is `PTRADD` in the capability type. With no C-reachable case there is nothing to
validate a change to argument lowering against, and an unvalidated change to argument lowering is
how a quiet miscompile gets in rather than kept out. Filed with the reproducer instead.

Sibling sites of the same class, and why they are quiet, are in
`docs/history/28-09-2026_00-00-00_frame-index-pointer-type-sweep.md`.
