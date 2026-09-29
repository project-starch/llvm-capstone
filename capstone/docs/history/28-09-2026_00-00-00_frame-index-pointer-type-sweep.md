# The getPointerTy / frame-index sweep, verified against source  (2026-09-25)

C-50 was ONE instance of a class: a pointer into the frame spelled with the DEFAULT address space,
which on this target is a 64-bit integer, where a stack object is a capability (datalayout A200).
The reference spelling is `TargetLowering::getFrameIndexTy(DL)` =
`getPointerTy(DL, DL.getAllocaAddrSpace())` (`llvm/include/llvm/CodeGen/TargetLowering.h:393-395`).

All line numbers verified by reading the current source, not taken from a report.

## FIXED
- `:24397` LowerCall, byval local copy. Was `getPointerTy(DL)`. This is C-50.

## THE SHARPEST REMAINING SITE, and it is NOT the same shape as C-50
`:24487-24496`, LowerCall's `CCValAssign::Indirect` path:

    SDValue SpillSlot = DAG.CreateStackTemporary(StoredSize, StackAlign);
    ...
    SDValue Address = DAG.getNode(ISD::ADD, DL, PtrVT, SpillSlot, PartOffset);

`CreateStackTemporary` goes through `getFrameIndexTy`, so `SpillSlot` is a **c128**. `PtrVT` at
`:24347` is `getPointerTy(DAG.getDataLayout())`, i.e. **i64**. So this builds an `ISD::ADD` with an
i64 result type over a c128 operand — and this target models capability arithmetic as `ISD::PTRADD`
on `MVT::c128` (`:409`), not `ADD`.

**RESOLVED 2026-09-25: REACHABLE, AND IT ASSERTS.** No longer a reading-based claim.

    ; i129 or wider, in an argument position that forces a split
    declare void @sink(i64,i64,i64,i64,i64,i64,i64, i256) addrspace(200)
    define void @f(i256 %v) addrspace(200) {
      call addrspace(200) void @sink(i64 0,i64 0,i64 0,i64 0,i64 0,i64 0,i64 0, i256 %v)
      ret void }

    llc -mtriple=capstone64 -O2
    Assertion `N1.getValueType() == N2.getValueType() && N1.getValueType() == VT &&
               "Binary operator types must match!"' failed.
    #12 CapstoneTargetLowering::LowerCall(...) CapstoneISelLowering.cpp:24505

(24505 in the fixed worktree = 24496 unmodified; the C-50 hunk adds nine lines above it.)
The stack frame names the exact site predicted from reading, which is the confirmation.

  * NOT vector-gated. It arrives through CapstoneCallingConv.cpp:567 --
    `ValVT.isScalarInteger() && (ArgFlags.isSplit() || !PendingLocs.empty())` -- so RVV being
    non-functional on this target (fatal-scalable-stack.ll) does not protect it.
  * PRE-EXISTING, not caused by the C-50 fix: the pre-fix baseline llc
    (llvm/cmake-build-debug, b7b31421e9fa) aborts identically on the same IR.
  * Threshold is sharp: i128 compiles, i129 and wider assert.
  * NOT REACHABLE FROM C TODAY: the only C spelling is `_BitInt(N)`, and clang rejects
    N > 128 on this target ("signed _BitInt of bit sizes greater than 128 not supported").
    So it needs hand-written IR, another frontend, or a future raise of that cap.

Priority follows from that last point: real, precisely localised, minimal reproducer in hand,
but no C program can currently trigger it. File it; do not rush it.

This is INCONSISTENT BY CONSTRUCTION: unlike C-50, where the frame index was uniformly an integer
and the defect only appeared at selection, here the two types disagree at the point of construction.
It cannot be correct as written. **Reachability UNRESOLVED** — the path needs a split/indirect
argument, and I could not construct one that reaches it. If reached it should assert in `getNode`
rather than silently miscompile, which is the better failure, but that is not verified either.

## SAME SHAPE, formal-argument side
`:24179` `DAG.getNode(ISD::ADD, DL, PtrVT, ArgValue, Offset)` in the incoming-argument indirect
path, with `PtrVT` = i64 from `:24131`. Whether `ArgValue` is a capability there depends on how it
was unpacked; UNRESOLVED.

## LATENT, integer frame index, quiet for a reason
- `:15414/:15417` `lowerEH_DWARF_CFA` — `getPointerTy(DL)` then `getFrameIndex`. Reachable only via
  `__builtin_dwarf_cfa`.
- `:23993/:24002` `unpackFromMemLoc` — explicitly `MVT::getIntegerVT(...getPointerSizeInBits(0))`.
  Produces `LD %fixed-stack.N, 0`, which PEI resolves onto the capability frame register; no add is
  built on it.
- `:24131/:24230` the vararg save loop — `PtrVT` is i64 and the loop uses `getMemBasePlusOffset`,
  yet selection gives `CIncOffsetImm %fixed-stack.N, 16/32/...` and never `ADDI`, because those
  offsets are multiples of the alignment and so never become the `or disjoint` that C-50 turned on.
  **Quiet only by arithmetic accident; one alignment change from live.**
- `:24040` `MVT::i32`, RV32D soft-float only — unreachable on capstone64.

## Why C-50 fired and these do not
C-50's `+8` on a 16-aligned object has its low bits known zero, so `add FI, 8` is rewritten to
`or disjoint`, which selects to the integer `ADDI`. `or` cannot form on a c128. Every quiet site
above is quiet because its offsets never take that rewrite — not because its types are right.

## Recommendation
Do NOT bulk-retype these. Each needs its own reachable reproducer or it is an unverifiable change to
argument lowering. The `:24487` site is the one worth an issue of its own, because it is provably
inconsistent rather than merely latent.
