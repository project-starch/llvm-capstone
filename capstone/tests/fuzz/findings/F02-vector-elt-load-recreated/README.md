# F02: variable-index extractelement asserts in DAGCombiner at -O2 · FIXED 2026-09-05

> **FIXED 2026-09-05.** Root cause: the element load of the vector stack temporary carries a pointer info with address space 200 and no value (`MachinePointerInfo::getUnknownStack`); when the index folds to a constant the pointer becomes a bare FrameIndex and `SelectionDAG::getLoad` re-infers a fixed-stack pointer info whose address space came from `TargetMachine::getAddressSpaceForPseudoSourceKind` — 0 by default. The CSE key includes that address space, so DAGCombiner's alignment refinement got a NEW node where it asserts it gets the same one back. Fix, two halves: (1) `CapstoneTargetMachine::getAddressSpaceForPseudoSourceKind` returns the alloca address space for Stack/FixedStack/ConstantPool/JumpTable/GOT (the AMDGPU precedent), so a fixed-stack pointer info carries address space 200; (2) `InferPointerInfo` in `SelectionDAG.cpp` gives a value-less, address-space-0 pointer info on a capability pointer the capability address space BEFORE inferring, so a load created with `MachinePointerInfo()` keeps the same CSE key after its pointer folds to a frame index — the hook alone moved the mismatch to that case and `vararg.ll` asserted the other way round. Pinned by `llvm/test/CodeGen/Capstone/fuzz-f02-f03-vector-elt-stack-temp.ll`, which was red on the unfixed llc (the assert) and is green now; Capstone lit green.


**Found by llvm-stress on 2026-09-04 (seed 32, -O2 only), the first run on the compiler with
the C-39 fix.** Before that fix every seed died earlier, in getVectorSubVecPointer; this is the
next thing on the same path.

Signature (`signature.txt`): `NewLoad.getNode() == N` -- DAGCombiner rebuilt the load of the
vector's stack temporary (alignment inference) and the CSE map handed back a DIFFERENT
existing node, which the combiner asserts cannot happen.

`reduced.ll` (9 lines, `capstone/tests/reduce.sh`): a variable-index element access on a
`zeroinitializer` vector. Reproduce:

    llc -mtriple=capstone64 -O2 -o /dev/null reduced.ll

-O0 compiles it. Not reachable from the C corpora (no vector code), so it is filed, listed in
`known-signatures.txt`, and left for the next fix cycle rather than blocking cycle 1. The fix
belongs in shared code (the expansion's pointer info or alignment on the fat-pointer stack
temporary), so the shared-patch manifest is re-baselined with it.
