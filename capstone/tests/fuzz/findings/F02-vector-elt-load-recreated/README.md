# F02: variable-index extractelement asserts in DAGCombiner at -O2

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
