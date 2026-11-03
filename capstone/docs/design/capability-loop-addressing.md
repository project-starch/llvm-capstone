# Capability loop address reduction

The compiler preserves capability provenance while reducing repeated address
arithmetic. It does not convert non-integral pointers to integers or create
far neighbor capabilities in a loop preheader.

## Cursor preservation in IR

A capability can lose its tag at either the representable-window boundary or
the machine's address-policy boundary. Returning to a valid address does not
restore the tag. Offset rewrites must preserve the value of each surviving
cursor, remove intermediate cursors, or reuse an original accessed cursor.
Distance from an access alone is insufficient: even `p - 1` can be refused
under phys56 when `p` is zero.

InstCombine keeps multi-index capability GEPs whole and does not move constants
across variable GEPs. Adjacent merging and cancellation that remove a cursor
remain legal. LICM does not swap capability GEP offsets to hoist one of them.
SCEV expansion uses an integer offset recurrence plus a complete-offset GEP
for non-integral pointers, including in non-canonical mode. This covers both
IndVarSimplify exit values and invariant IV users without a pointer PHI that
propagates a refused cursor from skipped iterations. It also keeps pointer
sums whole instead of splitting their offsets by loop. These restrictions
apply to all non-integral address spaces in this fork, not just AS200. CodeGenPrepare
does not create common capability bases for large offsets, and the optional
separate-offset pass leaves capability GEPs intact.

The generic LSR solver does not reduce non-integral pointer values. Its
post-increment normalization can initialize a recurrence at an unaccessed
`p - step`, and it can step a nonnull pointer on skipped iterations. The
integer offsets still participate in ordinary LSR; the specialized reduction
below is the path for introducing capability recurrences.

## Integer offset sharing

`performCIncOffsetCombine` distributes a scale over an integer index plus a
constant. For large displacements, the result remains one capability
increment with the complete integer byte offset. This shares the scale and
exposes invariant integer constants to machine code motion without creating
an intermediate capability cursor. The existing small-displacement rewrite
continues to use the representable-window restriction.

## Accessed cursor recurrence

After the ordinary LSR solver, `reduceCapabilityIndexGroups` recognizes groups
that IVUsers leaves as integer-index uses. Its scope is
an innermost simplified loop with a native 64-bit counter, zero start, positive
constant increment, and signed and unsigned no-wrap. The base must be loop
invariant, dominate the preheader, and use a non-integral pointer with a
64-bit index. Groups retain the exact base and source element type.

A backwards all-successors analysis proves that every iteration accesses the
same in-bounds current-element GEP. Direct loads and stores establish the
anchor. Uncovered exits, cycles, and instructions that might not transfer
control prevent the proof. Identical accesses on both arms of a diamond can
therefore establish an anchor even when neither access dominates the latch.

The capability PHI starts with the original base. Its byte increment must
satisfy `isCapabilityStepInWindow`, including the final unused increment.
Every increment consumed by another iteration reaches that iteration's
accessed anchor. A refused final increment is unused; no live-out value may
be reconstructed from it.
Residual neighbor GEPs stay at their original instruction sites and preserve
the checked address. In particular, the transformation retains a narrowed
base produced by `shrink`; it never substitutes an earlier, wider base.

## Selectable integer fallback

`-mllvm -lsr-capability-byte-iv` selects a shared integer byte counter instead
of the capability recurrence. Equal-sized groups for the same counter share
one PHI across their different bases. Each original GEP becomes a byte GEP
at its original site, using that counter plus the scaled residual index.
The added recurrence is ordinary modular integer arithmetic. Conditional
accesses need no common capability anchor, and the integer step need not fit
the capability window.

The option is hidden and disabled by default. Automatic selection based on
register pressure remains unresolved: the byte form still needs the large
invariant offsets, and replacing the cursor alone does not prove a pressure
improvement. Compare complete validated profiles before introducing a policy.

## Regression coverage

The focused tests are:

- `llvm/test/CodeGen/Capstone/cincoffset-scaled-large-offset.ll`
- `llvm/test/CodeGen/Capstone/lsr-capability-diamond-iv.ll`
- `llvm/test/CodeGen/Capstone/lsr-capability-byte-iv.ll`
- `llvm/test/Transforms/LoopStrengthReduce/Capstone/capability-pointer-iv.ll`
- `llvm/test/Transforms/LoopStrengthReduce/Capstone/cursor-boundaries.ll`
- `llvm/test/Transforms/InstCombine/capstone-gep-cursors.ll`
- `llvm/test/Transforms/LICM/capstone-gep-reassociate.ll`
- `llvm/test/Transforms/IndVarSimplify/capstone-exit-cursor.ll`
- `llvm/test/Transforms/CodeGenPrepare/capstone-gep-cursors.ll`
- `llvm/test/Transforms/SeparateConstOffsetFromGEP/capstone-gep-cursors.ll`

They cover positive and negative large displacements, narrow indices, skipped
accesses, early exit calls, mismatched anchors, oversized steps, wrapping
counters, preservation of `shrink`, and sharing the byte recurrence across
conditional accesses to different bases. Run them with the Capstone build
and machine instruction verification after sourcing
`capstone/tests/capstone-test-env.sh`.
