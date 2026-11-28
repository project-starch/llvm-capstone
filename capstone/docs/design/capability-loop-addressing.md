# Capability loop address reduction

The compiler preserves capability provenance while reducing repeated address
arithmetic. It does not convert non-integral pointers to integers or create
far neighbor capabilities in a loop preheader.

## Cursor preservation in IR

A capability can lose its tag at either the representable-window boundary or
the machine's address-policy boundary. Returning to a valid address does not
restore the tag. Offset rewrites must preserve the value of each surviving
cursor, remove intermediate cursors, reuse an original accessed cursor, or form
a cursor the program did not form when that cursor is used only as an operand of
an address-bits compare (`icmp eq`/`ne`), never as an access operand, an IV seed
or a live-out, and the period guard below holds. Such a cursor may be refused
(its tag cleared); the compare reads the integer address bits, which the
refusal preserves.
Distance from an access alone is insufficient: even `p - 1` can be refused
under phys56 when `p` is zero.

### The global-address exception (2026-11-13)

The project lead relaxed the rule above for one backend fold, and only under
its own flag, `-mllvm -capstone-fold-global-lo` (default off). Under the
image-gp ABI a global's capability is `auipc; addi %pcrel_lo; scc gp`, and the
fold drops the `addi`: the capability is derived from the auipc's partial
address and each access carries the `%pcrel_lo` (`auipc; scc gp; ld
%pcrel_lo(cap)`). That partial cursor lies up to 2 KiB before or after the
accessed address, and the program never formed it. Two conditions make it
safe, and both are required:

- **The cursor never survives.** The fold is made only when the derived
  capability's every use is the base of a load or store with one common
  immediate offset: never the stored value, a copy, a call argument, a
  compare, a `cincoffset`, a `shrink`/`shrinkto` or a debug use, and never when
  anything in the function writes gp (`CapstoneMergeBaseOffset.cpp`,
  `foldIntoCapGlobalMemoryOps`).
- **The cursor stays inside gp's bounds.** gp spans the image from the text
  segment's start. The virtual image's link script (`capstone/runtime/virtual/
  link.ld`) asserts that `.text` opens the image and is at least 2 KiB, and
  pads 2 KiB after `.bss`. Every gp-reachable symbol therefore has 2 KiB of
  gp on both sides, so a partial cursor is inside gp's bounds, which the
  address policy accepts. The rule's own concern, a cursor refused at a
  window or policy boundary, cannot arise.

An image linked without that script gets no such guarantee, so the flag is
for virtual-profile images only. Constant-pool entries reached through gp are
folded under the same conditions.

SelectionDAG lowering sums all indices of a non-integral GEP in the integer
index type before applying one pointer offset. InstCombine keeps multi-index
capability GEPs whole and does not move constants
across variable GEPs. Adjacent merging and cancellation that remove a cursor
remain legal. LICM does not swap capability GEP offsets to hoist one of them.
SCEV expansion uses an integer offset recurrence plus a complete-offset GEP
for non-integral pointers, including in non-canonical mode. This covers both
IndVarSimplify exit values and invariant IV users without a pointer PHI that
propagates a refused cursor from skipped iterations. It also keeps pointer
sums whole instead of splitting their offsets by loop. These restrictions
apply to all non-integral address spaces in this fork, not just AS200. CodeGenPrepare
shares same-sign constant offsets from the offset closest to zero.
The new cursor lies between the base and each original final cursor. It is
placed at a dominating original GEP or at a common dominator from which
every path reaches a group GEP or a load through the exact base. Loads reject
UNINIT and sealed capabilities, as does cursor arithmetic. No store establishes
coverage: ordinary stores at immediate zero, as well as capability stores,
can use an UNINIT base. The earlier claim that ordinary stores establish this
proof was wrong. The coverage walk refuses uncovered exits, cycles and calls
that may not transfer control. Otherwise the existing non-null speculation rule
is required. Negative groups use the mirrored interval from their maximum
offset to zero, with nonpositive residuals. Opposite signs form separate
groups; a target-preferred base outside the applicable interval is ignored.
The optional separate-offset pass leaves capability GEPs intact.

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
an intermediate capability cursor. Small displacements also stay in the
integer offset; the backend cursor rule does not permit a new intermediate
cursor merely because it is nearby.

## Accessed cursor recurrence

After the ordinary LSR solver, `reduceCapabilityIndexGroups` recognizes
single-index GEPs with an affine 64-bit integer index in an innermost
simplified loop. The start can be nonzero or loop invariant, and the constant
step can have either sign. One GEP per base is sufficient. The base must be
loop invariant, dominate the preheader, and use a non-integral pointer with a
64-bit index. Groups retain the exact base and source element type.

A backwards all-successors analysis proves that every iteration accesses the
same in-bounds current-element GEP. Equality is checked on integer index
SCEVs with the exact same pointer base. Direct loads and stores establish the
anchor. Uncovered exits, cycles, and instructions that might not transfer
control prevent the proof. Identical accesses on both arms of a diamond can
therefore establish an anchor even when neither access dominates the latch.

The capability PHI starts at the iteration-zero anchor in the loop preheader.
Every increment consumed by another iteration reaches that iteration's
accessed anchor. The final unused increment may lose its tag; live-out uses
retain the current accessed value rather than that increment. Integer offset
arithmetic is modular at the GEP index width. An extension of a wrapping
narrow counter is not accepted unless SCEV proves it affine at that width.
Residual neighbor GEPs stay at their original instruction sites. In
particular, the transformation retains a narrowed base produced by `shrink`;
it never substitutes an earlier, wider base.

On Capstone64, a single latch exit may compare the post-increment cursor with
a separately formed endpoint instead of retaining an integer exit counter.
The endpoint is a non-inbounds GEP from the original base, used only for address
equality. It never seeds the accessed recurrence or feeds a memory operation.
The final increment and endpoint may lose their tags: comparisons still use
their integer address bits. The accessed-anchor proof remains required.

ScalarEvolution must provide an exact loop-invariant backedge count and a
constant maximum below the address recurrence's period. For a nonzero byte
step S, that period is `2^(64 - ctz(S))`; bounding the backedge count strictly
below it prevents an earlier address from comparing equal to the endpoint.
This is the period guard. Besides the other conditions below, the transform
requires that the constant maximum backedge count M satisfies
`M < 2^(64 - ctz(S))` (in the code, `getActiveBits(M) <= 64 - ctz(S)`). The
loop then compares at most M + 1 post-increment cursors, and none of the first
M can equal the endpoint, so the exit fires on the iteration the integer test
would have fired. An exhaustive check of a 10-bit ring over every step and
every count found no unsound accept and no needless reject of this predicate.
For a byte step of one the period is 2^64 and an unbounded 64-bit count
passes; for a step that is a multiple of eight it does not.
An unrestricted 64-bit counter scaled by eight does not establish this bound.
The integer recurrence's remaining uses must belong exclusively to the exit
test, so computing the endpoint can remove that recurrence. Early exits,
unknown counts and counters with other uses retain their integer tests.

## Selectable integer fallback

`-mllvm -lsr-capability-byte-iv` selects a shared integer byte counter instead
of the capability recurrence. Groups with the same byte start and step share
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
- `llvm/test/CodeGen/Capstone/gep-cursor-sequence.ll`
- `llvm/test/CodeGen/Capstone/gep-complete-offset.ll`
- `llvm/test/CodeGen/Capstone/lsr-capability-diamond-iv.ll`
- `llvm/test/CodeGen/Capstone/lsr-capability-byte-iv.ll`
- `llvm/test/Transforms/LoopStrengthReduce/Capstone/capability-pointer-iv.ll`
- `llvm/test/Transforms/LoopStrengthReduce/Capstone/cursor-boundaries.ll`
- `llvm/test/Transforms/LoopStrengthReduce/Capstone/accessed-affine-cursor.ll`
- `llvm/test/Transforms/LoopStrengthReduce/Capstone/cursor-exit.ll`
- `llvm/test/Transforms/InstCombine/capstone-gep-cursors.ll`
- `llvm/test/Transforms/LICM/capstone-gep-reassociate.ll`
- `llvm/test/Transforms/IndVarSimplify/capstone-exit-cursor.ll`
- `llvm/test/Transforms/CodeGenPrepare/capstone-gep-cursors.ll`
- `llvm/test/Transforms/SeparateConstOffsetFromGEP/capstone-gep-cursors.ll`

They cover positive and negative large displacements, narrow indices, skipped
accesses, early exit calls, mismatched anchors, large and negative steps, narrow wrapping
counters, preservation of `shrink`, and sharing the byte recurrence across
conditional accesses to different bases. Run them with the Capstone build
and machine instruction verification after sourcing
`capstone/tests/capstone-test-env.sh`.
