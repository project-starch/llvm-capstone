# The bench lane's audit, and the four claims of ours it corrected (2026-09-12)

An audit of the fifty commits since the bench lane last looked found four claims this lane had
pushed the same day. Three were overclaims in `ref/fpga-silicon-measurements-for-paper.md`, the
document that feeds the paper. All four are corrected in place; this note records what happened and
the one pattern they share.

## The pattern, because it is the reusable part

**Three of the four are a HEADLINE or a SUMMARY ROW asserting something the BODY beneath it
correctly qualified.** Not one of them was a wrong measurement. In every case the caveat was already
written, one or two paragraphs down, by this lane, on the same day — and the summary overrode it.

* the bridge: body said "on a *different domain build*"; heading said "THE BRIDGE PAIR HOLDS"
* the Sublet figure: body said "each at its own working geometry, not the discipline in isolation";
  heading said "The discipline costs 9.6 %"
* the allocator claim: body gave the two numbers; summary turned 0.17 pp into "does not depend"

The lesson is not "write caveats". The caveats were there. It is that **the summary is what gets
read and quoted, and a caveat in the body does not travel with it.** A claim in a heading, a table
row or a state-doc bullet has to be independently defensible, because that is the form in which it
will be cited.

## What each correction was

**C1 — "THE BRIDGE PAIR HOLDS" retracted.** sw59 was not the bridge arm. The agreed protocol was
§7k's images unchanged, one post-flash boot, ratio against 1.220 inside the 0.171 pp cross-boot
band. sw59 ran **different images on both arms** — `2f4e6b73b85b569e` (cell ④) and
`f4cf7caed144d952` (cell ①), against sw56's `49994ed31852` and `072595ff0866`. So the 0.76 pp gap is
a build difference, not a band failure, and the §7f–§7k corpus is *not* in question. But no re-tie
was delivered and the bridge is still owed.

**The audit could not settle this from the record, which is the more serious finding.** §4g.2 cited
its arms by LABEL only — the exact breach of the cite-by-hash rule R-29 exists to prevent — so
"the bridge ran and failed its band" and "this was never the bridge" were indistinguishable in the
document. They differ enormously: the first puts the corpus in doubt. The hashes are now in §4g.2,
and §7k's own images (which appeared nowhere in the doc) are now in §7k.

**C2 — "allocator-independent" softened to "indistinguishable at this precision".** 1.2124 vs
1.2107 is 0.17 pp against a 0.171 pp band, i.e. 0.99× the band: it cannot resolve a difference in
either direction. The deterministic QEMU pair (0.25 pp, 1.3e-7) *can*, and shows lookaside costing
marginally less — so the supportable summary across both platforms is "barely depends".

**C3 — "discipline" corrected to "configuration".** The QEMU figure had already been retracted as a
discipline cost because the arms differ in heap geometry (910,008 vs 2,097,152). The silicon pair
carries the identical mismatch. The 5.5× gap inherits the caveat: suggestive of the O(bytes) reclaim
rather than a measurement of it.

**C4 — R-33's gating sentence scoped down.** It claimed the representability fix must land before
the size-100 arena. The artifacts actually built are 128 MiB = 2²⁷ plus two 64 KiB regions — all
powers of two, all widening by zero, since a power of two is representable at any granule. The
claim came from reading the docs' "measured 120 MiB" as the artifact size when the delivered build
is 128 MiB. The hazard is confined to a re-measured, non-round arena.

**C5 — not an error, a missing cross-reference.** §4g.2's "6.0 % apart, not explained here" is
explained one series up: §7i derives that the board baseline's instret carries the timer tick,
predicting 6.06 % where sw59 reads 6.03 %. They agree to 0.02 pp.

## What it cost, and what it did not

No measurement was wrong and no number changed. What was wrong was what the numbers were said to
mean. The cost was a day of other lanes reasoning from an overclaimed bridge and a gating constraint
that did not apply — including a rehearsal held, briefly, behind a fix its artifacts did not need.
