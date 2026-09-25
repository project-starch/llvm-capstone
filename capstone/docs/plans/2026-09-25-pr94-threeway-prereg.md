# Pre-registration: the #94 three-way compile, written BEFORE the run

#94 (`compiler/recover-provenance`) rewrites `inttoptr(X)` into `gep(p, X - ptrtoint(p))` where X
provably comes from ONE capability in the same function. The board asked what it does to C-32 at
`setupLookaside`, and whether it makes D-prime unnecessary or conflicts with it. This records the
predicted readings before the compile, so the answer cannot be fitted afterwards.

## The three arms
All on the Sublet-patched SQLite amalgamation, -O2, the port's own harvested flags:

  A  current        sublet patch as on dev, compiler without #94
  B  current + #94  same source, compiler WITH #94
  C  D-prime + #94  D-prime patch, compiler WITH #94

## Predictions

**A (already measured, the control).** On the linked speedtest1 image: 4 mixed / 0 INT-ONLY, two of
them at `setupLookaside` (+0x23d58, +0x23e58); the other two at `relocatePage` and
`whereLoopAddBtreeIndex`. On the standalone TU: 1 INT-ONLY + 1 mixed at `setupLookaside`.

**B — #94 SHOULD act here, and this is the arm I am least sure of.** The current code does
`uptr a = (uptr)pStart;` where `pStart` is a capability, then `db->lookaside.pMiddle = (void*)a;`.
That is a `ptrtoint` of a capability feeding an `inttoptr`, in one function, with the source
dominating — #94's stated trigger. So I predict the `inttoptr`s become GEPs on `pStart`.

What that does to the C-32 shape is NOT predictable from the pass's rules, and I am recording that
rather than guessing: the rewrite creates NEW USES of `pStart`, and C-32's mechanism is
`MachineSinking::PerformSinkAndFold`, which is all-or-nothing per def. More uses can suppress the
sink-fold or leave it unchanged. **Both outcomes are informative and neither refutes #94.**
  - if the setupLookaside sites DISAPPEAR in B, #94 fixes C-32 at this site by itself and the
    question of whether D-prime is still wanted becomes a real one;
  - if they REMAIN, #94 does not fix C-32 here and D-prime is doing work #94 cannot.

**C — I predict NO CHANGE from the D-prime-alone measurement: 0 sites at `setupLookaside`.**
Reason, from #94's own leaf rule: after D-prime, `pStart` is a `uptr` assigned from
`sqlite3MallocLinear(...)`, a CALL. #94 classifies a call result as "an integer and no source", and
the only other store to that local is the constant 0. Union of sources is empty, so #94 has nothing
to recover provenance from and must leave the casts alone. If C differs from D-prime-alone, my
reading of the leaf rule is wrong and I want to know.

## Falsifiers
- C showing any `setupLookaside` site => my claim that #94 cannot touch D-prime's casts is wrong.
- B and C both showing zero AND identical counters => D-prime may be redundant with #94; that is a
  real finding and should not be argued away.
- Any arm where `relocatePage`/`whereLoopAddBtreeIndex` also vanish => suspect the instrument, not
  the compiler. Those two are untouched by both changes and are the control that shows the scan
  still fires.
- A new site anywhere in any arm => #94 has a cost I have not accounted for.

## What this cannot settle
The scanner is static and says so: "necessary, not sufficient -- taggedness is dynamic." None of the
three arms is silicon, and none of them is a runtime measurement. Behaviour equivalence has to come
from the Sublet counters and the speedtest1 completion marker, as it did for D-prime
(split=5481 mrev=37884 delin=32575 revoke=37884 init=5309, identical before and after).
