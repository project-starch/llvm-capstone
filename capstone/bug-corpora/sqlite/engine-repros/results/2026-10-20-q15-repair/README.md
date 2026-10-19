# memsys5 Q-15 repair: runs A-D, read against PREREGISTRATION.md

The sqlite engine corpus, arm virtual-nested-pools, 33 cases and 2 controls. Each run is one session through
tier3-appvm-arm.sh with engine-repros/shared/run-virtual.py. `run-<X>/verdicts.jsonl` and `inputs.json` are the
bundles; the consoles stay outside the repo.

| run | images | kit | reading | prediction |
|---|---|---|---|---|
| A | unrepaired (dev) | qkit7-eb (exact bounds) | CAUGHT 27, MISSED 6; controls fault | holds |
| B | repaired | qkit7-eb | equal to A, 33 of 33 cells (verdict, cause, function); case 29 CAUGHT (cause 28, sqlite3Get4byte), not pad-masked; Q15_PAD on the 6 MISSED cases | holds |
| C | repaired | qkit7 (compact) | equal to B, 33 of 33 cells; 0 cause-29 lines on the console; controls fault | holds |
| D | unrepaired (dev) | qkit7 (compact) | 31 NO-READING with cause 29 in capstone_cap_derive; cases 18 and 22 CAUGHT; controls fault | **does not hold as written** |

**D, the planted negative.** I predicted control-failed on all 33 "as sq1 read". That was wrong in form.
- sq1's controls failed because its images predated the v6 commit's aligned heap, so the arena derive itself was
  inexact. These dev images carry that alignment, as the pre-registration's own Images section says.
- So D refuses exactly where an inexact child is derived, and the trace pre-pass had already named those places: its
  31 padded cases are exactly D's 31 cause-29 cases (set equality checked). Cases 18 and 22 and both controls had no
  padded derive, and run normally on both images.
- The negative fires: 31 cause-29 refusals unrepaired, 0 repaired, on the same compact kit.

**Pad-masking:** none. Case 29, the one spatial case with a block pad of its own (48 bytes), reads CAUGHT at the same
function on A, B and C. Q15_PAD across the 6 completed cases: 2 or 3 padded derives each, 64 bytes at most, which are
the lookaside buffer's block and its generation.

The OFF arm (memsys5) is byte-identical to dev's, 33 of 33.
