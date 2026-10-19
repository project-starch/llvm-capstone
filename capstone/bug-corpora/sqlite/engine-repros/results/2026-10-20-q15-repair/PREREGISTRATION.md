# memsys5 Q-15 repair, sqlite engine corpus, virtual-nested-pools: pre-registered 2026-10-20 before runs A-D

The repair is lane q15-memsys5 7b31f34ae378:
- `sublet-3220000-memsys5.patch` rounds each memsys5 block child and the lookaside generation up to their bounds
  granule (tail pad only, inside the block) and counts the pad;
- `repro322_common.h` prints Q15_PAD after a completed case.

Why: the arm read NO-READING control-failed 33/33 on the compact profile (t3-pred-sq1), with cause 29 in
capstone_cap_derive.

## Images

All built with build-virtual.py, SDK t3-freeze-nb (effective -fno-builtin), amalgamation 7fc92e19, --ext-src
sqlite-src-3220000, -O0, controls by --observe. 33 cases and 2 controls each.
- **repaired** (m5-builds3/on-q15): this lane.
- **unrepaired** (m5-builds3/on-dev): dev 2ca4dab92724. It carries the v6 commit's aligned heap, which the recorded
  sq1 images predate.
- **OFF arm (memsys5): byte-identical** between this lane and dev, 33 of 33. A first version of the report macro
  added 24 bytes of .text to every OFF image; this control caught it, and the macro is now empty.

## The trace pre-pass, already run (repaired images, exact-bounds kit qkit7-eb, CAPSTONE_Q15_TRACE=1)

Every padded derive, printed as it happens, so a faulting case still shows its pads:
- **29 cases:** only 64-byte pads, two per connection. These are the 120,000-byte lookaside buffer's block and its
  generation (rounded to 120,064). No slot capability reaches either: slots are derived at sz = 1,200, below 4 KiB,
  so they are exact.
- **None:** cases 18 and 22 and both controls.
- **Case 05:** one 104-byte block pad. Its defect is temporal, and revocation does not depend on length.
- **Case 29** (fz02, the get4byte page overread): one 48-byte block pad. It is the only spatial case with a block pad
  of its own: the one pad-masking candidate.

The same pass, not judged: the 27 cases recorded CAUGHT for this arm faulted, the 6 recorded MISSED did not, and
case 29 faulted (cause 28). So prediction B for case 29 below is informed by that pass, not blind.

## Runs and predictions (each one session through tier3-appvm-arm.sh, sqlite engine-repros/shared/run-virtual.py)

- **A: unrepaired on qkit7-eb (exact bounds).** The reference. Controls fire (uaf-mem5 and bounds-mem5 fault). Every
  case reads its case.json virtual-nested-pools verdict: CAUGHT 27, MISSED 6.
- **B: repaired on qkit7-eb.** Equal to A cell by cell (verdict and fault function).
  - Case 29 is predicted CAUGHT. Had its overread landed in its 48-byte pad, it would read MISSED, a true MISSED
    reported as pad-masked; the pre-pass's fault says it does not.
  - Every MISSED case prints Q15_PAD.
- **C: repaired on qkit7 (compact).** Equal to B cell by cell. 0 cause 29 in capstone_cap_derive on the console.
  Controls fire.
- **D: unrepaired on qkit7 (compact), the planted negative.** NO-READING control-failed on all 33, with cause 29 in
  capstone_cap_derive in the controls, as sq1 read. If D does not read 29, C's 0 says nothing about the repair.
