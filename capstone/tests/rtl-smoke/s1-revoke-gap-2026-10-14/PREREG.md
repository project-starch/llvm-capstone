# s1-revoke-gap: does a REVOKE pay for an MREV that retired long before it? (pre-registered 2026-10-14)

## Question

capstone-sublet step 6 (capstone-ariane d93a8d66b) reorders MREV (check, deepen-write, handle). In RTL simulation the
first MREV of a program then retires 8 cycles sooner and the timed REVOKE after it commits at the SAME absolute cycle
as before (r12-s1-n8: cycle 4284 on both trees), so the revoke's mcycle bracket reads +8 at every N while the
program's total time is unchanged. That REVOKE was ~440 cycles behind the last MREV with only LCCs between. What pins
the revoke's completion is unresolved: a drain of MREV's last node write that later instructions hide, or an artefact
of the simulation memory model. The board's R1 release-cost harness (`sublet/r1/r1_slots_pools.c`, timed_give_to)
brackets a REVOKE whose handle was minted at fixture time: `ldc handle; csrr mcycle; REVOKE; csrr mcycle`. Its
published numbers (76 cycles at 2 nodes, 22.7 per node) either hold on the Sublet bitstream or their intercept moves
by 8. One binary, run on both bitstreams, decides.

## The pair

`build.sh` wraps capstone-ariane `verif/tests/custom/capstone/s1-revoke-gap.S` (copied into `tests/`, checked equal)
with `sup-bare-2026-10-03-ladder`'s bare M-mode wrapper; legacy instructions only (mrev, revoke, stc, ldc, lcc, and
CAPCREATE for the park capability), so the FP and the Sublet bitstream run the same bytes.

| image | GAP | sha256 | board_rec |
|---|---|---|---|
| `images/s1gap-close.bin` | 0 | `b351bc50610e61bc545946eb79de027cbaa4d8251d657e2c6cbf21ef65894422` | 0x80003080 |
| `images/s1gap-far.bin` | 1000 | `5a35240633e85f2c970dde1af0c16dc9585e4fbf36c2938e2f20eb877f659e9c` | 0x80003080 |

The two images differ in the four GAP immediates, the final GAP reading and the banner character only (the board
lane's `cmp -l` of the first build, 2026-10-14; the final build changes only the node handling below, identically in
both). `inc/` is a copy of the ladder's wrapper with one addition: with `-DBOARD_REC_NODE=<capability register>`
every capability the record macros create is attached (capnode) to that register's node, because the test revokes
the base's node that a plain capcreate would carry; the test keeps the never-revoked handle M0 in x18 for it.

Shape: capenter gives the linear base; four handles are minted on it, outermost first, and parked in memory through a
NONLIN store capability (as R1 parks its handle). Then, innermost first, four repeats of [GAP iterations of a plain
ALU loop bracketed by mcycle; ldc the handle; csrr; revoke; csrr]. Revoking the innermost kills the base and each later
revoke walks the corpses below it, so the four costs differ by index and are compared per index. Only k = 1 follows an
MREV; k = 2..4 follow the previous REVOKE and serve as same-binary repeat controls.

Readings, in order: for k = 1..4 the gap cycles, the revoke cycles, the revoke cause (0); then the base's lcc validity
after revoke 1 (0), total traps (0), GAP, 4. The wrapper prints them as `SV` lines after `SR <count>`; the count is 16.

## Outcomes, written before any board run

- **A**: the revoke cycles are identical across the step-5 and step-6 trees at every k, on both arms. The pin seen in
  r12-s1 is MREV-adjacent in a way this shape does not reproduce, or a model artefact. Prediction for silicon: the two
  bitstreams read the same on both arms; R1's 76 / 22.7 hold.
- **B**: k = 1 differs by 8 between the trees on GAP 0 and not on GAP 1000. The pin is a drain shorter than ~2,000
  cycles. Silicon: s1gap-far equal on both bitstreams, s1gap-close k = 1 +8 on Sublet; R1's bracket (fixture-time
  MREV) holds.
- **C**: k = 1 differs by 8 on both arms. Not a drain of MREV's write; whatever completes the revoke completes 8
  cycles later relative to the program on step 6. Silicon: both arms +8 at k = 1 on Sublet; R1's intercept moves by 8,
  its slope does not.

A reading that fits none of these is a finding, not noise; the first thing read is then the per-index pattern on
k = 2..4.

## Simulation readings (Verilator, S12_MEM_DELAY=12, the pinned container), filled before the board run

Readings in order: k=1 (gap, revoke, cause), k=2, k=3, k=4, base validity, traps, GAP, 4. Hex as the trace prints them.

| tree | arm | k=1 | k=2 | k=3 | k=4 | tail | cycles to tohost |
|---|---|---|---|---|---|---|---|
| FP 4a4261bb4 | close (GAP 0) | 0x8 0x17d 0x0 | 0xb5 0x31 0x0 | 0x88 0x31 0x0 | 0x88 0x31 0x0 | 0x0 0x0 0x0 0x4 | 4,707 |
| FP 4a4261bb4 | far (GAP 1000) | 0xfa6 0xe6 0x0 | 0xfd1 0x31 0x0 | 0xffe 0x31 0x0 | 0xffe 0x31 0x0 | 0x0 0x0 0x3e8 0x4 | 20,307 |
| step 5 464abe2ce | close | 0x8 0xe6 0x0 | 0x83 0x31 0x0 | 0x88 0x31 0x0 | 0x88 0x31 0x0 | 0x0 0x0 0x0 0x4 | 4,812 |
| step 5 464abe2ce | far | 0xfa6 0xe6 0x0 | 0xfd1 0x31 0x0 | 0xffe 0x31 0x0 | 0xffe 0x31 0x0 | 0x0 0x0 0x3e8 0x4 | 20,435 |
| Sublet tip d93a8d66b | close | 0x8 0xe6 0x0 | 0x83 0x31 0x0 | 0x88 0x31 0x0 | 0x88 0x31 0x0 | 0x0 0x0 0x0 0x4 | 4,812 |
| Sublet tip d93a8d66b | far | 0xfa6 0xe6 0x0 | 0xfd1 0x31 0x0 | 0xffe 0x31 0x0 | 0xffe 0x31 0x0 | 0x0 0x0 0x3e8 0x4 | 20,427 |

Read before the board run:

- **Step 5 vs the Sublet tip: identical at every index on both arms.** Outcome **A** for the question this pair was
  built for: step 6's 8-cycle move does not show in R1's shape, nor in the close shape with the handle reloaded by
  ldc a few instructions behind the last mrev. (r12-s1's +8 has the handle in a register, eight or more mrevs, and
  the revoke ~440 cycles later; this pair does not reproduce that distance and does not claim to explain it.)
- **The FP tree differs from both Sublet trees at k = 1 of the close arm only**: revoke 0x17d (381) against 0xe6
  (230), and the next repeat's gap bracket 0xb5 (181) against 0x83 (131); the far arm and k = 2..4 are identical.
  A revoke issued a few instructions behind an mrev waits for the rev-node unit to finish the mrev's writes; on the
  FP tree the mrev retires before that drain ends (no deepening passes), on the Sublet trees the two deepening walks
  inside mrev absorb it. The whole program is still 105 cycles shorter on FP (4,707 against 4,812): the cost moved,
  it was not added. Same mechanism class as the s1-n8 observation, seen from the other side.
- The gap bracket at k = 2..4 of the close arm (0x88 = 136 with GAP 0 against 0x8 at k = 1) shows a revoke's walk
  continuing after the instruction retires: the next csrr waits on it. The revoke bracket (0x31 = 49) therefore
  measures the issue-to-retire part, not the walk.

**Silicon predictions, the same image on both bitstreams** (patterns, not cycle values: the simulation's memory
model is S12_MEM_DELAY=12, silicon's latency differs): far arm equal on both at every k; close arm k = 2..4 equal;
close arm k = 1 LOWER on Sublet than on FP by roughly the FP mrev's drain, never higher. A Sublet reading higher than
FP's at any index is a finding against outcome A and is read first. For R1: its bracket is the far shape, so the
published 76 / 22.7 are predicted to hold on the Sublet bitstream.


## On the board

Exploratory arms in the FP chain's bare set, after the controls and outside the acceptance verdict (the board lane's
run-capstl.sh `s1gap` set), then again on the Sublet boot. The verdict is the per-index difference between the two
bitstreams' readings of the SAME image, graded against the outcomes above.
