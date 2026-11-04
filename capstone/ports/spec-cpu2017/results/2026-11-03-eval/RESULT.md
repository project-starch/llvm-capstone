# Compiler evaluation on silicon, steps 1 and 2: result (first part)

The boots were pre-registered in PREREG.md: 8666e7a2cdf4, and Addendum 1 at 7cbc8edbae4a.

**Graded so far:** 9 of 12 benchmark-board pairs.
- Board 1 (bitstream 3): lbm, nab, imagick, gcc and xz.
- Board 2 (ObjBounds 603528b8b): lbm, nab, imagick and gcc.
- Still running: mcf on both boards, and xz on board 2. E4b is mcf-only and has no reading yet.

**Grader.** `eval-grade.py`, run from `$CAPSTONE_WORK/spec`. It reads each stage's result.json, prints PENDING for a stage that has
not finished, and exits 1 when any check fails. Its output at the time of writing is below.

```
== board 1
lbm      MapLife  | dev/base: instret 0.8521 (twin 0.8521, E2 ok) cycles 0.9591 E3 holds; in -60.5s out -10.9s E5 ok; E1 ok | mk/dev: instret 0.9996 (twin 0.9996, E2 ok) cycles 0.9986 E4 rep; in -2.0s out +1.9s E5 ok; E1 ok
lbm      Full     | dev/base: instret 0.8521 (twin 0.8521, E2 ok) cycles 0.9590 E3 holds; in -60.6s out -10.7s E5 ok; E1 ok | mk/dev: instret 0.9996 (twin 0.9996, E2 ok) cycles 0.9988 E4 rep; in -1.7s out +1.5s E5 ok; E1 ok
nab      MapLife  | dev/base: instret 0.9256 (twin 0.9256, E2 ok) cycles 0.9525 E3 holds; in -46.1s out -6.8s E5 ok; E1 ok | mk/dev: instret 0.9951 (twin 0.9952, E2 ok) cycles 0.9707 E4 rep; in -27.1s out -5.6s E5 ok; E1 ok
nab      Full     | dev/base: instret 0.9256 (twin 0.9256, E2 ok) cycles 0.9417 E3 holds; in -57.4s out -12.6s E5 ok; E1 ok | mk/dev: instret 0.9951 (twin 0.9952, E2 ok) cycles 0.9685 E4 rep; in -29.2s out -4.0s E5 ok; E1 ok
imagick  MapLife  | dev/base: instret 0.9279 (twin 0.9279, E2 ok) cycles 0.8717 E3 holds; in -2.1s out -0.1s E5 ok; E1 ok | mk/dev: instret 0.9696 (twin 0.9696, E2 ok) cycles 0.8799 E4 holds; in -1.7s out -0.1s E5 ok; E1 ok
imagick  Full     | dev/base: instret 0.9279 (twin 0.9279, E2 ok) cycles 0.8947 E3 holds; in -1.7s out -0.2s E5 ok; E1 ok | mk/dev: instret 0.9696 (twin 0.9696, E2 ok) cycles 0.8773 E4 holds; in -1.7s out -0.2s E5 ok; E1 ok
xz       Full     | dev/base: instret 0.9966 (twin 0.9966, E2 ok) cycles 0.9769 E3 rep; in -131.3s out -1.0s E5 ok; E1 ok | mk/dev: instret 0.9904 (twin 0.9904, E2 ok) cycles 0.9371 E4 rep; in -349.8s out -73.8s E5 ok; E1 ok
mcf      Full     PENDING
gcc      Full     | dev/base: instret 0.9483 (twin 0.9475, E2 ok) cycles 0.9661 E3 holds; in -0.3s out +0.2s E5 ok; E1 ok | mk/dev: instret 0.9681 (twin 0.9678, E2 ok) cycles 0.9174 E4 holds; in -0.7s out -11.4s E5 FAIL; E1 ok
== board 2
lbm      Spatial  | dev/base: instret 0.8521 (twin 0.8521, E2 ok) cycles 0.9630 E3 holds; in -54.3s out -9.9s E5 ok; E1 ok | mk/dev: instret 0.9996 (twin 0.9996, E2 ok) cycles 0.9991 E4 rep; in -1.2s out -2.1s E5 FAIL; E1 ok
nab      Spatial  | dev/base: instret 0.9256 (twin 0.9256, E2 ok) cycles 0.9490 E3 holds; in -47.1s out -9.3s E5 ok; E1 ok | mk/dev: instret 0.9952 (twin 0.9952, E2 ok) cycles 0.9782 E4 rep; in -19.1s out -2.5s E5 ok; E1 ok
imagick  Spatial  | dev/base: instret 0.9279 (twin 0.9279, E2 ok) cycles 0.8794 E3 holds; in -1.9s out -0.1s E5 ok; E1 ok | mk/dev: instret 0.9696 (twin 0.9696, E2 ok) cycles 0.8947 E4 holds; in -1.4s out -0.2s E5 ok; E1 ok
xz       Spatial  PENDING
mcf      Spatial  PENDING
gcc      Spatial  | dev/base: instret 0.9479 (twin 0.9475, E2 ok) cycles 0.9581 E3 holds; in -0.3s out -0.0s E5 ok; E1 ok | mk/dev: instret 0.9678 (twin 0.9678, E2 ok) cycles 0.9296 E4 holds; in -0.5s out -11.4s E5 FAIL; E1 ok
```

In this output, MapLife is the board 1 `capstone-spatial` arm and Full is `capstone-lifetimes`. The `rep` entries are
reported, not predicted.

## Verdicts

| # | Verdict |
|---|---|
| E1 | **Holds** in all 18 graded stages. Every control passed, every arm is rc 0 and specdiff PASS, every bench line is fresh, and every output equals the twin's. |
| E2 | **Holds.** The largest difference between the board's instret ratio and the twin's is gcc's 0.08 %. Every other ratio agrees with the twin's within 0.03 %. |
| E3 | **Holds** on lbm, nab, imagick and gcc, on both boards and in every arm. mcf is pending. Reported: xz Full dev/baseline 0.977. |
| E4 | **Holds** on imagick and gcc, both boards. mcf and E4b are pending. |
| E5 | **Fails as worded** in two places. Both are explained below. |

### E5's two failures

**gcc, step 2, both boards.** The time outside the context fell 11.4 s while in-context time fell only 0.5-0.7 s.
- The cause is memory. Peak pages fall from 6,209 to 2,245 on board 1 and to 2,227 on board 2.
- The QEMU twin reads the same drop (census-eval-boarddev 6,209 → census-eval-boarddevmk 2,227). So the drop belongs to the images, not to the board.
- gcc's test run spends 47 s of its 55 s outside the context, and the per-page cost dominates that time. The prediction did not foresee that the codegen flag would change the page count.
- **Which allocation shrinks is not identified.** The hypothesis is the stack: deeper recursion through frames that the stack-copy rewrite enlarges. It is not checked.

**lbm Spatial, step 2.** In-context −1.2 s, outside −2.1 s. Both changes are about 0.1 % of a 2,034 s command, within the
run-to-run spread. This reading fails the wording and is not read as an effect.

## Findings beyond the predictions

**Cycles fall far more than instructions in step 2, also without the lifetime check.**

| Benchmark | Instret (devmk / dev) | Cycles, Full or MapLife (bitstream 3) | Cycles, Spatial (ObjBounds) |
|---|---|---|---|
| imagick | 0.9696 | 0.8773 / 0.8799 | 0.8947 |
| gcc | 0.968 | 0.9174 | 0.9296 |
| xz | 0.9904 | 0.9371 | pending |
| nab | 0.995 | 0.9685 / 0.9707 | 0.9782 |
| lbm | 0.9996 | 0.999 | 0.9991 |

- Step 2 removes STC/LDC pairs that copy a capability through a stack slot.
- On imagick, 3 % fewer instructions take 10.5-12.3 % fewer cycles.
- On ObjBounds the lifetime check costs nothing: that is the bare kit's cost pattern, which HOLDS on board 2. So the saving there
  is not the lifetime check.
- Per removed pair, taking two removed instructions per pair (an assumption: the flag may change other code as well), silicon
  saves 23 cycles on imagick and 27 on nab (ObjBounds), and 28-36 on imagick, gcc and xz (bitstream 3).

**The store-to-load hypothesis is refuted.** The first version of this file (dd8bfa5e08cb) suggested a same-slot
store-to-load stall in the LSU: a load matching a buffered store waits in WAIT_PAGE_OFFSET. The rtl lane tested it in simulation
(dev bf2fc1c9d6bd, docs/history/04-11-2026_03-13-48_stc-ldc-pair-cost.md; pre-registered on capstone-ariane d463206ac).
- **The pair cost does not depend on the slot.** `stc A; ldc A` costs exactly what `ldc A+0x10`, `A+0x40` and `A+0x1000` cost.
  Per iteration: 20.44 / 11.48 / 11.46 cycles on b3 / ObjBounds / C.5a, with 0 WAIT_PAGE_OFFSET entries. Plain sd/ld pairs do
  pay the wait.
- **What the simulation reads instead:** LDC and STC go through the DYN unit, which takes one op at a time. The next DYN op is
  dispatched 3 cycles after the previous one's result. So the store has drained before the load reaches the load unit.
- **The cost of a pair** over the loop on ObjBounds/C.5a: a capability pair 8.4 cycles, a plain pair to different slots 2.4.
  These are simulation patterns (memory delay 12), not silicon costs.
- **The gap.** The sim pair cost is about a third of the 23-36 cycles silicon saves per removed pair. The rest is not explained.
  A counter boot on the dev and devmk imagick stages, with issue stall and scoreboard full, is the next reading.

**Totals, steps 1 and 2 together** (devmk against the 3 Nov baseline, in-context cycles of run commands):

| Benchmark | Full | MapLife | Spatial | Run time, Full (whole command) |
|---|---|---|---|---|
| imagick | 0.785 | 0.767 | 0.787 | 0.888 |
| gcc | 0.886 | | 0.891 | 0.779 |
| nab | 0.912 | 0.925 | 0.928 | 0.913 |
| xz | 0.916 | | pending | 0.954 |
| lbm | 0.958 | 0.958 | 0.962 | 0.966 |

lbm's step-1 instruction drop (−14.8 %) buys only 4 % of cycles: the removed instructions are not where lbm waits.

## Correction to Addendum 1

Addendum 1 gives the bare kit's lifetime-check cost as "3.98 and 4.98 cycles per op". Those two numbers are the grader's
stall per op. The cost in cycles per LDC/STC op is 3.96 and 4.96. No prediction depends on the difference.

## Limits

- N = 1 per arm.
- The baseline ran on 1-3 Nov, these runs on 3-4 Nov.
- The control is the only check on drift between days.
