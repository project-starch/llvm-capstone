# F-04: RETRACTED as a compiler finding -- a per-boot guest wedge that hits any image

**Filed 2026-09-05 from the csmith campaign as "seed 7 wedges at -O0 and matches at -O2";
retracted the same day after a position test.** The wedge follows the guest's state within a
boot, not the image or the optimisation level. It is a runtime symptom (QEMU, the kernel
module or the monitor), handed to the board lane; the compiler is not implicated.

## What was observed, in order

| run | position in its boot | image | result |
|---|---|---|---|
| campaign 1 and 2 | 12th domain | cs7-O0 (`d1f543e17b0f`) | WEDGE |
| campaign 2, after the reboot | 1st | cs7-O2 | RET 505522532 = native |
| bisection batch | 2nd | cs7-O0 rebuilt (`143e3d98b7e2`, code byte-identical, symbol table differs) | RET 505522532 = native |
| position batch B | 12th | cs7-O2 (the very image that passed above) | WEDGE |
| position batch C | 1st | cs7-O0 (the campaign's image) | RET 505522532 = native |
| position batch C | 5th | cs2-O2, which had passed in every earlier batch | WEDGE |
| position batch C, after the reboot | 8th of the second boot | cs7-O0 | WEDGE |
| position batch A | -- | -- | the boot itself never reached the login prompt (infra flake) |

So the claim "an -O0-only miscompile" was one step past the evidence: it rested on -O0 wedging
and -O2 passing, and the -O2 pass had been the first item after a reboot. Every wedge is
preceded by QEMU's `[CAPSTONE] Print = Scalar(0x1234)` (`helper_csdebugprint`), which also
appears during a normal boot, so it is the last thing printed, not the cause.

## What it means for the campaign

A WEDGE row is not a compiler verdict until the same image wedges as the FIRST domain of a
fresh boot; the batch runner reboots and continues, and the campaign counts WEDGE as bad so
a run with one is not read as clean. The plan's "items-per-boot ceiling" calibration was
exactly the question this answers: it is not a fixed ceiling but a nondeterministic wedge
that can hit as early as the fifth domain.

Logs: `/tmp/capstone/fuzz/camp-cycle2/batch.log`, `/tmp/capstone/fuzz/camp-cycle2b/batch.log`,
`/tmp/capstone/fuzz/f04-position/batch-{A,B,C}.log` (scratch; not committed). `cs7.c` is kept
here as the program that first showed it.
