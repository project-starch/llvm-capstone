# Negative control for the thirteen-case suite, 2026-09-21

**4/4 oracles fired.** No arm reported a pass on an input the program refused.

    matrix.tsv    the four result lines, all `passed = False` BY DESIGN
    inputs.json   the same image hashes as ../20260921-qemu/ for cases 0 and 12

## What this is for

`../20260921-qemu/` reports 26/26. That number is worth nothing until the
checks behind it are known to be capable of saying FAIL.

## How it was produced

The same images, byte-identical, with one difference: the input record's
`count` field is 2 instead of 1, so the driver's own `CHECK(in->count == 1)`
refuses it and `wm_give_up` returns before any case runs.

    run-defects.py OUT --cases 0,12 --modes spatial,sublet --negative-control

The flag inverts the exit status: 0 means every arm failed as it must.

## Why these two cases

Case 0 is the base shape — a fault expected at the read probe. Case 12 is the
recorded non-detection, whose `sublet` oracle requires a *completion*; the
control shows that oracle can fail too, so its pass is not vacuous.
