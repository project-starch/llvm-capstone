# Negative control for the twenty-case suite, 2026-09-19

**4/4 oracles fired.** No arm reported a pass on an input that never ran the
case.

    matrix.tsv    the four result lines, all `passed = False` BY DESIGN
    inputs.json   the same binary hash as ../20260919-qemu-20/

## What this is for

`../20260919-qemu-20/` reports 40/40. That number is worth nothing until the
checks behind it are known to be capable of saying FAIL — the single most
expensive class of mistake on this project is a clean result from an instrument
that could not have produced any other.

## How it was produced

The same `defects.dom`, byte-identical (compare `inputs.json` in both
directories), with one difference: the input record's `count` field is 2 instead
of 1, so the domain's own `CHECK(input->count == 1)` refuses it and `pym_fail`
returns before any defect is performed.

    run-defects.py <out> --cases 0,10 --modes spatial,sublet --negative-control

The flag inverts the exit status: 0 means every arm failed as it must, 1 means an
oracle is vacuous.

## Why two cases and not one

Case 0 is the base shape — free, reuse, stale read. Case 10 is the only case
whose block is ended by a **realloc** rather than a free, and whose driver
asserts the realloc moved the block. Running both means the control covers the
one case that could have failed for a reason unrelated to the oracle.

## What each oracle then reported

| arm | why it must fail | what it reported |
|---|---|---|
| spatial | nothing completed, so `report->completed` stays 0 | FAIL |
| sublet | no stale access happened, so no capability fault | FAIL, `cause=0 pc=0` |

All four fired. Combined with the 40 passing arms, each oracle is now known to
produce both outcomes on the same binary, with the input as the only variable.
