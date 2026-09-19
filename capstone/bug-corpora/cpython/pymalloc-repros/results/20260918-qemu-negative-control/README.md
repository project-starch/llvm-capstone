# Negative control for the defect suite, 2026-09-18

> **SUPERSEDED by `../20260919-qemu-20-negative-control/`**, which covers all twenty reachable
> defects instead of the first eight, against a later binary. Kept for the
> trail; its rows are a strict subset and were not wrong.

**2/2 oracles fired.** Neither arm reported a pass on an input that never ran the
case.

    matrix.tsv    the two result lines, both `passed = False` BY DESIGN
    inputs.json   the same binary hash as ../20260918-qemu/

## What this is for

`../20260918-qemu/` reports 16/16. That number is worth nothing until the checks
behind it are known to be capable of saying FAIL — the single most expensive
class of mistake on this project is a clean result from an instrument that could
not have produced any other.

## How it was produced

The same `defects.dom`, byte-identical (compare `inputs.json` in both
directories), with one difference: the input record's `count` field is 2 instead
of 1, so the domain's own `CHECK(input->count == 1)` refuses it and `pym_fail`
returns before any defect is performed.

    run-defects.py <out> --cases 0 --modes spatial,sublet --negative-control

The flag inverts the exit status: 0 means every arm failed as it must, 1 means an
oracle is vacuous.

## What each oracle then reported

| arm | why it must fail | what it reported |
|---|---|---|
| spatial | nothing completed, so `report->completed` stays 0 | FAIL |
| sublet | no stale access happened, so no capability fault | FAIL, `cause=0 pc=0` |

Both fired. Combined with the 16 passing arms, each oracle is now known to
produce both outcomes on the same binary, with the input as the only variable.
