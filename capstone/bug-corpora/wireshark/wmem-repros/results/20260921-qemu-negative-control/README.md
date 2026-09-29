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

> **Correction (2026-09-29): the refusal happens earlier than stated above.**
>
> - The record is refused by the **guest-side loader**, before any domain exists. Its length check
>   (`ports/wireshark/wmem/src/linux-guest/domain-loader.c`, present since `f26c4bb4b746`, before
>   this run) requires `bytes == header + count * event`. The runner packed one 48-byte event
>   behind a count of 2, so the loader returned 3 before `create_dom`.
> - The driver's `CHECK(in->count == 1)` is never reached.
> - Established from the source of the loader and the runner at `a6d5d6685f49`. This run's raw
>   logs are not retained on this host.
> - Today's re-run of the negative control shows exactly this: `__EXIT_CODE__3` and no domain load
>   (`../20260929-qemu-chunk-port/`).
> - **The conclusion is unaffected:** every arm FAILED, so the oracles can fail. What this control
>   does not show is the probe-address comparison failing, because no domain ran. That check is
>   shown able to fail in `../20260929-qemu-chunk-port/`, by re-judging real faults against the
>   wrong probe.

    run-defects.py OUT --cases 0,12 --modes spatial,sublet --negative-control

The flag inverts the exit status: 0 means every arm failed as it must.

## Why these two cases

Case 0 is the base shape — a fault expected at the read probe. Case 12 is the
recorded non-detection, whose `sublet` oracle requires a *completion*; the
control shows that oracle can fail too, so its pass is not vacuous.
