# vidstabtransform: a raw plane pointer parked in a third-party library

The separate-buffer path of `vsTransformPrepare()` stores a shallow copy of the
source frame in the library's own state and allocates nothing, so `srcMalloced`
stays 0. When a later frame takes the in-place path, `vsFrameIsNull(&td->src)`
is false, `vsFrameAllocate()` is skipped, and `vsFrameCopy()` writes through the
pointer the previous frame left behind.

This is the write-shape counterpart to the other cases: the stale reference does
not read the new owner's data, it **overwrites** it.

    arm=fixed wrote_through_stale=0 new_owner_byte=0x22
    arm=buggy wrote_through_stale=1 new_owner_byte=0x99

The CPython corpus files this shape as *parked with no bound on reuse*. What is
distinctive here is where it is parked: inside an opaque third-party library's
state, where no amount of care in FFmpeg's own code would find it.

## Paired arms in a Capstone domain

Case 38 of the port's pool lifetime probes. It is the corpus's only **write**
probe, and the oracle shows it: this case faults at a different published
address from the read cases, `ff2_probe_write` rather than `ff2_probe_read`.

| mode | outcome |
|---|---|
| 0 spatial | **completes** — the write lands in the new owner's storage |
| 2 Sublet | **faults**, cause 24, at the published `ff2_probe_write` address |

    bash security-tests/qemu/run.sh <out> --cases 38 --modes 0,2 --rounds 1
