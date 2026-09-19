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
