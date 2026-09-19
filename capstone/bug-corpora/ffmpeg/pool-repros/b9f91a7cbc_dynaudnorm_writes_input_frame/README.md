# dynaudnorm: rewriting the input frame its sender still holds

The same class in the opposite direction. Instead of rewriting a frame it kept
and shared, the filter modifies the frame it **received**, which its sender may
still hold, rather than taking writable storage first.

    arm=fixed holder=sender shared_when_written=0 sender_saw=0xA1 sender_now=0xA1 freed_to_malloc=0
    arm=buggy holder=sender shared_when_written=1 sender_saw=0xA1 sender_now=0xB2 freed_to_malloc=0

Keeping this separate from the retained-frame cases is the point: the storage is
the same pool storage and the violation is the same class, but who the other
holder is — upstream rather than downstream — is what a borrow discipline has to
get right in both directions.
