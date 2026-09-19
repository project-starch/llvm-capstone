# vp9: a flush that discards references it does not release

`vp9_decode_flush()` releases `s->s.frames[]`, `s->s.refs[]` and
`s->s.ref_frames[]` but leaves `s->next_refs[]` referenced. Under frame
threading, `vp9_decode_update_thread_context()` seeds a worker's `refs[]` from
the source worker's `next_refs[]`, so references the flush was supposed to
discard survive, are resurrected into a worker, and a later inter frame passes
the availability check and decodes against them.

    arm=fixed resurrected=0 decoded_against_epoch=0xFF freed_to_malloc=0
    arm=buggy resurrected=3 decoded_against_epoch=0x77 freed_to_malloc=0

## Why this case is different from the others, and why it is worth keeping

**Nothing is freed.** The retained reference keeps the storage alive, so there is
no stale pointer and no reissued address — `freed_to_malloc=0` is printed to
make that unmissable. What is violated is the lifetime *contract*: the flush is
the point at which those references stop being usable, and they are used after
it.

That is the row the paper argues no quarantine can reach, because a quarantine
keys on `free()` and there is no `free()` here. It is also the row where this
port's own protected arm is expected to **lose**: mode 2 revokes each last
return to the pool, and there is no last return, so a Sublet arm would complete
rather than fault. Catching it needs revocation at the flush, which is a
different adapter policy, not a different allocator.

No domain arm is registered for this case for exactly that reason. Adding one
that completes in both modes would record the gap honestly; adding one that
faults would require changing what the adapter revokes, and that is a design
decision, not a fixture.
