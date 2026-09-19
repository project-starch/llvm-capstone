# avectorscope: drawing again into a frame it has already shared

`avf_avectorscope` keeps `s->outpicref` across calls and hands a reference on. On the next frame
it draws into the same storage, which a consumer still holds and reads.

    arm=fixed holder=avectorscope shared_when_written=0 consumer_saw=0xA1 consumer_now=0xA1 freed_to_malloc=0
    arm=buggy holder=avectorscope shared_when_written=1 consumer_saw=0xA1 consumer_now=0xB2 freed_to_malloc=0

Nothing is freed, the pointer stays tagged and in bounds, and only the identity
of the data changes. This is class 3 of
[the sharing taxonomy](../../../docs/design/sharing-bug-taxonomy-and-novelty.md),
the row it places in the Security column with no CHERI knob for it.

The call sequence is identical to
[`8061098418`](../8061098418_abitscope_writes_shared_frame/README.md) and the
three other siblings, and the cases are kept apart for the reason the CPython
corpus keeps its eight free/reuse cases apart: they are separate upstream
reports in separate consumers, and the recurrence is the argument.
