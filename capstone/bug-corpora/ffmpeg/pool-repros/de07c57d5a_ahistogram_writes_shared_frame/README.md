# ahistogram: drawing again into a frame it has already shared

`avf_ahistogram` keeps `s->out` across calls and hands a reference on. On the next frame
it draws into the same storage, which a consumer still holds and reads.

    arm=fixed holder=ahistogram shared_when_written=0 consumer_saw=0xA1 consumer_now=0xA1 freed_to_malloc=0
    arm=buggy holder=ahistogram shared_when_written=1 consumer_saw=0xA1 consumer_now=0xB2 freed_to_malloc=0

Nothing is freed and the pointer stays tagged and in bounds — but the
downstream reference is **still valid**, and its borrow has not ended. What is
violated is **exclusivity**, by the writer, while the reader's view is
legitimately live. That is the dimension of class 6 in
[the sharing taxonomy](../../../docs/design/sharing-bug-taxonomy-and-novelty.md),
not class 3, which is duration. An earlier version of this file filed it as
class 3; the corpus README explains the correction. **This is not a temporal
case.**

The call sequence is identical to
[`8061098418`](../8061098418_abitscope_writes_shared_frame/README.md) and the
three other siblings, and the cases are kept apart for the reason the CPython
corpus keeps its eight free/reuse cases apart: they are separate upstream
reports in separate consumers, and the recurrence is the argument.
