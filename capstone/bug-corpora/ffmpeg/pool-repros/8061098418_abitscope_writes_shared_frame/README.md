# abitscope: drawing again into a frame it has already shared

The filter keeps `s->outpicref` across calls. In mode 1 it **clones** it, so the
clone it sends downstream shares the same pooled storage. On the next frame it
draws into `s->outpicref` again — into storage a consumer still holds and reads.

    arm=fixed shared_when_written=0 consumer_saw=0xA1 consumer_now=0xA1 freed_to_malloc=0
    arm=buggy shared_when_written=1 consumer_saw=0xA1 consumer_now=0xB2 freed_to_malloc=0

## Why this case is exclusivity and not duration

Nothing is freed and the pointer stays tagged and in bounds. But the downstream
reference is **still valid and its borrow has not ended** — nobody told the
reader anything. What is violated is **exclusivity**, by the writer, while the
reader's view is legitimately live.

An earlier version of this file called that class 3, *reuse-not-free*, of
[the sharing taxonomy](../../../docs/design/sharing-bug-taxonomy-and-novelty.md).
That was wrong. Class 3's dimension is **Duration**, and its example is SQLite's
`column_text`, where the API documents that the pointer is valid only until the
next `step()` — a borrow with a stated end, used past it. Here there is no
stated end. The dimension is **Exclusivity**, class 6's. **This is not a
temporal case**, and a paper about temporal safety should not carry it.

`av_buffer_is_writable` and `av_buffer_get_ref_count` are the real ones from the
extracted `buffer.c`, so the fixed arm's decision to take fresh storage is made
by upstream's own predicate and not by the fixture.

## Upstream fixed this five times in one day

`8061098418` is one of five commits dated 2026-03-04 … 2022-03-04 that add the
same missing check to five different filters: `avf_abitscope`,
`avf_aphasemeter`, `avf_ahistogram`, `avf_avectorscope` and `f_ebur128`.
`af_dynaudnorm`, `af_speechnorm` and `snow` carry the same shape from other
dates. The recurrence is the argument, as it is for the eight CPython
free/reuse cases: one discipline covers a class upstream keeps rediscovering.
