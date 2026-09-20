# FFmpeg pool defect corpus

Consumer-side temporal defects whose stale pointer is memory an `AVBufferPool`
or `AVRefStructPool` handed out. `pool_release_buffer` (`libavutil/buffer.c:344`)
pushes a payload onto a LIFO freelist and `av_buffer_pool_get` (`:390`) hands the
identical `buf->data` back, so nothing reaches `malloc` and same-address reuse is
a property of the allocator rather than of a run.

One binary, case and arm chosen at run time. `run.sh` builds it against the
port's native pool library, runs the **fixed** arm first as the control, and
exits 75 with no verdict if that control does not hold.

    461fb22053_af_join_dedup_bound/            a reference is never taken; stale read
    1886c3269d_h264_refs_partial_clear/        reset bounded by the count, not the array
    316531e61c_vidstab_parked_plane_pointer/   pointer parked in a library; stale write
    a024f8c541_vp9_flush_leaves_next_refs/     contract violated with no free anywhere
    8061098418_abitscope_writes_shared_frame/  in-place rewrite of storage a reader holds
    2a5a14f3ca_aphasemeter_writes_shared_frame/   same, another filter
    de07c57d5a_ahistogram_writes_shared_frame/    same, another filter
    faac31cc86_avectorscope_writes_shared_frame/  same, another filter
    dc8e83b4e0_ebur128_writes_shared_frame/       same, another filter
    1ee3c984b9_snow_writes_shared_picture/        same, encoder side
    b9f91a7cbc_dynaudnorm_writes_input_frame/     same class, opposite direction

Eleven cases, and they are **not one class**. Four are temporal: an object's
lifetime ends and a retained pointer is used afterwards. Seven are not, and
calling them class 3 of the sharing taxonomy was wrong — see below.

The inventory and triage that selected these cases, and the three other
pool-backed specimens not yet built, are in
[`docs/ref/ffmpeg-pool-consumer-defects.md`](../../../docs/ref/ffmpeg-pool-consumer-defects.md).

## Scope

Real: `libavutil/buffer.c`, the pool itself, compiled unmodified through the
port. Reduced: the consumer, to the allocator call sequence it makes, in the
same order, plus the part of `AVFrame` the defect touches. Each case's
`PROVENANCE.md` states that split line by line.

`run.sh` builds native paired arms that differ only in whether the upstream fix
is applied, which shows the defect and its absence but no protected outcome. The
protected arms are the port's QEMU probes, where the same sequence runs spatial
against Sublet under a fault-PC oracle; each case README names its case number.
Nothing here claims an AddressSanitizer
result: the port's payload arena is itself one allocation, so ASan is blind to
it by construction and its silence would measure the fixture, not FFmpeg.

## The seven writability cases are exclusivity, not duration

These seven were first filed here as taxonomy class 3, *reuse-not-free*. That was
a misclassification and it is corrected rather than quietly dropped, because the
distinction is the one the paper turns on.

The taxonomy separates by **dimension**:

| class | dimension |
|---|---|
| 3 reuse-not-free | **Duration**, with no allocator event |
| 6 TOCTOU / double-fetch | **Exclusivity** — both sides hold access, one mutates while the other reads |

Class 3's example is SQLite's `column_text`: the API *documents* that the
pointer is valid only until the next `step()`. The borrow has a stated end, and
the consumer used it past that end. That is duration.

In these seven a filter holds a frame, shares a clone downstream, and draws into
it again. **The downstream reference is still valid and its borrow has not
ended** — nobody told the reader anything. What is violated is exclusivity, by
the writer, while the reader's view is legitimately live. That is class 6's
dimension, not class 3's.

Only `a024f8c541_vp9_flush_leaves_next_refs` is temporal among the eight non-PR
cases: references that the flush was supposed to discard survive it and are
resurrected, so a later frame decodes against references whose lifetime had
ended. No allocator event, but a genuine duration violation.

**A paper about temporal safety should not carry the seven.** They are real
defects on pool storage and worth keeping, but under exclusivity, and the
argument for them is a borrow discipline rather than a revocation one.

## What the measurement showed, and what it now means

Probe case 39 runs the shape in a domain and **mode 2 completes**: this port's
adapter revokes at the pool's allocation cycle, nothing hooks `av_buffer_ref`,
and with nothing returned there is nothing to revoke.

Read as an exclusivity result that is not a surprise but a statement of scope:
revocation ends a lifetime, and no lifetime ends here. Covering these needs the
*borrow* side — a lend that costs the lender its write authority until the loan
returns — which is a different primitive from the one this port applies.
