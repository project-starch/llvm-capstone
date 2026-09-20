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

Three cases, three shapes. The inventory and triage that selected them, and the
further pool-backed specimens it found that are not built here, are in
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

