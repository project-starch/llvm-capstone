# FFmpeg pool defect corpus

Consumer-side temporal defects whose stale storage is memory an `AVBufferPool`
handed out. `pool_release_buffer` (`libavutil/buffer.c:344`) pushes a payload
onto a LIFO freelist and `av_buffer_pool_get` (`:390`) hands the identical
`buf->data` back, so nothing reaches `malloc` and same-address reuse is a
property of the allocator rather than of a run.

    00_461fb22053_af_join_dedup_bound/            a reference is never taken; stale read
    01_1886c3269d_h264_refs_partial_clear/        reset bounded by the count, not the array
    02_316531e61c_vidstab_parked_plane_pointer/   pointer parked in a library; stale write

Three cases, three shapes. The inventory and triage that selected them, and the
further pool-backed specimens it found that are not built here, are in
[`docs/ref/ffmpeg-pool-consumer-defects.md`](../../../docs/ref/ffmpeg-pool-consumer-defects.md).

## The contract

The layout and the `case.json` fields are the corpus contract in
[`cpython/pymalloc-repros/SCHEMA.md`](../../cpython/pymalloc-repros/SCHEMA.md),
which is the authority; it is referenced rather than copied, because a contract
that exists twice is two contracts. One directory per case,
`NN_<upstream-fix>_<slug>/` holding `case.c`, `case.json` and `PROVENANCE.md`;
case numbers dense from 0; `case.c` declares the number its directory carries
and the driver refuses a fixture that names another.

Where this corpus differs, and why:

* **An extra arm, `native-fix-differential`.** The other corpora's arms differ
  by *protection*, the defect being present in both. Here the native pair
  differs by whether the **upstream fix** is applied, which is a different axis
  and is named rather than folded into `spatial`/`sublet`. The protected arms
  are the port's probe cases 36–38 and do differ by protection only.
* **`native-detect` is declared and not written**, and not merely unwritten but
  tautological here: the port's payload arena is one allocation, so ASan's
  silence would measure the fixture rather than FFmpeg.

## Running

    bash runners/run-native.sh [outdir]

One program per case, built from its `case.c` plus `shared/driver.c`, each run
twice. The control arm runs first and an infrastructure failure exits 75 with no
verdict. The protected arms live with the port, because they need a toolchain and a
guest a per-case script would have to reinvent. A Capstone domain:

    bash ../../../ports/ffmpeg/buffer-pool/security-tests/qemu/run.sh <out> \
      --cases 36,37,38 --modes 0,2 --rounds 1

and CheriBSD with PoisonCap, where the same three cases are registered as
`pool-<mode>-<case>`:

    python3 ../../../ports/ffmpeg/buffer-pool/host/cheribsd/poisoncap/run.py \
      <build> <out> --stage pool --disable-default-revocation \
      --case poison-live --case poison-read --case poison-write \
      --case poison-reuse --case poison-reused-read \
      --case pool-0-36 --case pool-2-36 --case pool-0-37 --case pool-2-37 \
      --case pool-0-38 --case pool-2-38 --sdk ... --rootfs ... --image ...

Keep the five `poison-*` controls in that selection. Without them the run shows
only that mode 2 ends differently from mode 0, which is a differential and not
evidence that poisoning was active; with them the platform is demonstrated
independently of these cases. `selection.json` records `complete_suite: false`
for any subset, so a partial run cannot later read as a full one.

## The three arms side by side

| case | spatial | Sublet | PoisonCap mode 0 | PoisonCap mode 2 |
|---|---|---|---|---|
| `af_join` | completes | faults, cause 24 | completes | **SIGPROT**, exit 162 |
| `h264_refs` | completes | faults, cause 24 | completes | **SIGPROT**, exit 162 |
| `vidstab` | completes | faults, cause 24 | completes | **SIGPROT**, exit 162 |

The two protected systems are not interchangeable and the oracles say so: a
domain halts and publishes a fault PC, which the runner compares against the
address that boot printed for its probe; a CheriBSD process reports a status,
so there the setup marker is what separates the result from an arbitrary crash.
No timing comparison is made or implied — the two run on different emulators.

## Scope

Real: `libavutil/buffer.c`, compiled unmodified through the port. Reduced: the
consumer, to the allocator call sequence it makes, and the part of `AVFrame`
that sequence touches — the port extracts the allocator, not `frame.c`. Each
case's `PROVENANCE.md` states that split for itself.
