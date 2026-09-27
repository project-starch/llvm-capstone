# FFmpeg 9.0.1: pool lease reuse and explicit payload operations

The [lease-gap figure](reuse-gaps.pdf) and [payload-operation figure](payload-operations.pdf)
measure the complete configured Matroska/MPEG-4 decoder, not allocator replay.
Each process decodes 1, 4 or 16 independent copies of the same pinned 30-frame
clip. The four arms are Capstone original, Capstone + Sublet, PoisonCap spatial,
and the selective-state PoisonCap temporal adapter. **All 36/36 runs pass**:
three independent processes per arm/workload, with the exact 30/120/480-frame
output oracle in every process. The two Capstone modes use one identical domain
binary; the two PoisonCap modes use one identical CheriBSD binary.

The observer counts real FFmpeg pool *lease issues* and logical returns at the
shared payload allocator. A reissue gap is the number of subsequent lease
issues between release and the same payload block's next application lease;
gap one is immediate. Trusted callback/teardown reacquisitions are excluded.
The denominator is **all application lease issues**, including never-reused
blocks. The observer stores only integer block indices and issue counts, not
payload capabilities, in 16,656 static bytes per arm. Its 32 log₂ gap bins
reconcile exactly with the independent issue/reissue totals.

| Streams | Frames | Issues, each arm | Same-block reissues, each arm | Gap ≤15 / all issues, each arm | PoisonCap temporal payload spans processed |
|---:|---:|---:|---:|---:|---:|
| 1 | 30 | 382 | 339 (88.744%) | 339 (88.744%) | 6.978 MiB |
| 4 | 120 | 1,528 | 1,485 (97.186%) | 1,377 (90.118%) | 28.813 MiB |
| 16 | 480 | 6,112 | 6,069 (99.297%) | 5,529 (90.461%) | 116.155 MiB |

**All four arms are identical in every gap bin**, at all three workload sizes
and in every repetition. This pool performs synchronous revocation on return,
so the corrected PoisonCap policy can preserve immediate reuse here; the
SQLite memsys5 allocator's quarantine policy instead delays reuse. The
[SQLite release-gap result](../sqlite-reuse-gaps-20260927/README.md) and this
pool result must remain separate, because they observe different allocator
boundaries and policies.

The payload-operation panel counts byte spans explicitly targeted by the
**current selective PoisonCap adapter**. At 16 streams it issues `cpoison` over
55,249,920 B, `cclearpoison` over 54,934,848 B and copies 11,612,160 B,
totaling 121,796,928 B of cumulative operation spans
(116.155 MiB). Its snapshot backing peaks at 36,288 B and returns to zero.
The spatial PoisonCap arm does none of these operations. The Capstone pool
backend uses Sublet revocation and does no corresponding per-granule
poison/clear/copy pass, but hardware/node activity is not included in this
counter. Poisoning and clearing operate on poison state; their byte counts are
span lengths, **not** payload bytes necessarily written to DRAM. None of these
counts is measured bandwidth, time, RSS or total memory cost. They characterize
this adapter and workload, not a lower bound for every PoisonCap implementation.

## Comparability and accounting checks

Both platforms use the same prepared FFmpeg 9.0.1 source, 4 MiB inner payload
reservation, pool lifetime hooks, configured codec/demuxer, `-O1` application
builds, and shared integer-index observer. Target ABI, libc, compiler driver,
application entry and outer heap differ by platform. The published PoisonCap
guest has automatic outer libc revocation disabled; explicit nested mode-2
revocation remains active. The Capstone application SDK serves all 18 runs in
one persistent Linux guest, and PoisonCap serves all 18 in one CheriBSD guest.
The PoisonCap spatial control keeps the adapter's pool layout and disables its
temporal policy; it is not an unadapted stock-FFmpeg layout. This isolates the
policy increment within that platform, but its denominator differs from
SQLite's original-layout CheriBSD control.
Both FFmpeg library configurations request `--extra-cflags=-O1`; their
generated `CFLAGS` append `-O3` later, so library translation units compile
with `-O3` on both platforms. Their remaining `CFLAGS` differences are two
compiler-specific warning switches; Capstone uses `target-os=none` and
PoisonCap uses `target-os=freebsd`.

The new observer does not change the previously measured allocator ledgers:
all 18 Capstone final outer-heap and inner-pool records match the earlier
[whole-decoder campaign](../ffmpeg-pool-memory-20260927/README.md) exactly.
All 18 PoisonCap release-phase jemalloc allocated series, sweep counts and
snapshot endpoints match the earlier [selective adapter
follow-up](../memory-followup-20260927/README.md). The plotted curves contain
three process repetitions; allocation events within one process are not
independent samples. All repeats coincide binwise, so no confidence interval
is inferred.

The preliminary legacy Capstone domain host completed the 1- and 4-stream
cells but faulted late in the 16-stream qualification. Its diagnostic log is
in the external [raw archive](archive.json), and none of its attempts enter
these 36 measured runs. The application SDK, which had passed this workload
before the observer was added, completed all 18 Capstone attempts and returned
zero live domains and regions after each process. The earlier full-copy
PoisonCap snapshot result is also excluded: it was an adapter artifact, not
the selective policy measured here.

## Evidence and reproduction

[summary.json](summary.json) has every per-process histogram, oracle hash,
binary hash, policy and rewrite counter; [bins.csv](bins.csv) keeps all 1,152
bin observations (36 × 32). [archive.json](archive.json) identifies a separate
raw archive containing all transcripts, both measured binaries and builds,
the pinned clip, independent output oracles, source snapshot and the excluded
legacy diagnostic, without guest private keys. The [Capstone build
recipe](../../build-ffmpeg-capstone-reuse.py), [PoisonCap decoder
entry](../../ffmpeg-poisoncap-decode.c), [shared pool
observer](../../../../ports/ffmpeg/buffer-pool/src/shared/pool-allocator.c)
and [plotter](../../plot-ffmpeg-reuse-gaps.py) are in this worktree.

With the experiment Python environment and the raw archive restored under
`/tmp/capstone`, validate and redraw both paper-width PDFs with:

```sh
source capstone/tests/capstone-test-env.sh
/tmp/capstone/application-memory/venv/bin/python \
  capstone/experiments/study/plot-ffmpeg-reuse-gaps.py \
  --capstone /tmp/capstone/ffmpeg-reuse-capstone-sdk-runs \
  --poisoncap /tmp/capstone/ffmpeg-reuse-poisoncap-runs \
  --out capstone/experiments/study/results/ffmpeg-reuse-gaps-20260927
```

For figure-only redraw without the VM logs, pass `--summary` with the checked
`summary.json`. The plotter rejects missing attempts, changed frame oracles,
wrong pool modes, malformed bins, changed allocator ledgers, and differences
between matched four-arm histograms.
