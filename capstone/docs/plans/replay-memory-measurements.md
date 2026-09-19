# Allocator memory measurements through QEMU replay

## Scope

Measure real allocator decisions and resource demand under recorded requests.
This is not an application-throughput or cache/DRAM experiment. Native
recording, extracted native replay and paired capability-ABI replay are
distinct validation steps. Keep failed attempts and pin the exact binaries,
source configuration and trace bytes.

The first campaign uses FFmpeg's AVBufferPool and AVRefStructPool because
spatial and Sublet share the same backing implementation and geometry. The
PostgreSQL spatial and Sublet backing policies differ; a future comparison
there must be labeled as the complete port's behavior, not pure protection
overhead. CPython and ggml remain follow-up integrations for this campaign.

## First measurement matrix

Fresh FFmpeg 9.0.1 MPEG-4 recordings at 30 frames/second:

| Label | Duration | Dimensions | Purpose |
|---|---:|---|---|
| short | 1 second | 320x180 | Small complete workload |
| turnover | 2 seconds | 320x180 | More recorded activity at the same dimensions |
| larger | 2 seconds | 640x360 | Larger buffer requests |

Each recording is decoded by stock and instrumented FFmpeg with identical
frame hashes, and compared event-for-event with native allocator replay.
Replay each fixed recording three times in each of spatial and Sublet modes,
in a fresh VM per attempt, with a declared 1,048,576-node capacity. These are
repeatability draws of the same serialized trace, not three independently
sampled workloads or an unbounded turnover test.

The collector is [measure.py](../../ports/ffmpeg/buffer-pool/host/memory/measure.py).
It freezes the matrix before execution, preserves each attempt and rejects
changed execution artifacts. It has no automatic retry policy. A failure
requires diagnosis. Explicit `--resume --resume-reason ...` verifies accepted
results again and appends attempts only for unfinished points. Old manifests
and collector identities remain recorded; changing execution artifacts or
the matrix requires a new campaign.

## Measurements and acceptance

- Every allocator observation must match the native recording. The comparison
  includes backing identities, callbacks, reuse gaps and payload counters.
- Export an event-indexed payload series and synchronized combined peaks.
  Do not sum peaks attained at different events or confuse backing retained
  by a pool with idle bytes: pool backing includes live objects.
- Report payload and metadata carving watermarks as such. They include
  retained reusable capacity and are not live occupancy or total overhead.
- Count runtime-reported primitives and initialized bytes separately from
  the observed allocator decisions. These are software operation counters,
  not cycles, node occupancy, hardware traffic or a full resource ledger.
- Check valid aliases, deferred close, and stale-access rejection with the
  same fresh build's companion lifetime fixtures outside measurement runs.
- Native recorder and replay share observer code. Exact agreement validates
  this extraction but is not an independent implementation of its semantics.
  Hand-built accounting controls additionally check peak aggregation and
  rejection of mismatched, unbalanced and failed observations.

## Explicit missing coverage

Fixed payload, metadata and trace/output regions are reported separately;
the first campaign does not determine minimum capacities. Replay bookkeeping,
static domain storage and hardware node/tag storage prevent interpreting the
two carving watermarks as complete process or protection memory overhead.

Further work is the complete memory ledger and independent payload/metadata
budget matrix; valid repeated epochs with actual node reclamation; and
validated node-visit counters for fixed-subtree locality experiments. Adding
other ports requires each port's own accounting and baseline validation.
Raw logs stay outside the checkout; only compact verified results belong in
the port's results directory.
