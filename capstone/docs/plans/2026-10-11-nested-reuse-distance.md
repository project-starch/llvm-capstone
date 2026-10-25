# Reuse distance at the nested allocators, stock against protected: pre-registration, 2026-10-11

Committed and pushed before any instrumented build or run. A reading that differs from what is
written here is a finding; this file is not edited afterwards.

## The question

The QE report's reuse figure (section 5, `fig:mallocbench-reuse`) shows that protecting the system
allocator leaves its reuse distances nearly unchanged. This measures the same quantity at each
application's nested allocator: does the lifetime patch change when the allocator reuses an
address?

## Definition (the report's)

Reuse distance counts allocations between the release of an address and its next allocation, at
one allocator boundary. Distances go into power-of-two buckets by bit length: bucket k holds
distances d with 2^(k-1) <= d < 2^k. The observer is `experiments/study/reuse-gap-observer.h`
(integer-only, no capability retained). Its `bins[b]` holds bit length b + 1, so bucket k =
bins[k-1]. The event index is the allocator's own hand-outs (`reuse_gap_attempt` once per
hand-out), and a reuse is a hand-out at a start address that was released before. The run is
rejected if the observer's `error` is nonzero.

## Arms and workloads

Each program, from one source snapshot, two arms on the platform of
`2026-10-11-protected-same-answers.md` (QEMU af37cc32, launcher 3975314a, module a1c6cb6b):
**stock** and **protected** (the lifetime patch), both with the same observer hooks compiled in
behind one define. The hooks only read addresses; they are placed at the allocator's hand-out and
release, the points the lifetime patch also instruments, and they record the address the
allocator itself uses (not the caller's capability). Workloads are those of
`2026-10-11-protected-memory-behaviour.md`:

| Program | Allocator boundary observed | Workload |
|---|---|---|
| SQLite 3.22.0 | memsys5 malloc/free (and the lookaside, as a second boundary) | speedtest1 `main --size 1`, 8 MiB memsys5 heap |
| PostgreSQL 17.5 | memory-context chunk hand-out and pfree, all contexts | `work.sql`, single-user |
| CPython 3.13.7 | pymalloc block hand-out and free | the run-ports.py JSON/GC workload |
| mruby 4.0.0-rc2 | GC object slot hand-out (`mrb_obj_alloc`) and sweep free | the AO render benchmark |
| Perl 5.36.3 | SV head hand-out and free | `smoke.pl` at 2,000 objects |
| memcached 1.6.45 | slab chunk hand-out and free (and cache.c objects, as a second boundary) | the protocol oracle's set/get workload |
| FFmpeg 9.0.1 | AVBufferPool and AVRefStructPool entry hand-out and return | the decode workload |
| Wireshark 4.6.8 | wmem block and block_fast object hand-out and free | the capture set |

One run per arm suffices where the program is deterministic; where the memory-behaviour runs saw
variation between repetitions (Perl, CPython), three repetitions per arm.

## Predictions

1. **Equal for seven programs:** stock and protected give the same reuse histogram at every
   observed boundary, bucket for bucket, for PostgreSQL, CPython, mruby, Perl, memcached, FFmpeg
   and Wireshark. The patches leave placement and free-list order alone. Where repetitions vary,
   the prediction is that the arms vary alike (overlapping ranges per bucket).
2. **SQLite differs:** the patch makes memsys5 report the request instead of the block, and SQLite
   sizes buffers from that, so the allocation sequence changes (memory-behaviour run, ce31d2a10e6e).
   The median bucket is predicted equal; the share of reuses whose bucket differs is reported.
3. **The figure:** one panel per program in the style of `fig:mallocbench-reuse`, the cumulative
   fraction of reuses against the allocation distance, stock wide and light, protected thin and
   dark. Predicted: the two curves overlap in seven panels.

For every program the report gives issues, releases, reuses, distinct start addresses and the
share of reuses whose bucket differs between the arms (the report's `reusediff` measure).
