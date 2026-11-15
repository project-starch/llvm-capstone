# memcached 1.6.45 with one worker and a fixed client seed, second addendum, pre-registration 2026-11-15

Committed before any run of this addendum. A reading that differs from what is written here is a
finding; this file is not edited afterwards.

## Why

The one-worker addendum (`docs/plans/2026-11-15-memcached-one-worker.md`, records in
`ports/memcached/app/results/2026-11-15-one-worker/`) left a noise floor of 0.0022 at the slab boundary.
mc-crusher seeds every connection's generator with `pcg32_srandom_r(&c->rng, time(NULL), (intptr_t)c)`
(`mc-crusher.c:644` at 7ca3f14c), the clock and the connection's heap address, so the key sequences
differ between runs and the evictions follow the keys. A fixed seed removes that source.

## The one change

`experiments/study/reuse-hooks/memcached/mc-crusher-fixed-seed.patch` (sha256 3ca9062dba21...) replaces that
line with a constant seed, 0x6d632d6372757368, and the connection's order in the configuration as the
PCG stream: `pcg32_srandom_r(&c->rng, 0x6d632d6372757368ULL, conn_seq++)`. mc-crusher opens its
connections in configuration order on its main thread, so each connection gets the same stream in every
run and the sixteen streams stay distinct. mc-crusher has no option for this.
`build-load-client.sh` applies the patch when `MCC_PATCH` names it, and records the patch and its sha256
in `SOURCES.txt`.

As a control, the same script builds the unpatched client once more. If that build reproduces run-5's
mc-crusher (sha256 6fd931c281cc...), the patched build differs from it by the patch alone. The runs use
the patched mc-crusher and run-6's mc-harness binary (725ca8d1373367ae...).

Everything else is the one-worker setup: run-6's images from 07f60dc6f7f, unchanged (stock
6ddd173e904143a8..., protected 84ae8f5f3ef09c4f...), `-l 127.0.0.1 -p 21299 -U 0 -m 32 -t 1 -A -o
slab_automove=0` with `EVENT_NOEPOLL=1`, the oracle, `mc-crusher-established.conf` (240,000 commands),
the kit, and `run-reuse.py` with the arguments `run-established.sh` passes except `--flags` and the load
client. Two repetitions per arm, all four boots in parallel (at most four QEMU instances). Records go to
`~/asr-work/eval/run-6/reuse/memcached-t1seed-<arm>-<rep>/` with `provenance.txt`, which records the
client patch and its sha256.

## Checks

As in the one-worker addendum: observer error 0 at both boundaries, `capstone-job` exit 0,
processed = 240,000, every class's chunk size, chunks per page and pages, and `total_malloced`, equal in
all four runs. The transcripts should read e0a254c47e7e... (the one-worker transcript) unless an expiring
oracle item stays linked past phase 4, as in the one-worker protected-2.

## Predictions

1. **Slab noise floor.** stock-1 x stock-2 reusediff = 0 at the slab boundary.
2. **Arms at the slab boundary.** Every stock x protected reusediff is 0, or at most the stock floor.
3. **Cache boundary.** The cache boundary may keep the timing effect of the expiring oracle items, one to
   three reuses in another bucket. Every stock x protected value is at most the largest within-arm
   value plus three reuses.

One source of variation remains at the slab boundary, and it can keep prediction 1 above 0. mc-crusher
serves its sixteen connections from one event loop, so the order in which sets and gets of different
connections reach the server follows socket readiness, and gets move items in the LRU. memcached's LRU
maintainer and crawler threads also act on time.
