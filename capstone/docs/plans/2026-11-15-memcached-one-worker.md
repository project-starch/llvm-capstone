# memcached 1.6.45 with one worker thread, addendum to the port layout fix, pre-registration 2026-11-15

Committed before any run of this addendum. A reading that differs from what is written here is a
finding; this file is not edited afterwards.

## Why

Run-6 (`docs/plans/2026-11-15-memcached-port-layout-fix.md`, records in
`ports/memcached/app/results/2026-11-15-port-layout-fix/`) ran memcached with four worker threads on
one hart. Their interleaving changes the order of allocations, so two runs of the same build differ:
the within-arm reusediff at the slab boundary reached 0.0039, at the cache boundary 0.0177, and one
protected run (protected-1) read 0.13 at the cache boundary against every other run. With one worker
the server takes every command in one thread, so the order of allocations should follow the load
alone and the comparison between the arms should be clean.

## The one change

The server runs with `-t 1` instead of `-t 4`:
`-l 127.0.0.1 -p 21299 -U 0 -m 32 -t 1 -A -o slab_automove=0`, with `EVENT_NOEPOLL=1`.

Everything else is run-6's setup:

- the images built from 07f60dc6f7f by `build-reuse.sh`, unchanged (stock sha256 6ddd173e904143a8...,
  protected 84ae8f5f3ef09c4f...). `-t` is a run-time flag, so no rebuild;
- the oracle (`mc-harness` phases 1 to 4), mc-crusher 7ca3f14c with `mc-crusher-established.conf`
  (240,000 commands), the same client binaries, the writev, shutdown and reuse-gap patches;
- `run-reuse.py` with the arguments `run-established.sh` passes, except `--flags`; the kit copy of
  run-6 (QEMU 9289114e06, kernel Image-0133b9e21cd, firmware f-keep).

Two repetitions per arm, stock and protected, all four boots in parallel (at most four QEMU instances).
Records go to `~/asr-work/eval/run-6/reuse/memcached-t1-<arm>-<rep>/` with `provenance.txt`.

## Checks

Per run: observer error 0 at both boundaries, `capstone-job` exit 0, processed = 240,000, every class's
chunk size, chunks per page and pages and `total_malloced` as in run-6. The oracle transcript cannot
equal run-6's (ee1522c8e0ec...): phase 4 prints the `threads` stat, which reads 1 instead of 4. The four
transcripts must be identical to each other and differ from run-6's in that line only.

## Predictions

1. **Noise floor.** stock-1 x stock-2 reusediff = 0 at both boundaries, and protected-1 x protected-2 = 0.
2. **Arms.** Every stock x protected reusediff is 0, or at most the stock floor of prediction 1, at both
   boundaries.
3. **Layout.** Every class, `total_malloced`, slab issues (60,352) and distinct starts (12,368) are equal
   in all four runs.

Two sources of variation remain with one worker, and either can keep prediction 1 above 0: mc-crusher
seeds each connection's key generator from the clock and the connection's address
(`2026-11-15-memcached-established-workload.md`, "Sources of variation"), and the order in which the one
worker serves sixteen client connections follows their arrival. memcached's LRU maintainer and crawler
threads also act on time. If prediction 1 fails, the stock floor of this addendum is the comparison for
prediction 2.
