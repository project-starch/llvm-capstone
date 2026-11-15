# memcached 1.6.45 with one worker and a fixed client seed, run-6 second addendum (2026-11-15)

The pre-registration is
[`docs/plans/2026-11-15-memcached-fixed-seed.md`](../../../../../docs/plans/2026-11-15-memcached-fixed-seed.md),
committed as 273275cd81b before these runs. It adds one change to the one-worker runs
([`2026-11-15-one-worker`](../2026-11-15-one-worker/README.md)): mc-crusher seeds every connection with a
constant and the connection's order in the configuration, through
`experiments/study/reuse-hooks/memcached/mc-crusher-fixed-seed.patch` (sha256 3ca9062dba2190752b49972274799fab835a4b019174e6f69776c17dfa386d99).
The server images (run-6's, from 07f60dc6f7f), the flags with `-t 1`, the oracle, the workload, the
kit and mc-harness are the one-worker setup's. The records are in
`~/asr-work/eval/run-6/reuse/memcached-t1seed-<arm>-<rep>/` on the build host, each with its
`provenance.txt`, which records the patch and its sha256. [`reuse.json`](reuse.json) is their assembly.

## Client

`build-load-client.sh` with `MCC_PATCH` built the patched mc-crusher (sha256 68c45a438915...). The same
script without the patch built the control (90fa3e0b05eb...). Stripped, the control is byte-identical to
run-5's mc-crusher (both e5fe445b710a...). The unstripped files differ in their debug information, which
holds the build directory. So the patched client differs from run-5's by the patch. mc-harness rebuilt
byte-identical to run-6's (725ca8d13733...), and the runs used run-6's binary.

## Runs

| run | boot s | load s | slabs issues / releases / reuses / distinct | cache issues / releases / reuses / distinct | error | evictions (3, 8, 13, 19) |
|---|---|---|---|---|---|---|
| stock-1 | 859.1 | 812.0 | 60,352 / 47,277 / 47,276 / 13,076 | 61,752 / 61,751 / 61,741 / 11 | 0, 0 | 29,084 (3,555, 7,596, 7,781, 10,152) |
| stock-2 | 848.7 | 801.6 | 60,352 / 47,277 / 47,276 / 13,076 | 61,748 / 61,747 / 61,736 / 12 | 0, 0 | 29,083 (3,557, 7,595, 7,780, 10,151) |
| protected-1 | 861.2 | 813.7 | 60,352 / 47,277 / 47,276 / 13,076 | 61,749 / 61,748 / 61,737 / 12 | 0, 0 | 29,083 (3,554, 7,597, 7,780, 10,152) |
| protected-2 | 860.7 | 812.2 | 60,352 / 47,277 / 47,276 / 13,076 | 61,746 / 61,745 / 61,734 / 12 | 0, 0 | 29,082 (3,557, 7,595, 7,779, 10,151) |

Every load completed with processed = 240,000, and `capstone-job` recorded exit 0 in every run. No run
printed anything on stderr but the observer lines. All four oracle transcripts are identical
(e0a254c47e7e, the one-worker transcript). In all four runs every class has the same chunk size, chunks
per page and pages as in the one-worker runs (class 13: 6 pages, class 19: 15), and `total_malloced`
reads 33,554,432.

## Share of reuses in another bucket (reusediff)

| boundary | stock-1 x stock-2 | protected-1 x protected-2 | s1 x p1 | s1 x p2 | s2 x p1 | s2 x p2 |
|---|---|---|---|---|---|---|
| slabs | 0.0015 | 0.0012 | 0.0023 | 0.0012 | 0.0023 | 0.0020 |
| cache | 0.000016 | 0.000065 | 0.000016 | 0.000081 | 0.000032 | 0.000065 |

Slab bucket counts, stock range against protected range: bucket 1 42,974 to 43,011 against 42,970 to
43,026, bucket 2 21 to 28 against 29 to 32, bucket 3 596 to 620 against 569 to 580, bucket 4 2,275 to
2,323 against 2,367 to 2,370, bucket 5 1,338 to 1,366 against 1,282 to 1,327.

## Verdicts

1. **Slab noise floor = 0: fails.** stock-1 x stock-2 reads 0.0015 (0.0022 with the clock seed).
2. **Stock x protected at the slab boundary within the stock floor: fails for three pairs of four.**
   Three pairs read 0.0020 to 0.0023 and one 0.0012, against a stock floor of 0.0015 and a protected
   floor of 0.0012.
3. **Cache boundary: holds.** The largest within-arm value is 0.000065, four reuses, and the largest
   stock x protected value 0.000081, five reuses, within the allowance of three more.

The fixed seed made the run deterministic where the keys decide: the four transcripts agree, issues,
releases, reuses and distinct starts agree, and evictions lie within 2 of each other (29,082 to 29,084).
The slab distribution still differs between two runs of one build. The keys are now the same, so the
remaining order comes from the server side: the one worker serves sixteen connections in the order
their data arrives, gets move items in the LRU, and memcached's LRU maintainer and crawler threads act
on time.

The arms differ by 0.0012 to 0.0023, the size of the within-arm values, and a third of run-5's gap with
the old layout (0.0056 to 0.0084). Bucket 1 of the two arms overlaps. Buckets 2 to 5 do not overlap in
these two repetitions per arm: protected has about 70 more reuses in bucket 4 (0.15 % of 47,276) and
fewer in buckets 3 and 5. Two repetitions per arm cannot tell a residual of that size from noise. One
candidate cause would be timing: the protected server does more work per allocation and free, which
moves the time-driven LRU threads relative to the commands, but here the protected boots ran only
0.2 to 1.5 % longer than stock. These runs do not test it.
