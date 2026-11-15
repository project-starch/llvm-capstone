# memcached 1.6.45 with one worker thread, run-6 addendum (2026-11-15)

The pre-registration is
[`docs/plans/2026-11-15-memcached-one-worker.md`](../../../../../docs/plans/2026-11-15-memcached-one-worker.md),
committed as 1866247dd2f before these runs. It adds one change to run-6
([`2026-11-15-port-layout-fix`](../2026-11-15-port-layout-fix/README.md)): the server runs with `-t 1`
instead of `-t 4`. The images are run-6's, unchanged, because `-t` is a run-time flag (stock sha256
6ddd173e904143a8..., protected 84ae8f5f3ef09c4f..., built from 07f60dc6f7f). The workload, client,
kit and patches are run-6's. `run-reuse.py` ran with the arguments `run-established.sh` passes, except
`--flags`, two servers per arm, all four boots in parallel. The records are in
`~/asr-work/eval/run-6/reuse/memcached-t1-<arm>-<rep>/` on the build host, each with its
`provenance.txt`. [`reuse.json`](reuse.json) is their assembly by `assemble-reuse.py`.

## Runs

| run | boot s | load s | slabs issues / releases / reuses / distinct | cache issues / releases / reuses / distinct | error | evictions (3, 8, 13, 19) |
|---|---|---|---|---|---|---|
| stock-1 | 959.9 | 909.2 | 60,352 / 47,277 / 47,276 / 13,076 | 61,764 / 61,763 / 61,752 / 12 | 0, 0 | 29,316 (3,644, 7,645, 7,872, 10,155) |
| stock-2 | 958.2 | 908.3 | 60,352 / 47,277 / 47,276 / 13,076 | 61,770 / 61,769 / 61,758 / 12 | 0, 0 | 29,314 (3,645, 7,645, 7,872, 10,152) |
| protected-1 | 979.3 | 929.3 | 60,352 / 47,277 / 47,276 / 13,076 | 61,770 / 61,769 / 61,758 / 12 | 0, 0 | 29,313 (3,648, 7,646, 7,872, 10,147) |
| protected-2 | 992.1 | 942.6 | 60,352 / 47,276 / 47,275 / 13,077 | 61,774 / 61,773 / 61,762 / 12 | 0, 0 | 29,314 (3,645, 7,645, 7,869, 10,155) |

Every load completed with processed = 240,000, and `capstone-job` recorded exit 0 in every run. No run
printed anything on stderr but the observer lines. In all four runs every class has the same chunk
size, chunks per page and pages, and `total_malloced` reads 33,554,432. The chunk sizes and chunks per
page are run-6's. One worker splits the pages differently from four: class 13 holds 6 pages and
class 19 holds 15 (4 and 17 with `-t 4`).

Oracle transcripts: stock-1, stock-2 and protected-1 read e0a254c47e7e. Against run-6 they differ only
in the line `STAT threads 1`. protected-2 reads f4c5e69899ab: it also prints `STAT curr_items 329`
instead of 328 in phase 4.

## Share of reuses in another bucket (reusediff)

| boundary | stock-1 x stock-2 | protected-1 x protected-2 | s1 x p1 | s1 x p2 | s2 x p1 | s2 x p2 |
|---|---|---|---|---|---|---|
| slabs | 0.0022 | 0.0027 | 0.0012 | 0.0018 | 0.0034 | 0.0017 |
| cache | 0.000032 | 0.000032 | 0.000016 | 0.000016 | 0.000016 | 0.000049 |

At the cache boundary bucket 1 holds 99.98 % of the reuses in every run: the one worker takes back
the buffer it returned. The differences are one to three reuses of about 61,750. At the slab boundary
bucket 1 holds 91.11 and 91.20 % in stock and 91.10 and 91.27 % in protected.

## Verdicts

1. **Noise floor = 0: fails at both boundaries.** stock-1 x stock-2 reads 0.0022 at the slab boundary
   and 0.000032 at the cache boundary. The plan named the slab boundary's remaining source: mc-crusher
   seeds each connection's key generator from the clock, so the key sequences differ between runs, and
   evictions follow the keys. The cache floor is two reuses in another bucket.
2. **Stock x protected within the stock floor: holds for three pairs of four at each boundary.** At the
   slab boundary stock-2 x protected-1 reads 0.0034 against a stock floor of 0.0022 (protected floor
   0.0027). At the cache boundary stock-2 x protected-2 reads 0.000049 against 0.000032, three reuses
   against two.
3. **Layout: holds for every class, `total_malloced` and slab issues. Distinct starts fail as written.**
   The plan carried run-6's 12,368. With one worker three runs read 13,076 and protected-2 reads 13,077.
4. **Transcripts: fail as written.** The plan asked for four identical transcripts. protected-2 differs
   in `curr_items`.

protected-2's two differences have one source: one item stays linked that the other runs freed.
`curr_items` is one higher at phase 4 (329) and after the load (13,072 against 13,071), slab releases
are one lower and distinct starts one higher. The oracle stores two items that expire at once (`touch
gone -1`, `set past 0 -1`). memcached removes an expired item only on a get or when its LRU crawler or
maintainer thread meets it, and both threads run on time. With `-t 4` all ten run-5 and run-6
transcripts agreed. With `-t 1` the timing of those threads moved one removal past phase 4 in one run.
These runs do not establish that mechanism.

One worker makes evictions nearly fixed: 29,313 to 29,316 in all four runs, against 30,938 to 31,216
with four workers. At the slab boundary three stock x protected values lie below both within-arm
values and one lies above them, and bucket 1 of the two arms overlaps.
