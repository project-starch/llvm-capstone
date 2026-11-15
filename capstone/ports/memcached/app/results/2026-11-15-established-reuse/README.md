# memcached 1.6.45 under mc-crusher: reuse distance with a noise floor (2026-11-15)

The pre-registration is
[`docs/plans/2026-11-15-memcached-established-workload.md`](../../../../../docs/plans/2026-11-15-memcached-established-workload.md),
committed as 78aa45070e0 before any measured run. The data is [`reuse.json`](reuse.json), assembled by
`experiments/study/reuse-hooks/memcached/assemble-reuse.py` from the six run records in
`~/asr-work/eval/run-5/reuse/memcached-<arm>-<rep>/` on the build host (each with its `provenance.txt`).

## What ran

Each boot ran one server: the protocol oracle (mc-harness phases 1 to 4), then mc-crusher
7ca3f14c with `mc-crusher-established.conf` (240,000 commands: value sizes 100, 500, 2,000 and
8,000 bytes, 10,000 zipf keys each, one set and three get connections per size), then `shutdown`.
The server ran as `memcached -l 127.0.0.1 -p 21299 -U 0 -m 32 -t 4 -A -o slab_automove=0` under
`capstone-job --user 65534:65534` with `EVENT_NOEPOLL=1`. There were three boots per arm, all six in
parallel, on the trap-route platform of run-3 and run-4 (kit-run-4). Each boot took 1,128 to 1,149 s,
and the load took 1,077 to 1,098 s of that, in both arms alike.

## Acceptance

All six runs are accepted:
- every boot completed and `capstone-job` recorded exit 0;
- observer error 0 at both boundaries;
- every load completed with processed = 240,000;
- the oracle transcript is identical in all six runs. Its sha256 is ee1522c8e0ec..., the transcript of
  the October virtual-route run. No run printed anything on stderr other than the observer lines.

## Counts

| Run | slabs issues | releases | reuses | distinct | bucket 1 share | cache issues | reuses | evictions |
|---|---|---|---|---|---|---|---|---|
| stock-1 | 60,352 | 47,986 | 47,984 | 12,368 | 0.9133 | 61,787 | 61,767 | 31,062 |
| stock-2 | 60,352 | 47,986 | 47,984 | 12,368 | 0.9150 | 61,777 | 61,757 | 31,047 |
| stock-3 | 60,352 | 47,985 | 47,984 | 12,368 | 0.9152 | 61,767 | 61,747 | 31,216 |
| protected-1 | 60,352 | 49,202 | 49,201 | 11,151 | 0.9216 | 61,783 | 61,763 | 33,624 |
| protected-2 | 60,352 | 49,203 | 49,201 | 11,151 | 0.9213 | 61,767 | 61,747 | 33,575 |
| protected-3 | 60,352 | 49,201 | 49,201 | 11,151 | 0.9208 | 61,778 | 61,758 | 33,438 |

The cache boundary has 20 distinct starts in every run. Every run's slab pages split the same way
(classes 3, 8, 13 and 19 hold 1, 2, 4 and 17 pages).

## Share of reuses in another bucket (reusediff)

| Boundary | within stock (noise floor) | within protected | stock x protected (9 pairs) |
|---|---|---|---|
| slabs | 0.0016 to 0.0030 | 0.0010 to 0.0025 | 0.0056 to 0.0084 |
| cache | 0.0027 to 0.0066 | 0.0023 to 0.0074 | 0.0010 to 0.0097 |

The median bucket is 1 in every run at both boundaries.

## Verdicts

1. **Output: holds.** All six transcripts are identical, and every load completed with 240,000.
2. **slabs counts: holds in part.**
   - Issues are 60,352 in every run, as predicted.
   - Distinct starts are lower in protected, as predicted (11,151 against 12,368).
   - The prediction said releases and reuses vary from run to run. They do not: reuses are 47,984
     in every stock run and 49,201 in every protected run. A hand-out is either a new start or a
     reuse, so reuses = issues - distinct starts, and both terms are fixed once the pages are split.
     The pages split the same way in every run.
3. **slabs distribution: holds within the stated allowance.** The noise floor is at most 0.0030. Every
   stock x protected pair is at most 0.0084, far below the bound of 0.0030 + 0.03. The difference
   between the arms is real, though: it is two to three times the noise floor, and the per-bucket
   ranges do not overlap. Its source is the one the plan named. Each protected page holds fewer
   chunks, so the full classes evict more often (about 33,500 evictions against 31,100). An
   eviction frees the chunk that the next hand-out takes, so it adds reuses at distance 1. Bucket 1
   holds 92.1% of the reuses in protected and 91.3 to 91.5% in stock.
4. **cache distribution: holds.** Every stock x protected pair (at most 0.0097) lies within the
   within-arm spread (up to 0.0074) plus 0.02, and the per-bucket ranges overlap. The noise floor is
   larger than at the slab boundary, as predicted.

## Deviations and limits

- **Response path (both arms):** `memcached-1.6.45-tcp-writev.patch` sends responses through `writev`.
  On the trap route `sendmsg` returns ENOSYS. run-3's memcached rows ran without it, so their
  transcripts are 780 `<EOF>` markers in 791 lines: every response failed.
- **Automover off:** `-o slab_automove=0`. The hooks do not see the mover's rescues, so a page move
  would make the observer report overlapping issues.
- **Memory limit:** `-m 32` instead of run-3's `-m 64`, so that every load class evicts.
- **Load generator:** mc-crusher is unmodified. Its zipf generator is seeded from the clock and the
  connection's address, so the key sequences differ between runs.
- **One record rebuilt:** the first assembly overwrote `memcached-stock-3/result.json`, because of a
  bug that is now fixed: `assemble-reuse.py` still wrote to `sys.argv[4]`. `run-reuse.py --rerecord`
  rebuilt that file from the run's own serial log and stage. On the intact stock-1, the same rebuild
  reproduces its record byte for byte. The rebuilt stock-3 record also matches what the first
  assembly had read from the original. Nothing was rerun. The run directory's `provenance.txt` records
  this.
