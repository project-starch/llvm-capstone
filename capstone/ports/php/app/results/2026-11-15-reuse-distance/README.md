# PHP reuse distance, stock against protected (2026-11-15)

The PHP row of the reuse-distance study, run against the pre-registration
`capstone/docs/plans/2026-11-15-php-reuse-distance.md` (762c8af4036), on the trap route of run-3
(Image-0133b9e21cd, qemu-1b, f-keep, libc at 7ba3f427f9, kit-run-3). PHP 5.0.0 has no Zend MM:
the boundaries are its Zend small-block cache (`zend-small`, blocks up to 80 bytes, `AG(cache)`)
and the larger blocks (`zend-large`). Workload: Zend/bench.php of PHP 5.1.0 at the pinned
one-tenth setting. Two boots per arm; all six exit 0, observer error 0 everywhere.

Files: `reuse.json` (collect.py over the six bundles in `~/asr-work/eval/run-5/reuse/php` on p13),
`provenance.txt` (hashes of kernel, kit, images, SDKs, libc and workload; the build history).

## Counts (each arm's two runs are identical in every count and bucket)

| Boundary | Arm | issues | releases | reuses | distinct starts | median bucket |
|---|---|---|---|---|---|---|
| zend-small | stock | 2,410,639 | 2,410,639 | 2,374,848 | 35,791 | 2 |
| zend-small | protected | 2,411,356 | 2,411,356 | 2,386,971 | 24,385 | 2 |
| zend-small | exact | 2,411,356 | 2,411,356 | 2,386,971 | 24,385 | 2 |
| zend-large | stock | 1,011,177 | 1,011,177 | 963,564 | 47,613 | 7 |
| zend-large | protected | 1,031,051 | 1,031,051 | 985,204 | 45,847 | 6 |
| zend-large | exact | 1,031,051 | 1,031,051 | 985,204 | 45,847 | 6 |

## reusediff (share of reuses in another bucket)

| Pair | zend-small | zend-large |
|---|---|---|
| stock-1 against stock-2 (noise floor) | 0 | 0 |
| protected-1 against protected-2 | 0 | 0 |
| stock against protected (rep 1, and all four pairs) | 0.567 % | 61.2 % |
| stock against exact (rep 1, and all four pairs) | 0.567 % | 61.2 % |

## Predictions

1. Same request stream: **does not hold as written.** The arms differ by a handful of events:
   cache_hits 2,354,019 against 2,354,021; emalloc misses (`fresh - moves`) 56,599 against 56,603
   (small) and 1,010,636 against 1,010,638 (large); protected small `moves` 732 against stock
   `moves + resizes` 730. At the large boundary `moves + resizes` agree exactly (20,413), and
   protected `resizes` is 0 at both. One timing column differs in width between the arms
   (strcat: 12.216 s stock, 276.080 s protected), the cause the plan allowed for, but the runs
   are deterministic, so the fallback clause (overlapping ranges) fails and the cause is not shown.
2. Histograms differ, centre stays: **holds at zend-small** (median bucket 2 in both, reusediff
   0.567 %), **fails at zend-large**: the median moves from bucket 7 to 6 and reusediff is 61.2 %.
   Every large hand-out is a fresh malloc, and the protected layer asks for other sizes and moves
   every realloc, so this boundary shows mallocng placing differently sized blocks.
3. Exact equals protected: **holds**, every count and bucket at both boundaries, both runs.
4. Clean runs: **holds**: error 0, exit 0, bench.php output equal apart from the timing column.
5. Repetitions: **holds**: each arm's runs are identical (noise floor 0), and stock against
   protected at zend-small (0.567 %) exceeds it.
