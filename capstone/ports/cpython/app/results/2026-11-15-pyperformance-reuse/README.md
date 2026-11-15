# CPython 3.13.7: reuse distance at pymalloc under pyperformance, stock against protected, 2026-11-15

The run of `capstone/docs/plans/2026-11-15-cpython-established-workload.md` (committed in
b3f26dbb24 before the run, not edited). Workload: a fixed subset of pyperformance 1.14.0 in one
process (`experiments/study/reuse-hooks/cpython/pyperformance-subset.py`), json_dumps x6,
json_loads x167, richards x1, deltablue x9, chaos x1, go x1, nbody x1, gc_collect x37,
PYTHONHASHSEED=0. Stock = pymalloc as released; protected = patch 0014. Two boots per arm, all
four in parallel on the trap-route platform of run-3 (QEMU c55fcdd2, kernel 0ea6d1ef, firmware
975a2421, launcher 150d9692), 2026-11-15 04:25Z to 05:23Z. Run directories on p13:
`~/asr-work/eval/run-5/reuse/cpython-{stock,protected}-{1,2}/` (rep.json, provenance.txt,
guest log). `reuse.json` here is `collect-reuse.py` over the four rep.json files.

## Verdicts

| Prediction | Verdict | Numbers |
|---|---|---|
| 1. Subset at least 10x start-up, at least 10^6 blocks | holds | 4,405,926 blocks per run; start-up alone 18,307 (0.42 %); start-up plus the pyperf import 153,134 (3.5 %, calibration) |
| 2. Noise floor below 0.001; issues and releases equal within each arm | holds | reusediff stock-1/stock-2 2.3e-7 (one reuse), protected-1/protected-2 4.6e-7 (two reuses) |
| 3. Issues and releases equal in all four runs; median bucket equal; between arms below 0.01 and at most 0.001 above the noise floor | holds | issues 4,405,926 and releases 4,391,698 in all four; median bucket 2 in all four; reusediff 1.090e-4 to 1.095e-4 over the four pairs |
| 4. Complete, exit 0, observer error 0 in both processes, same stdout | holds | all four, stdout equal (richards True, go 5, PYPERFORMANCE-OK 8) |

The arms differ, and the difference is systematic: both protected runs have 210 fewer reuses and
210 more distinct start addresses than both stock runs, and the between-arm share (0.011 % of
reuses in another bucket) is about 240 times the larger noise floor. The moved reuses sit at long
distances (buckets 13-21, 4,096 to 2^21 hand-outs; bucket 18 alone -374). With the one-liner,
the arms differed by 0.08 % on the 2026-10-11 platform (there protected had 6 more reuses) and by
0.012 % in run-3 on this platform (equal counters, 3 reuses moved between buckets 4, 6 and 7).
The cause of the difference here was not tested.

## Counters

| run | issues | releases | reuses | distinct starts | error | guest s |
|---|---|---|---|---|---|---|
| stock-1 | 4,405,926 | 4,391,698 | 4,322,874 | 83,052 | 0 | 2,857 |
| stock-2 | 4,405,926 | 4,391,698 | 4,322,874 | 83,052 | 0 | 2,866 |
| protected-1 | 4,405,926 | 4,391,698 | 4,322,664 | 83,262 | 0 | 3,387 |
| protected-2 | 4,405,926 | 4,391,698 | 4,322,664 | 83,262 | 0 | 3,405 |

Calibration (in the plan) also measured raytrace after the plan was written: one loop runs
1,493 s of guest time and hands out 1,053,617 blocks, which confirms its exclusion by time.
