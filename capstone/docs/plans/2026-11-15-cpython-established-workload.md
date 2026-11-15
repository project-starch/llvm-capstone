# CPython reuse distance under pyperformance, stock against protected: pre-registration, 2026-11-15

Committed before any measured run. A reading that differs from what is written here is a finding;
this file is not edited afterwards.

## The question

`2026-10-11-nested-reuse-distance.md` (16d2171a3edc) measured reuse distance at pymalloc with a
one-line workload written by us: a JSON round trip over 1,000 dicts and one `gc.collect()`. It
hands out about 52,000 pymalloc blocks, and interpreter start-up alone (`-S -c pass`) hands out
18,307 of them on the images of this run. This run asks the
same question with an established workload: under a fixed subset of pyperformance, the official
Python benchmark suite, does the lifetime patch change when pymalloc reuses an address? It also
measures how far two runs of one arm differ (the noise floor), so the stock-against-protected
difference can be read against it.

## Definition and boundary (unchanged)

Reuse distance counts pymalloc hand-outs between the release of a block's start address and its
next hand-out. Distances go into power-of-two buckets by bit length (bucket k holds 2^(k-1) <= d
< 2^k). The observer is `experiments/study/reuse-gap-observer.h`, compiled into obmalloc.c by
`experiments/study/reuse-hooks/cpython/obmalloc-reuse-gap-{stock,protected}.patch` behind
`-DEXP_REUSE_GAP` (hand-out hook before `pymalloc_alloc` returns, release hook in `pymalloc_free`
once the block is pymalloc's). Requests over 512 bytes go to the system allocator and are not
observed. The share of reuses in another bucket is the study's `reusediff`,
sum_k |a_k/N_a - b_k/N_b| / 2, as `collect-reuse.py` computes it.

## Workload

pyperformance 1.14.0 and pyperf 2.10.0 (the pyperf release published the same day), both from
PyPI, unchanged:

- `https://files.pythonhosted.org/packages/39/fc/793a5bf51635d90151c7a4c5ca8aad3fb5eef910321f7bad307314c4a857/pyperformance-1.14.0-py3-none-any.whl`,
  sha256 `94068ac2935b6952ccb5e8a302eb9cebc26e82454494bef6e53822738adf0049`
- `https://files.pythonhosted.org/packages/91/26/f7bd5e37c254c2671f4dfff316d123eba13663bdfee941a01b09ab02d72d/pyperf-2.10.0-py3-none-any.whl`,
  sha256 `79196bc4a11e3c926dd4c6b14c80136c6b37f884fe913cbc57037f37636e9841`

The subset, in run order, with the loop count of each benchmark (default arguments throughout):

| # | benchmark | timed call | loops |
|---|---|---|---|
| 1 | `json_dumps` | `bench_func(bench_json_dumps, data)`, cases EMPTY, SIMPLE, NESTED, HUGE | 6 |
| 2 | `json_loads` | `bench_func(bench_json_loads, objs)` | 167 |
| 3 | `richards` | `bench_func(Richards().run, 1)` | 1 |
| 4 | `deltablue` | `bench_func(delta_blue, 100)` | 9 |
| 5 | `chaos` | `bench_func(create_image_chaos, 256, 256, 5000, None, 1234)` | 1 |
| 6 | `go` | `bench_func(versus_cpu)` | 1 |
| 7 | `nbody` | `bench_time_func(bench_nbody, 'sun', 20000)` | 1 |
| 8 | `gc_collect` | `bench_time_func(benchamark_collection, 100, 20)` | 37 |

Calibration (stock image, one boot per benchmark, the driver with `--only=NAME --loops 1` and
`--loops 0`, 2026-11-15; not part of the measurement). "Per loop" is the loops=1 run minus the
loops=0 run; the loops=0 run holds the benchmark's compile and setup. Start-up alone (`-S -c
pass`) handed out 18,307 blocks in every one of the 23 calibration boots; start-up plus the
import of pyperf and the driver, with no benchmark, 153,134 blocks in 97 s.

| benchmark | blocks per loop | guest s per loop | setup blocks | setup s |
|---|---|---|---|---|
| `json_dumps` | 140,256 | 32.6 | 3,237 | 1.5 |
| `json_loads` | 6,821 | 1.2 | 5,000 | 2.5 |
| `richards` | 59,016 | 214.9 | 14,199 | 6.4 |
| `deltablue` | 14,077 | 21.5 | 19,494 | 8.7 |
| `chaos` | 230,230 | 309.4 | 73,331 | 82.5 |
| `go` | 375,839 | 777.3 | 16,484 | 7.6 |
| `nbody` | 1,259,870 | 548.3 | 5,685 | 3.3 |
| `gc_collect` | 2,317 | 5.4 | 3,111 | 1.4 |
| `unpack_sequence` (left out) | 800 | about 0 | 25,080 | 12.1 |
| `gc_traversal` (left out) | 10 | 14.1 | 283,370 | 49.2 |
| `raytrace` (left out) | not finished | more than 20 min | 17,689 | 8.5 |

Selection rule. The candidates are the JSON, object-graph, interpreter-style and GC benchmarks of
1.14.0: `json_dumps`, `json_loads`, `richards`, `deltablue`, `chaos`, `go`, `raytrace`, `nbody`,
`unpack_sequence`, `gc_collect` and `gc_traversal`. Every one is pure Python on the standard
library (pyperf is its only dependency) and needs no network, thread, subprocess or extension
module the port does not build (`_json`, `math`, `array` and `_random` are built in). A candidate
is in if its timed loop hands out pymalloc blocks and one loop fits the time budget. Left out:
`unpack_sequence` (800 blocks per loop, its loop only unpacks), `gc_traversal` (10 blocks per
loop, its containers are built once in setup) and `raytrace` (one loop runs longer than 20 minutes
under QEMU, so with the others the run would exceed 60 minutes). Loop counts: about 200 s of guest
time per benchmark in calibration, at least one loop (loops = max(1, round(200 / guest s per
loop))). Predicted run: about 4.4 million blocks and about 50 minutes of guest time.

Invocation. pyperf normally calibrates loops and spawns worker processes; a protected program
cannot exec, so the subset runs in one process. `experiments/study/reuse-hooks/cpython/pyperformance-subset.py`
runs each benchmark's own `run_benchmark.py` as `__main__` (its setup code and default arguments
unchanged) with `pyperf.Runner` replaced by a fixed-loop runner: `bench_func(name, f, *args)`
calls `f(*args)` LOOPS times and `bench_time_func(name, f, *args)` calls `f(LOOPS, *args)` once,
the calls pyperf makes for one value of LOOPS loops. There is no warm-up and no timing-driven
calibration. Between benchmarks the driver drops the benchmark's globals and calls `gc.collect()`.
The guest runs, in one boot per arm and repetition:

```
export PYTHONHASHSEED=0
PYTHONHOME=/mnt/vm/cpy ./capstone-vexec ./cpython.dom -S -c pass                              # start-up alone
PYTHONHOME=/mnt/vm/cpy ./capstone-vexec ./cpython.dom -S /mnt/vm/bench/pyperformance-subset.py  # the subset
```

`run-cpython-reuse.py --workload pyperformance --pyperformance-wheel ... --pyperf-wheel ...`
stages both wheels' files (pyperf's package, the candidates' `bm_*` directories, checked by
sha256) beside the driver and compiles them with the build's native 3.13.7, as it does the
standard library. PYTHONHASHSEED is 0 in every run. The oracle is stdout: one line per benchmark
with its deterministic return value (`richards` True, `go` its chosen move, the others none),
then `PYPERFORMANCE-OK 8`. Timings go to stderr.

The start-up process `-S -c pass` reports its own REUSE-GAP line (rep.json `startup`); the start-up
share is its hand-outs over the subset run's hand-outs. The subset run's line includes its own
start-up and the import of pyperf; the report also gives the share of start-up plus that import,
taken from the calibration run without benchmarks (153,134 blocks). The observer's table holds
2^20 distinct start addresses (error 1 when full); calibration reached at most 82,398.

## Arms, repetitions, platform

Both arms from one source snapshot, trap/eval ed8cc375c2 (libc and lifetime patches of llvm-capstone
`native-kernel-new-qemu` 7ba3f427f9), built by `build-cpython-reuse.sh`: **stock** (no lifetime
patch) and **protected** (patch 0014, `CPY_SUBLET=1`). Images: stock
`b34f8eb07ebcac63351e1e8941863c82cc49a326c63c75585c9161c8f698849d`, protected
`82edc69a6926c4be0d1f7af29e50ae93e93da30a00c15f4a01b427bee451671d`.

Two repetitions per arm (stock-1, stock-2, protected-1, protected-2), each its own boot, run in
parallel on the trap-route platform of run-3 and run-4: QEMU capstone-qemu `address-space-root`
9289114e06 (`c55fcdd2...`), kernel `Image-0133b9e21cd` (`0ea6d1ef...`), firmware `f-keep`
(`975a2421...`), launcher `capstone-vexec` (`150d9692...`), through
`runtime/virtual/run-staged.py`, with `--timeout 7200`. Results in
`~/asr-work/eval/run-5/reuse/cpython-<arm>-<rep>/` on p13, collected by `collect-reuse.py` (all
four rep.json files, both repetitions per arm).

A run is rejected if it did not complete, if the observer `error` is nonzero (the subset or the
start-up process), or if its stdout differs from the other runs'.

## Predictions

The workload is single-threaded, its inputs and random seeds are fixed (json_loads, chaos and go
seed their own generators), and the hash seed is fixed. Patch 0014 bounds each block and revokes
its lifetime on free; it does not change pymalloc's placement or free-list order.

1. **Larger than start-up:** the subset hands out at least ten times as many pymalloc blocks as
   start-up alone (start-up share at most 10 %), and at least 10^6 blocks in total.
2. **Noise floor:** stock-1 against stock-2 and protected-1 against protected-2 each give a
   `reusediff` below 0.001 (0.1 %); issue and release counts are equal within each arm.
3. **Stock against protected:** issue and release counts are equal in all four runs; the median
   bucket is equal; `reusediff` between the arms is below 0.01 (1 %) for every pair and no more
   than 0.001 above the larger noise floor.
4. **Sanity:** all four runs complete with exit status 0, observer error 0 in both processes, and
   the same stdout.

The report gives, per arm and repetition, issues, releases, reuses, distinct start addresses,
error and the output check, the start-up share, and the three `reusediff` values (noise floor
stock, noise floor protected, stock against protected).
