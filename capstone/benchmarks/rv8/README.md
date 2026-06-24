# RV8 benchmark suite (Capstone PureCap domain bring-up)

Third of the three benchmark suites (CoreMark ✓, BEEBS ✓, **RV8**) before the
SQLite / real-software stage.

- Upstream: **https://github.com/michaeljclark/rv8-bench** (mirror `rv8-io/rv8-bench`),
  MIT. "Integer-centric benchmarks for regression testing of the rv8 binary
  translator." Pinned commit in `fetch-rv8.sh`.
- Benchmarks (single `.c` each under `src/`): `qsort`, `aes`, `norx`,
  `dhrystone`, `primes`, `miniz`, `sha512`, plus `bigint.cc` (C++, deferred).

## Layout (mirrors CoreMark's split layout)

- In-tree orchestration: `capstone/benchmarks/rv8/`.
- Fetched upstream: `/tmp/capstone/rv8-src` (`fetch-rv8.sh`; not a submodule).
- Build outputs: `/tmp/capstone/rv8-build`.

## Approach

rv8-bench programs are hosted-Linux style (`main()`, `printf`, `malloc`,
`gettimeofday` timing, large workloads). Each is adapted to the Capstone domain
model — no stdio, bounded runtime-provided memory, result via an oracle — by:

- stripping the hosted includes and `-include`-ing `adapted/rv8_capstone_preamble.h`;
- stubbing irrelevant hosted services (`gettimeofday`/`printf`/`exit`) in
  `adapted/rv8_stubs.c`;
- providing a 16-byte-aligned bump allocator `adapted/rv8_malloc.c` (records can
  hold capability fields, so allocations must be 16-aligned — the dtoa lesson);
- reusing the BEEBS freestanding libc (`beebs_freestanding_string.c`,
  `beebs_softfloat_libm.c`, soft-float builtins) and the domain harness
  (`beebs_simple_domain.c`, `my_first_domain/{start.S,link.ld}`);
- an adapted oracle tail providing `initialise_benchmark`/`benchmark`/
  `verify_benchmark` that runs the kernel (timing/printf no-op) and checks a
  deterministic result. Heavy workloads are shrunk (and oracles recomputed) to
  fit domain memory.

## Status

| Benchmark | Status | Notes |
|-----------|--------|-------|
| dhrystone | **PASS** | `run-rv8-dhrystone.sh` → `__RV8_DHRYSTONE_PASSED__`. LOOPS pinned to 100000; oracle = canonical Dhrystone end-state self-check (`IntGlob==5`, `BoolGlob==1`, `Char1Glob=='A'`, `Char2Glob=='B'`, `Array1Glob[8]==7`, `Array2Glob[8][7]==LOOPS+10`). Self-contained, confirmed against a native gcc reference. |
| qsort | **PASS** | `run-rv8-qsort.sh` → `__RV8_QSORT_PASSED__`. Upstream ships its own in-place BSD qsort; main()'s 200 MB array replaced by a static `int[8192]` filled with the same recurrence. Oracle: sorted non-decreasing **and** element-sum preserved (permutation invariant) — self-contained. |
| aes, norx, sha512, primes, miniz | TODO | Increasing difficulty; `primes`/`miniz` need workload shrink (domain memory). `primes` needs `sqrt` (shared libm). crypto (`aes`/`sha512`/`norx`) → known fixed-vector digests/ciphertext. |
| bigint | DEFERRED | C++ — needs C++ runtime/ABI assessment. |

## Run

```bash
source capstone/tests/capstone-test-env.sh
bash capstone/benchmarks/rv8/run-rv8-dhrystone.sh   # __RV8_DHRYSTONE_PASSED__
```
