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
| sha512 | **PASS** | `run-rv8-sha512.sh` → `__RV8_SHA512_PASSED__`. Self-contained SHA-512 (no malloc); kept `<stdint.h>` (freestanding), stubbed `assert`. Rounds reduced to 1000 (×64 zero bytes); oracle = the 64-byte digest vs a native gcc reference for the same input. |
| aes | **PASS** | `run-rv8-aes.sh` → `__RV8_AES_PASSED__`. Standard Rijndael; small malloc'd round-key context only. Oracle = FIPS-197 AES-128 known-answer (key `00..0F`, pt `0011..FF` → ct `69C4E0D8..C55A`) **and** encrypt/decrypt round-trip. Self-contained. |
| norx, primes, miniz | TODO | `primes`/`miniz` need workload shrink (domain memory); `primes` needs `sqrt` (shared libm). `norx` → known fixed-vector ciphertext. |
| bigint | DEFERRED | C++ — needs C++ runtime/ABI assessment. |

## Run

```bash
source capstone/tests/capstone-test-env.sh
bash capstone/benchmarks/rv8/run-rv8-dhrystone.sh   # __RV8_DHRYSTONE_PASSED__
```
