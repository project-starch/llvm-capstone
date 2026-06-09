# Realistic benchmark bring-up

Roadmap for adding additional Capstone PureCap benchmarks without destabilizing
the validated CoreMark baseline.

## Baseline

CoreMark PureCap is complete and validated. Keep
`capstone/tests/runtime-qemu/run-coremark.sh` as the regression gate for benchmark
work. It must continue to print CoreMark's own correctness marker:
`Correct operation validated.`

Do not treat benchmark performance scores as success criteria yet. At this
stage, success is only a deterministic correctness marker or a build artifact
that proves the next small step works.

## Source policy

- Fetch benchmark sources into `$CAPSTONE_TMP_ROOT`, which defaults to
  `/tmp/capstone` under `capstone/tests/capstone-test-env.sh`.
- Do not vendor benchmark suites into this repository.
- Do not add benchmark suites as git submodules.
- Pin fetched sources to a known commit once a benchmark becomes part of the
  repeatable test path.

## Current BEEBS milestone status

The first five tiny deterministic BEEBS benchmarks are implemented and validated.
Do not add a full BEEBS suite runner yet.

Implemented status:

- `capstone/benchmarks/beebs/fetch-beebs.sh` fetches pinned BEEBS sources from
  `https://github.com/mageec/beebs.git` into `$CAPSTONE_TMP_ROOT/beebs-src`.
- `capstone/benchmarks/beebs/build-beebs-fac-capstone.sh` builds only the `fac`
  benchmark and produces `$CAPSTONE_TMP_ROOT/beebs-build/beebs_fac_capstone.dom`.
- `capstone/benchmarks/beebs/beebs_fac_domain.c` is a minimal build-only domain
  wrapper that calls `initialise_benchmark()`, `benchmark()`, and
  `verify_benchmark()`.
- `capstone/benchmarks/beebs/run-beebs-fac.sh` builds the `fac` domain and host,
  boots QEMU, and checks the `BEEBS_RET_CORRECT` marker.
- `capstone/benchmarks/beebs/build-beebs-insertsort-capstone.sh` builds only the
  `insertsort` benchmark and produces
  `$CAPSTONE_TMP_ROOT/beebs-build/beebs_insertsort_capstone.dom`.
- `capstone/benchmarks/beebs/beebs_insertsort_domain.c` calls
  `initialise_benchmark()`, `benchmark()`, and `verify_benchmark()` and records
  only a correctness marker.
- `capstone/benchmarks/beebs/run-beebs-insertsort.sh` builds the `insertsort`
  domain and host, boots QEMU, and checks the `BEEBS_RET_CORRECT` marker.
- The `insertsort` Capstone build script generates a temporary source wrapper in
  `$CAPSTONE_TMP_ROOT/beebs-build` that replaces the upstream benchmark/init/verify
  functions with equivalent accessor-based code. The accessors recompute the global
  array capability and apply the same `CAPSTONE_DELIN` pattern used by CoreMark,
  avoiding QEMU failures from reusing a consumed gp-derived linear capability.
- `capstone/benchmarks/beebs/build-beebs-fibcall-capstone.sh` builds only the
  `fibcall` benchmark and produces
  `$CAPSTONE_TMP_ROOT/beebs-build/beebs_fibcall_capstone.dom`.
- `capstone/benchmarks/beebs/beebs_fibcall_domain.c` calls
  `initialise_benchmark()`, `benchmark()`, and `verify_benchmark()` and records
  only a correctness marker.
- `capstone/benchmarks/beebs/run-beebs-fibcall.sh` builds the `fibcall` domain
  and host, boots QEMU, and checks the `BEEBS_RET_CORRECT` marker.
- `capstone/benchmarks/beebs/build-beebs-cnt-capstone.sh` builds only the
  `cnt` benchmark and produces
  `$CAPSTONE_TMP_ROOT/beebs-build/beebs_cnt_capstone.dom`.
- `capstone/benchmarks/beebs/beebs_cnt_domain.c` calls
  `initialise_benchmark()`, `benchmark()`, and `verify_benchmark()` and records
  only a correctness marker.
- `capstone/benchmarks/beebs/run-beebs-cnt.sh` builds the `cnt` domain and host,
  boots QEMU, and checks the `BEEBS_RET_CORRECT` marker.
- The `cnt` Capstone build script generates a temporary source wrapper in
  `$CAPSTONE_TMP_ROOT/beebs-build` that preserves the upstream deterministic
  matrix benchmark behavior while recomputing/delinearizing capabilities for
  global matrix and scalar state. This keeps the workaround local to the
  benchmark path.
- `capstone/benchmarks/beebs/build-beebs-bubblesort-capstone.sh` builds only the
  `bubblesort` benchmark and produces
  `$CAPSTONE_TMP_ROOT/beebs-build/beebs_bubblesort_capstone.dom`.
- `capstone/benchmarks/beebs/beebs_bubblesort_domain.c` calls
  `initialise_benchmark()`, `benchmark()`, and `verify_benchmark()` and records
  only a correctness marker.
- `capstone/benchmarks/beebs/run-beebs-bubblesort.sh` builds the `bubblesort`
  domain and host, boots QEMU, and checks the `BEEBS_RET_CORRECT` marker.
- The `bubblesort` Capstone build script generates a temporary source wrapper in
  `$CAPSTONE_TMP_ROOT/beebs-build` that preserves the upstream deterministic
  sort behavior while recomputing/delinearizing capabilities for global array
  and scalar state. The wrapper also avoids the current direct-compile crash in
  upstream `verify_benchmark()` from lowering the local expected-array initializer
  into the backend's unresolved memcpy call path.
- Do not add a full BEEBS suite runner until several single-benchmark wrappers are
  stable.

Validation for this milestone:

- `"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone`
- `bash capstone/tests/runtime-qemu/run-coremark.sh`
- `bash capstone/benchmarks/beebs/run-beebs-fac.sh`
- `bash capstone/benchmarks/beebs/run-beebs-insertsort.sh`
- `bash capstone/benchmarks/beebs/run-beebs-fibcall.sh`
- `bash capstone/benchmarks/beebs/run-beebs-cnt.sh`
- `bash capstone/benchmarks/beebs/run-beebs-bubblesort.sh`

If the current medium thinking level becomes insufficient during benchmark
bring-up, suspend work and tell the user before switching to high thinking.

## Later milestones

- Add the next single BEEBS benchmark: `prime`, after source and
  generated-assembly inspection.
- Expand BEEBS one benchmark at a time, carrying forward only the runtime and
  compiler workarounds proven necessary by that benchmark.
- Start RV8 only after at least one BEEBS benchmark runs end to end with a stable
  build and run pattern.
- Fix remaining backend bugs only when a specific benchmark exposes them, and
  keep the focused regression for each fix.

## Current non-goals

- Do not attempt full BEEBS bring-up in one pass.
- Do not start RV8 yet.
- Do not report or optimize performance scores.
