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

The first fourteen tiny deterministic BEEBS benchmarks are implemented and validated.
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
- `capstone/benchmarks/beebs/build-beebs-prime-capstone.sh` builds only the
  `prime` benchmark and produces
  `$CAPSTONE_TMP_ROOT/beebs-build/beebs_prime_capstone.dom`.
- `capstone/benchmarks/beebs/beebs_prime_domain.c` calls
  `initialise_benchmark()`, `benchmark()`, and `verify_benchmark()` and records
  only a correctness marker.
- `capstone/benchmarks/beebs/run-beebs-prime.sh` builds the `prime` domain and
  host, boots QEMU, and checks the `BEEBS_RET_CORRECT` marker.
- The `prime` Capstone build script generates a temporary source wrapper in
  `$CAPSTONE_TMP_ROOT/beebs-build` that preserves the upstream deterministic
  prime-check behavior while recomputing/delinearizing capabilities for scalar
  global state. This adds modulo/division coverage through the benchmark's
  `m % n` path.
- `capstone/benchmarks/beebs/build-beebs-recursion-capstone.sh` builds only the
  `recursion` benchmark and produces
  `$CAPSTONE_TMP_ROOT/beebs-build/beebs_recursion_capstone.dom`.
- `capstone/benchmarks/beebs/beebs_recursion_domain.c` calls
  `initialise_benchmark()`, `benchmark()`, and `verify_benchmark()` and records
  only a correctness marker.
- `capstone/benchmarks/beebs/run-beebs-recursion.sh` builds the `recursion`
  domain and host, boots QEMU, and checks the `BEEBS_RET_CORRECT` marker.
- The `recursion` Capstone build script generates a temporary source wrapper in
  `$CAPSTONE_TMP_ROOT/beebs-build` that preserves the upstream deterministic
  recursive Fibonacci behavior while recomputing/delinearizing capabilities for
  scalar global state.
- `capstone/benchmarks/beebs/build-beebs-janne-complex-capstone.sh` builds only
  the `janne_complex` benchmark and produces
  `$CAPSTONE_TMP_ROOT/beebs-build/beebs_janne_complex_capstone.dom`.
- `capstone/benchmarks/beebs/beebs_janne_complex_domain.c` calls
  `initialise_benchmark()`, `benchmark()`, and `verify_benchmark()` and records
  only a correctness marker.
- `capstone/benchmarks/beebs/run-beebs-janne-complex.sh` builds the
  `janne_complex` domain and host, boots QEMU, and checks the
  `BEEBS_RET_CORRECT` marker.
- The `janne_complex` Capstone build script generates a temporary source wrapper
  in `$CAPSTONE_TMP_ROOT/beebs-build` that preserves the upstream deterministic
  nested-loop behavior while recomputing/delinearizing capabilities for scalar
  global state.
- `capstone/benchmarks/beebs/build-beebs-tarai-capstone.sh` builds only the
  `tarai` benchmark and produces
  `$CAPSTONE_TMP_ROOT/beebs-build/beebs_tarai_capstone.dom`.
- `capstone/benchmarks/beebs/beebs_tarai_domain.c` calls
  `initialise_benchmark()`, `benchmark()`, and `verify_benchmark()` and records
  only a correctness marker.
- `capstone/benchmarks/beebs/run-beebs-tarai.sh` builds the `tarai` domain and
  host, boots QEMU, and checks the `BEEBS_RET_CORRECT` marker.
- The `tarai` Capstone build script generates a temporary source wrapper in
  `$CAPSTONE_TMP_ROOT/beebs-build` that preserves the upstream deterministic
  recursive behavior while recomputing/delinearizing capabilities for scalar
  global state.
- `capstone/benchmarks/beebs/build-beebs-cover-capstone.sh` builds only the
  `cover` benchmark and produces
  `$CAPSTONE_TMP_ROOT/beebs-build/beebs_cover_capstone.dom`.
- `capstone/benchmarks/beebs/beebs_cover_domain.c` calls
  `initialise_benchmark()`, `benchmark()`, and `verify_benchmark()` and records
  only a correctness marker.
- `capstone/benchmarks/beebs/run-beebs-cover.sh` builds the `cover` domain and
  host, boots QEMU, and checks the `BEEBS_RET_CORRECT` marker.
- The `cover` Capstone build script compiles the upstream source directly but
  uses `-fno-jump-tables`. Direct dense-switch jump-table lowering currently
  emits scalar table loads and a scalar indirect jump; the runtime trips
  `Cap mem access requires capability` before the correctness marker. The
  flag keeps this benchmark as deterministic switch/control-flow coverage
  without relying on the unsupported jump-table form.
- `capstone/benchmarks/beebs/build-beebs-duff-capstone.sh` builds only the
  `duff` benchmark and produces
  `$CAPSTONE_TMP_ROOT/beebs-build/beebs_duff_capstone.dom`.
- `capstone/benchmarks/beebs/beebs_duff_domain.c` calls
  `initialise_benchmark()`, `benchmark()`, and `verify_benchmark()` and records
  only a correctness marker.
- `capstone/benchmarks/beebs/run-beebs-duff.sh` builds the `duff` domain and
  host, boots QEMU, and checks the `BEEBS_RET_CORRECT` marker.
- The `duff` Capstone build script generates a temporary source wrapper in
  `$CAPSTONE_TMP_ROOT/beebs-build` that preserves the upstream Duff's-device
  byte-copy behavior while recomputing/delinearizing capabilities for the
  `source` and `target` byte arrays. It also uses `long` loop/index values in
  the accessor path to avoid the current backend selection gap for `i32` stack
  reloads used as capability offsets, and keeps `-fno-jump-tables` because this
  benchmark contains a switch.
- `capstone/benchmarks/beebs/build-beebs-levenshtein-capstone.sh` builds only the
  `levenshtein` benchmark and produces
  `$CAPSTONE_TMP_ROOT/beebs-build/beebs_levenshtein_capstone.dom`.
- `capstone/benchmarks/beebs/beebs_levenshtein_domain.c` calls
  `initialise_benchmark()`, `benchmark()`, and `verify_benchmark()` and records
  only a correctness marker.
- `capstone/benchmarks/beebs/run-beebs-levenshtein.sh` builds the `levenshtein`
  domain and host, boots QEMU, and checks the `BEEBS_RET_CORRECT` marker.
- The `levenshtein` Capstone build script generates a temporary source wrapper in
  `$CAPSTONE_TMP_ROOT/beebs-build` that preserves the upstream deterministic
  string-distance sum while avoiding hosted `strlen` and global pointer-table
  assumptions. It uses fixed string globals selected by index, a local strlen,
  a fixed-size flat DP table, and `long` loop/index values in address paths to
  avoid the current backend selection gap for `i32` stack reloads used as
  capability offsets.
- `capstone/benchmarks/beebs/build-beebs-jfdctint-capstone.sh` builds only the
  `jfdctint` benchmark and produces
  `$CAPSTONE_TMP_ROOT/beebs-build/beebs_jfdctint_capstone.dom`.
- `capstone/benchmarks/beebs/beebs_jfdctint_domain.c` calls
  `initialise_benchmark()`, `benchmark()`, and `verify_benchmark()` and records
  only a correctness marker.
- `capstone/benchmarks/beebs/run-beebs-jfdctint.sh` builds the `jfdctint`
  domain and host, boots QEMU, and checks the `BEEBS_RET_CORRECT` marker.
- The `jfdctint` Capstone build script generates a temporary patched source in
  `$CAPSTONE_TMP_ROOT/beebs-build` that preserves the upstream fixed-point
  integer DCT body and rewrites only `verify_benchmark()`. The local expected
  array initializer in upstream `verify_benchmark()` lowered into the current
  backend's unsupported `memcpy` call path; the patched verifier uses scalar
  expected-value checks and keeps success based on the upstream result vector.
- `capstone/benchmarks/beebs/build-beebs-fdct-capstone.sh` builds only the
  `fdct` benchmark and produces
  `$CAPSTONE_TMP_ROOT/beebs-build/beebs_fdct_capstone.dom`.
- `capstone/benchmarks/beebs/beebs_fdct_domain.c` calls
  `initialise_benchmark()`, `benchmark()`, and `verify_benchmark()` and records
  only a correctness marker.
- `capstone/benchmarks/beebs/run-beebs-fdct.sh` builds the `fdct` domain and
  host, boots QEMU, and checks the `BEEBS_RET_CORRECT` marker.
- The `fdct` Capstone build script generates a temporary patched source in
  `$CAPSTONE_TMP_ROOT/beebs-build` that preserves the upstream fixed-point DCT
  kernel but rewrites the benchmark and verifier copy/compare paths. The wrapper
  avoids hosted `memcpy`/`memcmp` and recomputes/delinearizes capabilities for
  the global input, working, and expected-result arrays before passing the
  working block into the upstream DCT kernel.
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
- `bash capstone/benchmarks/beebs/run-beebs-prime.sh`
- `bash capstone/benchmarks/beebs/run-beebs-recursion.sh`
- `bash capstone/benchmarks/beebs/run-beebs-janne-complex.sh`
- `bash capstone/benchmarks/beebs/run-beebs-tarai.sh`
- `bash capstone/benchmarks/beebs/run-beebs-cover.sh`
- `bash capstone/benchmarks/beebs/run-beebs-duff.sh`
- `bash capstone/benchmarks/beebs/run-beebs-levenshtein.sh`
- `bash capstone/benchmarks/beebs/run-beebs-jfdctint.sh`
- `bash capstone/benchmarks/beebs/run-beebs-fdct.sh`

During the `levenshtein` milestone, the first full sequential BEEBS run passed
through `recursion` and then hit a QEMU login prompt timeout before executing
`janne_complex`. A targeted rerun of `janne_complex`, `tarai`, `cover`, `duff`,
and `levenshtein` passed. Treat that as harness/login flakiness, not a benchmark
correctness failure.

If the current medium thinking level becomes insufficient during benchmark
bring-up, suspend work and tell the user before switching to high thinking.

## Later milestones

- Add the next single BEEBS benchmark after source and generated-assembly
  inspection. The recommended next candidate is `strstr`: it is deterministic,
  has a real verifier, adds compact string-search coverage, and avoids the
  floating-point hazards in `frac`/`sqrt` plus the no-verifier status of
  `bs`, `fir`, `select`, and similar benchmarks. Expect a benchmark-local
  wrapper if direct compile/runtime exposes scalar or stale capabilities for
  global string pointers and string literals.
- Expand BEEBS one benchmark at a time, carrying forward only the runtime and
  compiler workarounds proven necessary by that benchmark.
- Start RV8 only after at least one BEEBS benchmark runs end to end with a stable
  build and run pattern.
- Fix remaining backend bugs only when a specific benchmark exposes them, and
  keep the focused regression for each fix.

## Runtime harness note

`capstone/tests/runtime-qemu/run-domain-smoke.py` runs QEMU with `-snapshot` so
runtime tests do not dirty or corrupt the generated Buildroot `rootfs.ext2` image.
The Buildroot guest getty is pinned to `ttyS0` for deterministic serial login.
The runtime harness also forces QEMU `-smp 1` to avoid intermittent boot stalls.

## Current non-goals

- Do not attempt full BEEBS bring-up in one pass.
- Do not start RV8 yet.
- Do not report or optimize performance scores.
