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

## Next milestone: one BEEBS benchmark

Start with a single tiny deterministic BEEBS benchmark, preferably `fac` unless
local investigation shows another benchmark is simpler for the current runtime.
Do not add a full BEEBS suite runner in the first pass.

Initial skeleton status: implemented for `fac`.

- `capstone/benchmarks/beebs/fetch-beebs.sh` fetches pinned BEEBS sources from
  `https://github.com/mageec/beebs.git` into `$CAPSTONE_TMP_ROOT/beebs-src`.
- `capstone/benchmarks/beebs/build-beebs-fac-capstone.sh` builds only the `fac`
  benchmark and produces `$CAPSTONE_TMP_ROOT/beebs-build/beebs_fac_capstone.dom`.
- `capstone/benchmarks/beebs/beebs_fac_domain.c` is a minimal build-only domain
  wrapper that calls `initialise_benchmark()`, `benchmark()`, and
  `verify_benchmark()`.
- Do not add a full BEEBS suite runner until this one-benchmark pattern has a
  matching runtime wrapper.

Validation for this milestone:

- `"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone`
- `bash capstone/tests/runtime-qemu/run-coremark.sh`
- the BEEBS build script produces the selected benchmark `.dom`

## Later milestones

- Add a minimal BEEBS host/runtime wrapper and `run-beebs.sh`.
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
