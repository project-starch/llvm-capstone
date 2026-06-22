# Capstone soft-float + libm runtime (FP benchmark bring-up)

## Status

In place and validated by `cubic`, the first floating-point BEEBS benchmark
(`run-beebs-cubic.sh` → `__BEEBS_CUBIC_PASSED__`). This is the reusable
foundation for the remaining FP-blocked benchmarks.

## Problem

Capstone PureCap has no floating-point runtime in the bare-metal domain, and the
backend did not support FP libcalls at all:

1. **Empty libcall-name table.** `RuntimeLibcalls.td` gated the system library on
   `isRISCV` (`TT.isRISCV()`), false for `Triple::capstone64`. So `getLibcallName`
   returned null and *any* FP operation needing a libcall aborted in
   `TargetLowering::makeLibCall` ("unsupported library call operation",
   `TargetLowering.cpp:189`) — including soft-float arithmetic *and* libm
   transcendentals, even with `+f +d` hardware float.
2. **fp128 constants.** `long double` is `fp128` on Capstone. Type legalization
   softens an `fp128` ConstantFP into an arbitrary 128-bit *integer* constant,
   which instruction selection rejects ("capabilities are unforgeable").
3. **No libm / soft-float library** to link into the freestanding domain.

## Backend changes (committed in the LLVM tree)

- **`llvm/include/llvm/IR/RuntimeLibcalls.td`** — add `isCapstone`/`isCapstone64`
  predicates and `CapstoneSystemLibrary` (mirrors `RISCVSystemLibrary`, minus the
  RISC-V-specific `__riscv_flush_icache`). Additive and Capstone-gated; other
  targets' generated tables are unaffected. `Triple::isCapstone*()` already
  existed in `Triple.h`.
- **`llvm/lib/Target/Capstone/CapstoneISelLowering.cpp`** — a pre-legalize
  `ISD::ConstantFP` DAG combine (registered via `setTargetDAGCombine`) rewrites an
  `fp128` constant into a constant-pool load *before* type legalization softens
  it: `LOAD f128 (ConstantPool)` → softens to `LOAD i128` → selected as `ldc` of
  plain rodata (tag 0; only consumed by soft-float libcalls, never dereferenced
  as a capability). The genuine capability-forge path (`inttoptr` of a wide
  integer) has no `ConstantFP` and still hits the hard "unforgeable" error.
- **Coverage**: `llvm/test/CodeGen/Capstone/fp-libcall.ll` (named libcalls +
  fp128-constant `ldc`); `cap-constants-invalid.ll` still asserts the forge guard.

## In-domain runtime (per-benchmark build, under `capstone/benchmarks/beebs`)

- **Soft-float, not hardware float.** Compiling without `+f +d` keeps all FP in
  integer GPRs via compiler-rt builtins, so there is no FP register state to
  preserve across the host-call/domain-switch boundary. The needed double
  builtins compile cleanly for `capstone64`:
  `adddf3 subdf3 muldf3 divdf3 fixdfsi floatsidf comparedf2 fp_mode`
  (from `compiler-rt/lib/builtins`). `fp_mode.c` supplies `__fe_getround` /
  `__fe_raise_inexact`.
- **Avoid fp128 quad soft-float.** compiler-rt's quad routines (e.g. `divtf3.c`)
  do *not* compile for Capstone — they hit the i128 non-vector-shift legalization
  limit (deferred Bug #3). Benchmarks that use `long double` only for intermediate
  precision are adapted to `double` (documented source change). A general fp128
  path would require fixing i128 shifts first.
- **libm**: `adapted/beebs_cubic_libm.c` — a compact, self-contained,
  deterministic double libm (`fabs`, `sqrt` Newton, `exp`/`log` with Cody-Waite
  reduction, `pow = exp(y·log x)`, `sin`/`cos` fdlibm kernels, `acos` fdlibm
  rational). Validated to <1e-12 (cos/acos/sqrt) / ~7e-9 (cbrt) against the system
  long-double libm via the built-in `-DCUBIC_LIBM_TEST` harness. Reuse/extend this
  file for other FP benchmarks.

## Verification pattern

FP benchmarks whose upstream `verify_benchmark` returns -1 should verify against
a known oracle, not a bit-exact host reference (hand-rolled libm differs from
system libm by ULPs). `cubic` checks the documented polynomials' exact roots
within 1e-4 (`adapted/beebs_cubic_capstone_tail.c`).

## Next FP targets

`dtoa` (needs libc: malloc/memcpy/errno/float.h/fenv.h/locale + log/floor/ceil),
then `nbody`, `minver`, `ludcmp`, `qsort`, `select`, `sqrt`, `qurt`, `fasta`,
`frac`, `st`, `whetstone`, `newlib-*`, `matmult-int`. All now compile; each needs
its libm closure linked and a correctness oracle. A larger general-purpose fp128
path additionally needs the i128 non-vector-shift backend fix (Bug #3).
