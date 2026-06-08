# Current recommended next step

## Immediate milestone - Start BEEBS benchmark porting

**Goal**: add the first small BEEBS benchmark on the existing CoreMark-style split host/domain path.

**Why this first**: the prologue frame-lowering bug is fixed and validated. CoreMark now builds and runs with the compiled C `domain_main` wrapper instead of `coremark_domain_entry.S`, so the benchmark path no longer needs a per-domain handwritten entry point.

**Smallest useful first step**:
- choose one simple BEEBS benchmark with a tiny C workload and deterministic result,
- create a minimal build/run wrapper under `capstone/benchmarks/` or the existing runtime test structure,
- reuse the CoreMark compile/link/runtime pattern without removing the remaining backend workarounds,
- validate with the affected backend/runtime layer before expanding to more benchmarks.

**Test expectations**:
- `"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone` for backend changes,
- `bash capstone/tests/runtime-qemu/run-coremark.sh` if benchmark or backend codegen behavior changes,
- a focused BEEBS run wrapper once introduced.

## Remaining backend workarounds

The prologue bug is closed. Keep the remaining backend workarounds in place unless a focused benchmark step proves a root fix. Details: `plans/backend-compiler-fixes.md`.

## What not to regress

Do not delete `capstone/caplifive-buildroot/build/local.mk` - its absence silently switches the image to stock OpenSBI and breaks all runtime proofs.
