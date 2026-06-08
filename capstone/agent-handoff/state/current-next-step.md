# Current recommended next step

## Immediate milestone — Fix prologue frame lowering bug

**File**: `llvm/lib/Target/Capstone/CapstoneFrameLowering.cpp`

**Change**: Prologue currently emits `cincoffsetimm s0, sp, -N` (rd≠rs1), which consumes
`sp` (LINEAR capability). All subsequent `ldc`/`stc` using `sp` as base crash because
`sp.tag=0` after the first use. Fix: emit `cincoffsetimm sp, sp, -N` (rd==rs1, in-place
update, no consumption).

**Why this first**: This is the only backend bug that requires a **per-domain hand-written
assembly entry point**. Without this fix, every BEEBS and RV8 benchmark needs its own
`*_entry.S` file. The other 4 bugs are compile-flag workarounds that scale via build scripts.

**Test after fix**:
- CoreMark should compile and link without `coremark_domain_entry.S`.
- `run-coremark.sh` must still pass ("Correct operation validated.").
- Run: `"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone` — no regressions.

## After the prologue fix — benchmark porting sequence

1. **BEEBS** (https://github.com/mageec/beebs) — 20+ small embedded benchmarks.
   Build script pattern reuses the CoreMark approach directly.
   Each passing benchmark validates more of the compiled domain code path.

2. **RV8** (https://github.com/larkmjc/rv8-bench) — RISC-V performance benchmarks
   (dhrystone, memcpy, primes, qsort, sieve, ...). Port after BEEBS.

3. **Fix remaining backend bugs** only as they block specific benchmark programs,
   not speculatively. See `plans/backend-compiler-fixes.md` for the full catalog.

## What not to regress

Do not delete `capstone/caplifive-buildroot/build/local.mk` — its absence silently
switches the image to stock OpenSBI and breaks all runtime proofs.
