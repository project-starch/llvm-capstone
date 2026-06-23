# Current recommended next step

## Current BEEBS milestone - 71 benchmarks validated

71 BEEBS benchmarks now pass end-to-end. The most recent additions are `qsort`,
`qurt`, and `select` (`run-beebs-{qsort,qurt,select}.sh`) — FP benchmarks needing
only the soft-float builtins (no libm; they ship their own helpers or use only
float compares), each with an adapted oracle tail (upstream verifiers return -1):
`qsort` checks the sorted region is monotonic; `qurt` checks the last input's
known complex roots (2±2i, tolerance — it uses its own approximate sqrt);
`select` widens `arr` to [21] (fixing a latent 1-indexed over-read) and compares
the captured k-th return against a host reference. No compiler change.

The prior batch `frac`/`st`/`nbody` (`run-beebs-{frac,st,nbody}.sh`) reuses the
shared libm; `st`/`nbody` drove the correctly-rounded `sqrt`.

Two reusability changes: the libm is now the neutrally-named, shared
`adapted/beebs_softfloat_libm.c` (was `beebs_cubic_libm.c`), and its `sqrt` is now
**correctly-rounded** (Newton seed + exact two-product residual + round-to-
nearest-even; bit-exact vs the host over 230M values). The correctly-rounded sqrt
is required by benchmarks that compare results for **exact** equality (`st`,
`nbody`); `frac` needs only `fabs`. `cubic` re-verified after the sqrt change.
`ludcmp`/`minver` (prior additions) reuse the soft-float builtins only.

A runtime trace (instrumenting `helper_cscincoffset`, since reverted) showed the
earlier `ludcmp` `cscincoffset rs1->tag` crash was **not** a `cincoffset`
operand/canonicalization bug: the matrix algorithm runs fine. It is the
documented **Bug #9** (a `verify_benchmark` *local* const-initialized array,
`float exp_a[8][9]={...}`, lowered to a `memcpy` from `.rodata` into a stack array
whose destination capability comes back untagged). Workaround (source, no compiler
change): mark `exp_a`/`exp_b`/`exp_x` `static const` so they live in `.rodata`
(no stack copy, no `memcpy`) — same class of fix as mergesort / nettle-*. `minver`
needed only a correctness oracle (its upstream verify returns -1): an FNV-1a
checksum of the inverted matrix `a_i` + `det` vs a native float reference.
The **Bug #9 backend root cause** (untagged stack dest in a rodata→stack copy)
remains an open, deferrable backend task — see `plans/beebs-deferred-benchmarks.md`.

`sqrt` (63rd) is validated with `run-beebs-sqrt.sh` — the first FP benchmark to
*reuse* the soft-float runtime (no compiler change). It needs no libm (it ships its own
`sqrtfcn`) and has a real `verify_benchmark`; the only new infrastructure is the
shared `build-beebs-softfloat-common.sh` helper, which compiles the compiler-rt
float+double soft-float builtin set and is now also sourced by `cubic`.

`cubic` (62nd) is the **first floating-point benchmark**, validated with
`run-beebs-cubic.sh`.

`cubic` required standing up a soft-float + libm runtime (see
`design/capstone-softfloat-libm.md`). Two backend changes:
(1) `CapstoneSystemLibrary` in `RuntimeLibcalls.td` registers the runtime
libcall-name table (FP libcalls previously aborted at `TargetLowering.cpp:189`
with "unsupported library call operation" because the table was empty);
(2) a pre-legalize `ISD::ConstantFP` DAG combine in `CapstoneISelLowering.cpp`
loads fp128 constants from the constant pool (`ldc`) instead of softening them
into an unforgeable 128-bit capability immediate. The genuine capability-forge
guard (`inttoptr` of a wide integer) is unchanged (`cap-constants-invalid.ll`).
Runtime: `SolveCubic`'s `long double` is reduced to `double` (documented source
adaptation — avoids fp128 quad soft-float, which would also need an i128
non-vector-shift backend fix); doubles use compiler-rt soft-float builtins; a
compact self-contained `adapted/beebs_cubic_libm.c` provides
`fabs/sqrt/exp/log/pow/sin/cos/acos` (validated <1e-12 vs system libm). Verified
against the exact mathematical roots {2, 2.5, 6} and {2.5}.

`compress` (61st) is validated with `run-beebs-compress.sh`: pure-integer, no
compiler change, FNV-1a checksum of the LZW work product
(`in_count`/`out_count`/`free_ent` + `htab`/`codetab`) vs a native LP64 host
reference. Its historically documented "backend crash" was already stale.

`compress` no longer crashes the backend (the historically documented
"pre-existing backend crash" was resolved by intervening backend fixes); it is
a pure-integer source adaptation with no compiler change. Its upstream
`verify_benchmark` returns -1 ("no verification") and this BEEBS variant never
calls `output()`, so `comp_text_buffer`/`bytes_out` stay empty. The adapted tail
(`adapted/beebs_compress_capstone_tail.c`) instead checksums the LZW work
product (`in_count`/`out_count`/`free_ent` + `htab`/`codetab`) with FNV-1a
against a native LP64 host reference — exercising capability-mode array indexing
as a real correctness gate.

## Recent root fixes

Narrow truncating stores from i128 carrier to capability-addressed memory
are fixed:

- `selectLDC_STC` in `CapstoneISelDAGToDAG.cpp` now handles `MemVT = i32/i16/i8`
  truncating stores by emitting `SW`/`SH`/`SB` respectively.  The large-offset
  CIncOffset decomposition is also extended to cover SW/SH/SB.
- This arises when a pointer-difference result (i64 in an i128 any_extend carrier)
  is stored into a narrower integer field (`int len = ptr1 - ptr2`).
- `slre` is the proof benchmark.

The large-offset capability load/store backend blocker is fixed:

- `selectLDC_STC` in `CapstoneISelDAGToDAG.cpp` now handles constant offsets
  > 2047 for `ldc`/`stc` by emitting `CIncOffset(base, offset)` then
  `ldc/stc rd, 0(adjusted)`.
- `sglib-rbtree` was the proof benchmark: its iterator struct pushes
  `equalto` and `subcomparator` past the 2047-byte immediate range.

Pointer arithmetic fixes now cover:

- `ptr - integer` and `ptr + (-offset)` as `cincoffset base, -offset`.
- True `ptr - ptr` by extracting both capability cursors with `lcc ..., 2`,
  subtracting the XLEN cursor values, and sign-extending back to the `i128`
  carrier when needed.
- C pointer subtraction in Clang CodeGen truncates the result to C `ptrdiff_t`
  when the target pointer integer type is wider. This avoids ICmp type
  mismatches when comparing pointer differences on Capstone.
- InstCombine `or disjoint` pointer-carrier additions are lowered as capability
  offset arithmetic when one operand is a known capability base and the other is
  a known integer offset.

Constant-pool lowering for large scalar constants is fixed:

- `lowerConstant` now emits `LOAD:i64(LGA:i128(TargetConstantPool))` directly
  when a large i64 constant is placed in the constant pool, so the final load
  has a capability base rather than a raw integer constant-pool address.
- `lowerConstantPool` returns the capability address (`LGA:i128`) like
  `lowerGlobalAddress`.

`qrduino` and `miniz` both needed benchmark-local source adaptations for static
string pointer data: keep string literals as arrays so benchmark code derives
capabilities from the global symbol rather than loading untagged raw pointers
from data.

`miniz` additionally uses generated scratch sources under
`$CAPSTONE_TMP_ROOT/beebs-build`, strips hosted includes, provides inline libc
stubs, expands/aligned its bump heap, and rounds allocations to 16 bytes.

Verified gates for this milestone:

```bash
source capstone/tests/capstone-test-env.sh
"$CAPSTONE_LLVM_LIT" -sv clang/test/CodeGen/cap-ptr-compare.c
"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone
bash capstone/tests/runtime-qemu/run-coremark.sh
bash capstone/benchmarks/beebs/run-beebs-ctl-string.sh
bash capstone/benchmarks/beebs/run-beebs-qrduino.sh
bash capstone/benchmarks/beebs/run-beebs-sglib-rbtree.sh
bash capstone/benchmarks/beebs/run-beebs-miniz.sh
bash capstone/benchmarks/beebs/run-beebs-slre.sh
bash capstone/benchmarks/beebs/run-beebs-wikisort.sh
bash capstone/benchmarks/beebs/run-beebs-trio-sscanf.sh
```

## Remaining viable targets

No clean-add BEEBS target is known. Pick remaining targets only if you are ready
to fix a root issue or carry an invasive source adaptation.

Good next investigations:

- `trio`/`trio-snprintf`: the `va_list` capability storage/copying blocker is now
  **fixed** in the backend (`va_start`/`va_arg`/`va_copy` lower with `stc`/`ldc`
  and a 16-byte `cincoffset` stride; see `plans/backend-compiler-fixes.md`).
  `trio-sscanf` is validated with an embedded/minimal string-helper build.
  Full `trio` and `trio-snprintf` still need a deliberate soft-float/complex
  format-lib strategy; `trio-snprintf` also has `verify_benchmark = -1`, so do
  not add it as a normal correctness gate without changing the verifier story.
- FP-blocked benchmarks: require a deliberate soft-float/libcall strategy for
  Capstone, not one-off wrappers.

## Blocked (do not retry without root fix)

### FP-blocked: needs in-domain libm + libc (dtoa)

- `cubic`: **RESOLVED** (first FP benchmark). The runtime libcall-name table is
  now registered and an in-domain soft-float + libm runtime exists; see the
  milestone note above and `design/capstone-softfloat-libm.md`.
- `sqrt`: **RESOLVED** — pure soft-float (own `sqrtfcn`, no libm), real verify.
  Reuses `build-beebs-softfloat-common.sh`. `run-beebs-sqrt.sh`.
- `ludcmp`, `minver`: **RESOLVED** (Bug #9 source workaround / oracle, see the
  milestone note above). The earlier "cincoffset bug" hypothesis was disproven by
  a runtime trace; the matrix algorithm's `cincoffset`s are correct.
- `dtoa`: now compiles (libcall names resolve), but the bare-metal domain still
  lacks the libm/libc it needs — `log`/`floor`/`ceil` plus `malloc`,
  `memcpy`/`memmove`/`memset`, `strcpy`/`strlen`, `errno`, and freestanding
  `float.h`/`fenv.h`/`locale` shims (89 KB FP↔decimal library). The `cubic`
  soft-float runtime is reusable; `dtoa` mainly adds the libc surface. Larger
  follow-on. See `plans/beebs-deferred-benchmarks.md` (Bug #14).
- The soft-float runtime continues to unblock the remaining FP benchmarks below
  at the *compile* level; each still needs its libm closure linked + a
  correctness oracle (and exact-comparison verifiers need the correctly-rounded
  `sqrt`, now in place).

### Remaining FP benchmarks (compile via soft-float; not yet brought up)

- `whetstone` - needs `atan` (add to the libm) + an oracle; also its
  transcendentals would need to be bit-exact for an exact-comparison verifier.
- `fasta` - needs libc (`memcpy`/`strlen`/`malloc`-ish).
- `newlib-exp`, `newlib-log`, `newlib-mod`, `newlib-sqrt`, `stb_perlin` - math lib.
- `matmult-int` - hits the i128 non-vector-shift backend limit (Bug #3).
- `trio`, `trio-snprintf` - float / complex format lib.
- `dtoa` - heavy libc (`malloc`/`errno`/`float.h`/`fenv.h`/`locale`) + libm.

## Regression gate for backend/lowering/ABI changes

For non-trivial backend, lowering, ABI, or broad benchmark-runtime changes, do
not treat the change as fully validated until this full gate passes:

```bash
source capstone/tests/capstone-test-env.sh
"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone
bash capstone/tests/runtime-qemu/run-coremark.sh
bash capstone/benchmarks/beebs/run-all-beebs.sh
```

Smaller BEEBS subsets are still useful for narrow wrapper/doc changes and quick
pre-commit smoke checks, but they are not the full backend validation gate.

For runtime/HostCall changes, use `capstone/tests/runtime-qemu/run-hostcall-all.sh`
as the normal proof gate. For OpenSBI/kernel/module changes, use
`capstone/tests/runtime-qemu/run-nullblk-all.sh`. Individual wrappers remain the
right entry points for focused reruns and diagnosis.

## Known backend limitations (document when encountered)

- **memcpy/memmove/memset libcall**: the Capstone backend crashes with null
  symbol name when generating calls to these. Always provide inline stubs
  instead.
- **cincoffset commutative bug**: fixed in lowerADD (isIntegerOffset now covers
  scaled-index GEP; isCapabilityValue distinguishes genuine ldc loads from
  sextloads). edn was the last benchmark blocked by this.

## What not to regress

Do not delete `capstone/caplifive-buildroot/build/local.mk` - its absence
silently switches the image to stock OpenSBI and breaks all runtime proofs.
