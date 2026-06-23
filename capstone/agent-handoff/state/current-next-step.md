# Current recommended next step

## Current BEEBS milestone - 80 benchmarks validated

### Recent backend work (2026-06-24): Bug #3 fixed; capability globals tagged

`Bug #3` (i128 non-vector-shift legalization assertion) is **fixed in the
backend** — `lowerScalarI128Shift` now has a general constant-shift fallback for
operands the narrowing helper can't recognize (notably the `ashr/lshr i128` a
pointer-difference `(p-q)/sizeof(T)` lowers to). Validated by a domain probe + the
`matmult-int` repro + new lit coverage in `i128-xlen-lowering.ll`.

**Capability-global tagging is resolved** (constructor-codegen). A capability tag
cannot live in a static ELF image, so initialized capability globals (pointer
tables, string tables like `dtoa`'s `char *nums[]`, function-pointer tables)
loaded **untagged** and faulted on first use. New IR ModulePass
`llvm/lib/Target/Capstone/CapstoneCapGlobalInit.cpp` synthesizes a per-module
`__capstone_cap_init` that stores each capability-global element in place at
runtime (isel lowers to a tagged `cincoffset gp`+`delin`+`stc`);
`capstone/my_first_domain/start.S` calls it before `domain_main` (a weak no-op
default covers domains with none). Validated end-to-end: the three previously
faulting `static-cap-typed-load-repro` domains now pass unchanged (string-struct,
array=`nums[]` shape, function-pointer); 28/28 Capstone lit tests; `bs`
unaffected. Decision + implementation:
`capstone/agent-handoff/design/capability-globals-init-decision.md`; test
`static-cap-global-init.ll`. (Multi-module offset-table generalization is a
documented follow-on; not needed for single-module domains.)

This resolved `dtoa` **blocker #1** (untagged `nums[]`). `dtoa` remains the one
deferred benchmark only for **blocker #2** — arena 16-byte alignment in David
Gay's `char[]`/`double[]` bigint pool (a 16-byte `Bigint.next` capability landing
at an 8-mod-16 offset loses its tag). See `plans/beebs-deferred-benchmarks.md` §3
(shift fix) and §15 (`dtoa`). Finishing `dtoa` is the current next step.

### Benchmarks

80 BEEBS benchmarks now pass end-to-end. The most recent addition is
`janne_complex` (`run-beebs-janne_complex.sh`) — a trivial integer WCET
benchmark (nested data-dependent loops). It is fully self-contained: integer
only, includes only `support.h`, and its upstream `verify_benchmark` returns
`r == 1` (which `complex()` always yields), so it needs no soft-float, no libm,
no string lib, no adapted tail, and no host reference. The three wrappers just
delegate to `build-beebs-simple-{capstone,host}-common.sh` /
`run-beebs-simple-common.sh` (same minimal pattern as `bs`). No compiler change.

The prior addition was `fasta`
(`run-beebs-fasta.sh`) — the first of the libc-frontier benchmarks. Upstream
`fasta` discards all output and `verify_benchmark` returns -1, so the adapted
tail (`adapted/beebs_fasta_capstone_tail.c`) keeps the deterministic generator
core (`myrandom` LCG + `accumulate_probabilities`) and reimplements the two
consumers (`repeat_fasta`/`random_fasta`) to fold every generated character into
an FNV-1a checksum, compared exactly to a same-source host reference
(`0x24d70971e2d6dc0f`; `myrandom`'s f32 ops are correctly-rounded on both host
hardware float at `-ffp-contract=off` and target compiler-rt soft-float, so the
character stream is bit-identical). It introduced the shared freestanding
string/mem library `adapted/beebs_freestanding_string.c`
(`memcpy/memmove/memset/strlen/strcmp/strcpy` — the "pure computation" slice of
libc, locally implemented, the string counterpart to `beebs_softfloat_libm.c`;
`-ffunction-sections`/`--gc-sections` drops the unreferenced routines) and added
`floatdisf`/`floatundisf` to the shared soft-float builtin set. The host-gcc
recompute matches the reference bit-for-bit, confirming the generator is
compiler-independent. No compiler change.

The prior additions were
`matmult-float` and `whetstone` (`run-beebs-{matmult-float,whetstone}.sh`),
which complete the soft-float/libm-only FP class. Both reuse the soft-float
builtins (+ shared libm) with no compiler change, and both use the proven
"reference computed from the same source + same soft-float math, compared
exactly" pattern (IEEE float/double ops are bit-identical between host hardware
float at `-ffp-contract=off` and target compiler-rt soft-float).

- `matmult-float`: the same source as `matmult` built `-DMATMULT_FLOAT`
  (UPPERLIMIT 10, float[10][10]); soft-float builtins only (no libm). The adapted
  tail replaces the upstream local-`exp[][]` verifier (Bug #3/#9) with an FNV-1a
  checksum of the global `ResultArray` read as a flat byte stream (oracle
  `0xbdbace3d315e67a4`). Built `-ffunction-sections`/`--gc-sections` so the dead
  upstream `values_match` (which would pull in `frexpf`/`fabsf`) is dropped.
- `whetstone`: needed `atan` added to the shared `adapted/beebs_softfloat_libm.c`
  (fdlibm port, ~1.6e-16, validated by the self-test). Upstream `verify` is -1
  and the per-module results flow only through `POUT` (gated on `PRINTOUT`), so
  the domain is built `-DPRINTOUT`, the upstream printf `POUT` definition block
  is stripped, and the adapted tail's capturing `POUT` folds every module's four
  doubles into an FNV checksum compared (exact) to a same-libm host reference
  (`0x2f975c4609a1bfbb`).

The prior addition was
`stb_perlin` (`run-beebs-stb_perlin.sh`), a 3-D Perlin-noise benchmark. Its
oracle is self-contained: `benchmark()` computes a 10x10 noise plane and
compares every value against a `static const float expected[10][10]` global
(in `.rodata`, so no Bug #9), returning 0 iff all 100 match exactly. The adapted
tail just checks `res == 0`. Its only external dependency is `floor`, newly
added to the shared `adapted/beebs_softfloat_libm.c` (bit-exact, validated by
the libm self-test); everything else is the existing soft-float builtins. Built
`-ffp-contract=off`; host (gcc -O0 -ffp-contract=off) and target match the
embedded table bit-for-bit. No compiler change. Note: `matmult-int`'s upstream
source is byte-identical to `matmult/matmult.c`, which `run-beebs-matmult.sh`
already builds with `-DMATMULT_INT`, so it is effectively already covered.

The prior step added the four
`newlib-*` single-precision math benchmarks `newlib-sqrt`, `newlib-exp`,
`newlib-log`, `newlib-mod` (`run-beebs-newlib-{sqrt,exp,log,mod}.sh`). Each
`src/newlib-*/ef_*.c` is **self-contained** — it ships its own routine
(`__ieee754_sqrtf`/`expf`/`logf`/`fmodf`, integer bit-manipulation plus
non-contracted float arithmetic) with no libm/libc calls — so they reuse only
the soft-float builtins (`build-beebs-softfloat-common.sh`); no libm object, no
compiler change. Built with `-ffp-contract=off` so no FMA contraction can
diverge from the soft-float reference. `newlib-sqrt` keeps the upstream exact
`==` verifier (its `exp[]` is moved to `static const` to avoid Bug #9; the
correctly-rounded `__ieee754_sqrtf` is bit-identical to the embedded newlib
values); `newlib-exp/log/mod` have upstream `verify_benchmark == -1`, so each
gets an oracle tail that captures all five calls and exact-bit-compares them
against a host reference (`gcc -O0 -ffp-contract=off` over the same source).

The prior additions `qsort`,
`qurt`, and `select` (`run-beebs-{qsort,qurt,select}.sh`) — FP benchmarks needing
only the soft-float builtins (no libm; they ship their own helpers or use only
float compares), each with an adapted oracle tail (upstream verifiers return -1):
`qsort` widens `arr` to [21] and checks monotonicity plus a host-reference hash
over the sorted 1-indexed region; `qurt` captures and checks all three known
quadratic root cases (tolerance — it uses its own approximate sqrt);
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
compact self-contained `adapted/beebs_softfloat_libm.c` provides
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

`run-all-beebs.sh` now has low-token aggregate output: child wrapper output goes
to `$CAPSTONE_TMP_ROOT/run-all-beebs/*.attempt-N.log`, while the aggregate prints
compact pass/fail lines. It is serial by default, with opt-in isolated
parallelism via `RUN_ALL_BEEBS_JOBS=N`; each attempt gets its own build/share
workspace under the aggregate log directory. It retries only structured QEMU
infra flakes before benchmark execution twice by default and caps aggregate
boot-to-login waits at 90 seconds (`RUN_ALL_BEEBS_LOGIN_TIMEOUT`) so QEMU boot
flakes fail fast into that retry; real marker failures still stop immediately.

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

### Remaining uncovered benchmarks — the libc/format frontier only

The soft-float/libm-only FP class is now **complete** (`matmult-float` and
`whetstone` were the last two; `whetstone` is exact via the same-libm reference,
not a tolerance oracle). `matmult-int`/`matmult-float` source is byte-identical
to `matmult/matmult.c` (built `-DMATMULT_INT`/`-DMATMULT_FLOAT` respectively).
What remains are heavier, libc-dependent benchmarks, each its own effort:

- `fasta` - needs libc (`memcpy`/`strlen`/`malloc`-ish).
- `trio`, `trio-snprintf` - float / complex format lib (`trio-sscanf` is the
  validated proof wrapper; `trio-snprintf` also has `verify_benchmark = -1`).
- `dtoa` - heavy libc (`malloc`/`errno`/`float.h`/`fenv.h`/`locale`) + libm.

Plus the **Bug #9 backend root fix** (removes the `static const` source
workaround class across many benchmarks).

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
