# Current Capstone state

Minimal snapshot. Read first in every session.

## Verified baseline

All of the following pass on the `capstone-bootstrap` branch:

- LLVM Capstone backend builds the sample domain; `ld.lld` links native `EM_CAPSTONE`
- `capstone/caplifive-buildroot/build/local.mk` present — keeps the image on the Capstone-enabled OpenSBI path
- All HostCall probes pass: shared-region, stdout, filewrite, fileread, full file-handle
  lifecycle (open/write/read/sync/stat/truncate/close), path ops, combined file-object
- `run-nullblk-baseline.sh`, `run-nullblk-split-io.sh`, and
  `run-nullblk-split-rmmod.sh`
- `run-hostcall-all.sh`, `run-nullblk-all.sh`, and `run-all-beebs.sh` provide
  aggregate gates for reproducible full reruns; keep individual wrappers as the
  diagnostic entry points. The HostCall, `null_blk`, and full BEEBS aggregates
  have passed end to end; BEEBS has also passed with `RUN_ALL_BEEBS_JOBS=4`.
  `run-all-beebs.sh` is serial by default
  (`RUN_ALL_BEEBS_JOBS=1`) and has opt-in isolated parallelism via
  `RUN_ALL_BEEBS_JOBS=N`. It keeps child output in per-benchmark logs by default
  and prints compact pass/fail lines; set `RUN_ALL_BEEBS_VERBOSE=1` for streamed
  child output. It retries structured QEMU infra flakes before benchmark
  execution twice by default (`RUN_ALL_BEEBS_BOOT_RETRIES=0` disables this) and
  caps aggregate boot-to-login waits at 90 seconds by default
  (`RUN_ALL_BEEBS_LOGIN_TIMEOUT`), but does not retry benchmark marker failures.
- QEMU runtime smoke tests use snapshot mode, so repeated runs do not mutate `rootfs.ext2`
- Buildroot getty is pinned to `ttyS0`, avoiding intermittent boot-to-login hangs through `/dev/console`
- QEMU runtime smoke tests force `-smp 1`, avoiding intermittent boot stalls under the current OpenSBI/QEMU setup
- `run-coremark.sh` - all three algorithms, "Correct operation validated."; CoreMark now uses
  compiled C `domain_main`, not `coremark_domain_entry.S`
- `capstone/benchmarks/beebs/run-beebs-fac.sh` - first BEEBS benchmark runs end to end
  and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-insertsort.sh` - second BEEBS benchmark runs end to end
  and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-fibcall.sh` - third BEEBS benchmark runs end to end
  and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-cnt.sh` - fourth BEEBS benchmark runs end to end
  and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-bubblesort.sh` - fifth BEEBS benchmark runs end to
  end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-prime.sh` - sixth BEEBS benchmark runs end to
  end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-recursion.sh` - seventh BEEBS benchmark runs end
  to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-janne-complex.sh` - eighth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-tarai.sh` - ninth BEEBS benchmark runs end
  to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-cover.sh` - tenth BEEBS benchmark runs end
  to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-duff.sh` - eleventh BEEBS benchmark runs
  end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-levenshtein.sh` - twelfth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-jfdctint.sh` - thirteenth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-fdct.sh` - fourteenth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-strstr.sh` - fifteenth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ndes.sh` - sixteenth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-arraybinsearch.sh` - seventeenth
  BEEBS benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-queue.sh` - eighteenth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-listinsertsort.sh` - nineteenth
  BEEBS benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-listsort.sh` - twentieth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-expint.sh` - twenty-first BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-aha-compress.sh` - twenty-second BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-md5.sh` - twenty-third BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-cast128.sh` - twenty-fourth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-matmult.sh` - twenty-fifth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-crc32.sh` - twenty-sixth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-mergesort.sh` - twenty-seventh BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-stringsearch1.sh` - twenty-eighth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-bs.sh` - twenty-ninth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-fir.sh` - thirtieth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-lcdnum.sh` - thirty-first BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ns.sh` - thirty-second BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ud.sh` - thirty-third BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nsichneu.sh` - thirty-fourth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-arraysort.sh` - thirty-fifth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-arrayheapsort.sh` - thirty-sixth
  BEEBS benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-arrayquicksort.sh` - thirty-seventh
  BEEBS benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-dllist.sh` - thirty-eighth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-hashtable.sh` - thirty-ninth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-aes.sh` - fortieth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-picojpeg.sh` - forty-first BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-sha256.sh` - forty-second BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-huffbench.sh` - forty-third BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-rijndael.sh` - forty-fourth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-crc.sh` - forty-fifth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-statemate.sh` - forty-sixth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-arcfour.sh` - forty-seventh BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-des.sh` - forty-eighth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-aha-mont64.sh` - forty-ninth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-dijkstra.sh` - fiftieth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ctl-stack.sh` - fifty-first BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ctl-vector.sh` - fifty-second BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-edn.sh` - fifty-third BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ctl-string.sh` - fifty-fourth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-qrduino.sh` - fifty-fifth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-rbtree.sh` - fifty-sixth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-miniz.sh` - fifty-seventh BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-slre.sh` - fifty-eighth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-wikisort.sh` - fifty-ninth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-trio-sscanf.sh` - sixtieth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-compress.sh` - sixty-first BEEBS
  benchmark runs end to end and validates its adapted LZW-state checksum marker
- `capstone/benchmarks/beebs/run-beebs-cubic.sh` - sixty-second BEEBS
  benchmark runs end to end with the soft-float/libm runtime and root oracle
- `capstone/benchmarks/beebs/run-beebs-sqrt.sh` - sixty-third BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ludcmp.sh` - sixty-fourth BEEBS
  benchmark runs end to end with the local const-array source workaround
- `capstone/benchmarks/beebs/run-beebs-minver.sh` - sixty-fifth BEEBS
  benchmark runs end to end and validates its adapted matrix checksum marker
- `capstone/benchmarks/beebs/run-beebs-frac.sh` - sixty-sixth BEEBS
  benchmark runs end to end with shared soft-float/libm support
- `capstone/benchmarks/beebs/run-beebs-st.sh` - sixty-seventh BEEBS
  benchmark runs end to end with correctly-rounded software `sqrt`
- `capstone/benchmarks/beebs/run-beebs-nbody.sh` - sixty-eighth BEEBS
  benchmark runs end to end with correctly-rounded software `sqrt`
- `capstone/benchmarks/beebs/run-beebs-qsort.sh` - sixty-ninth BEEBS
  benchmark runs end to end with a widened 1-indexed array and sorted-region hash
- `capstone/benchmarks/beebs/run-beebs-qurt.sh` - seventieth BEEBS benchmark
  runs end to end and validates all three quadratic root cases
- `capstone/benchmarks/beebs/run-beebs-select.sh` - seventy-first BEEBS
  benchmark runs end to end with a widened 1-indexed array and return-value oracle
- `capstone/benchmarks/beebs/run-beebs-newlib-sqrt.sh` - seventy-second BEEBS
  benchmark; self-contained `__ieee754_sqrtf`, upstream exact verifier with
  `exp[]` moved to `static const` (Bug #9), soft-float builtins only
- `capstone/benchmarks/beebs/run-beebs-newlib-exp.sh` - seventy-third BEEBS
  benchmark; self-contained `__ieee754_expf`, oracle tail vs host reference
- `capstone/benchmarks/beebs/run-beebs-newlib-log.sh` - seventy-fourth BEEBS
  benchmark; self-contained `__ieee754_logf`, oracle tail vs host reference
- `capstone/benchmarks/beebs/run-beebs-newlib-mod.sh` - seventy-fifth BEEBS
  benchmark; self-contained `__ieee754_fmodf`, oracle tail vs host reference
- `capstone/benchmarks/beebs/run-beebs-stb_perlin.sh` - seventy-sixth BEEBS
  benchmark; 3-D Perlin noise, self-contained oracle (`benchmark()` compares a
  10x10 plane against a `static const` table and returns 0 on full match);
  only external dep is `floor`, added to the shared soft-float libm
- `capstone/benchmarks/beebs/run-beebs-matmult-float.sh` - seventy-seventh BEEBS
  benchmark; `matmult` source built `-DMATMULT_FLOAT` (float[10][10]), soft-float
  builtins only, FNV-1a checksum of the global `ResultArray` vs a host reference
  (`--gc-sections` drops the dead `values_match`/`frexpf`/`fabsf`)
- `capstone/benchmarks/beebs/run-beebs-whetstone.sh` - seventy-eighth BEEBS
  benchmark; classic Whetstone over the shared libm (added `atan`); built
  `-DPRINTOUT` with a capturing `POUT` that FNV-folds every module's outputs,
  compared exactly to a same-libm host reference

Most BEEBS correctness-marker wrappers now share `beebs_simple_domain.c` and
`beebs_simple_host.c`. Keep separate per-benchmark domain/host files only when
the marker ABI or host behavior is genuinely different; currently the older
`fac`, `fibcall`, and `insertsort` wrappers keep custom markers.

Most Capstone-specific benchmark source adaptations live in explicit `.c` files
under `capstone/benchmarks/beebs/adapted/`; shell scripts generally orchestrate
fetch/build/link/run rather than embedding C source. Full-replacement adapted
files (bubblesort, prime, cnt, duff, janne_complex, tarai, levenshtein,
recursion) are compiled directly. Prefix/tail files (crc32) and tail-append
files (strstr, insertsort, jfdctint, fdct, aha-compress, nettle-md5,
nettle-cast128, nettle-arcfour, nettle-des) are concatenated with the stripped
upstream source at build time. `huffbench` uses checked-in adapted C snippets
for its freestanding prefix and RNG replacement. `aha-mont64` uses a checked-in
rewrite helper for constant hoisting. `ndes` uses a checked-in rewrite helper
for pointer-based aggregate passing and explicit table delinearization.
`ctl-string`, `qrduino`, `miniz`, `slre`, and `trio-sscanf` are generated as
scratch sources under `$CAPSTONE_TMP_ROOT/beebs-build` because their adaptations
are local include/stub/allocation/verifier rewrites rather than reusable
replacement translation units.  `slre` additionally uses a checked-in tail file
(`adapted/beebs_slre_capstone_tail.c`) to avoid the `char *regexes[]` global
pointer array that would require caprelocs.  `wikisort` uses a checked-in tail
file to keep the upstream prefix while replacing the Range/sort/test tail.
`trio-sscanf` strips hosted includes, builds with `TRIO_SSCANF`,
`TRIO_EMBED_STRING`, float/file/dynamic-string features disabled, a minimal set
of embedded `triostr` helpers, and checked-in freestanding libc stubs.
`compress`, `cubic`, `minver`, `qsort`, `qurt`, and `select` use adapted
oracle tails because the upstream verifiers return `-1`. FP benchmarks use
compiler-rt soft-float builtins and, where needed, the shared
`adapted/beebs_softfloat_libm.c` domain libm.

`build-beebs-simple-capstone-common.sh` now supports `BEEBS_EXTRA_DEFINES`
(array of `-D` defines, e.g. `BEEBS_EXTRA_DEFINES=(QUICK_SORT)`),
`BEEBS_STRIP_FROM_REGEX` plus `BEEBS_ADAPTED_TAIL_SRC` for single-source
tail-replacement adaptations, and includes `-fno-jump-tables` unconditionally
(jump tables use raw integer addresses which fault on Capstone since loads
require capabilities).

## Resolved blocker

The 2026-06-09/10 split `null_blk` unload blocker is resolved. The hang was
diagnosed as lost timer progress after split-domain activity: QEMU traces showed
that the final timer H-interrupt was taken while `mie.MTIP` was disabled, after
which OpenSBI did not reprogram the timer and RCU/percpu-ref progress stopped.

The fix is in `capstone/capstone-qemu`:

- Capstone H-interrupt selection in `riscv_cpu_local_irq_pending()` now considers
  only interrupts enabled by `env->mie`.
- `rmw_mie64()` calls `riscv_cpu_check_interrupts()` after `mie` changes so a
  pending H-interrupt becomes deliverable when software reenables it.

The split null_blk package also keeps the safer fixes found during investigation:
metadata is borrowed per domain call instead of permanently shared, and
`null_validate_conf()` copies back only validated scalar configuration fields.

All temporary Linux/OpenSBI/QEMU trace and printk diagnostics were removed before
the verified run.

## Important distinction

The validated path is the **split host/domain runtime path**, not a full hosted
`capstone64-unknown-linux-gnu` Linux userspace. The helper is ordinary guest Linux;
the domain is a Capstone-loaded domain.

## Known backend bugs (stable workarounds in place)

The prologue frame-lowering bug is fixed and validated. Three remaining LLVM backend
workarounds from CoreMark bring-up stay in `capstone/benchmarks/coremark/build-coremark-capstone.sh`
and should only be removed after focused root fixes. Details: `plans/backend-compiler-fixes.md`.

The `va_list` capability-tag-loss backend bug is fixed and validated: `va_start`/
`va_arg`/`va_copy` now lower with capability ops (`stc`/`ldc`, 16-byte `cincoffset`
stride). The CoreMark `ee_printf_asm.S` trampoline is removed — `ee_printf` uses a
standard C `va_list` and CoreMark still validates. This unblocks the `va_list`
prerequisite for `trio`.

The `sub i128` pointer-decrement backend blocker is fixed and validated:
`ptr - integer` and `ptr + (-offset)` now lower through `cincoffset` with a
negated XLEN offset.

The `sub i128` pointer-difference backend blocker is also fixed and validated:
`ptr - ptr` now lowers by extracting both capability cursors with `lcc ..., 2`,
subtracting the XLEN cursor values, and sign-extending the integer result back
through the `i128` carrier when needed. `ctl-string` is the proof benchmark.

The i128 non-vector-shift assertion (Bug #3) is fixed (`lowerScalarI128Shift`
general constant-shift fallback). **Capability globals are now auto-tagged**: the
`CapstoneCapGlobalInit` ModulePass synthesizes a per-module `__capstone_cap_init`
(called from `my_first_domain/start.S` before `domain_main`) that materializes
initialized capability globals in place at runtime — a tag cannot live in the
static image. Validated via `static-cap-typed-load-repro` + lit
`static-cap-global-init.ll`. Design:
`design/capability-globals-init-decision.md`.

## Where to go next

- Next milestone: `state/current-next-step.md`
- Test entry points: `ref/testing-matrix.md`
- Deep design docs: `design/`
