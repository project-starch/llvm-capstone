# Current recommended next step

## Immediate milestone - Add the next BEEBS benchmark batch (Batch 3)

**Goal**: extend the validated BEEBS set with benchmarks that need minor stubs
or multi-file compilation.

**Why this next**: Batch 1 (bs, fir, lcdnum, ns, ud) and Batch 2 (nsichneu,
sglib-arraysort, sglib-arrayheapsort, sglib-arrayquicksort) are all committed
and validated. The common script now has `-fno-jump-tables` and `BEEBS_EXTRA_DEFINES`
support.

## Candidate pool — Batch 3 (stubs/multi-file)

- `rijndael`: `src/rijndael/aes.c` + `src/rijndael/aesxam.c`. Strip
  `stdio.h`, `stdlib.h`, `ctype.h`. `fpos_t` is self-defined. Multi-file compile.
- `picojpeg`: `src/picojpeg/libpicojpeg.c` + `src/picojpeg/picojpeg_test.c`.
  Strip `string.h`, provide memcpy stub. Multi-file compile.
- `sglib-dllist`: Strip `stdio.h`, `stdlib.h`, `string.h` (unused). `BEEBS_DEFINE_NULL=1`.
- `sglib-hashtable`: Same as sglib-dllist.
- `nettle-aes`: Strip `assert.h`, add `#define assert(x) ((void)(x))` stub.
- `nettle-sha256`: Strip `assert.h`, add assert stub + `abort()` stub.
- `huffbench`: Strip many headers, provide `memset` stub. Verify no FP libcalls.

## Newly classified FP-blocked (defer)

- `qsort` — `float arr[20]` comparisons → soft-float libcalls
- `select` — `float arr[20]` comparisons → soft-float libcalls
- `sqrt` — explicit float arithmetic
- `qurt` — explicit float arithmetic
- `fasta` — float probability arithmetic

## Previously deferred

- `slre`: Clang frontend PHINode type mismatch (Bug #11). Not fixable at source.
- `nbody`, `trio`, `frac`, `st`, `stb_perlin`, `whetstone`, `newlib-*`: FP blocked.
- `sglib-rbtree`, `aha-mont64`, `dijkstra`, `edn`, `ctl-string`, `qrduino`,
  `nettle-arcfour`, `ludcmp`, `nettle-des`, `statemate`: compile-time backend
  crashes or non-trivial source adaptation required.
- `wikisort`: Range by-value ABI bug (Bug #10) throughout; invasive rewrite.
- `compress`, `dtoa`, `cubic`: backend crashes.

## Test expectations

For each new benchmark commit:

- the new `run-beebs-<name>.sh`
- `"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone`
- `bash capstone/tests/runtime-qemu/run-coremark.sh`
- focused existing BEEBS regressions: `fac`, `strstr`, `ndes`, `expint`

## Thinking-level rule

Stay at medium thinking while the work remains mechanical or locally debuggable.
If a benchmark exposes a hard backend/compiler bug, unclear architecture
semantics, or repeated failed runtime debugging where higher thinking looks
necessary, suspend work and tell the user before continuing.

## What not to regress

Do not delete `capstone/caplifive-buildroot/build/local.mk` - its absence silently
switches the image to stock OpenSBI and breaks all runtime proofs.
