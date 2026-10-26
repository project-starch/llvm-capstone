# Virtual-profile microbenchmarks: a CoreMark-derived workload, RV8 and BEEBS

Ordinary C programs built with an SDK's `capstone-cc -O2` and run through `capstone-vexec`, for the
silicon comparison of two heap configurations:

| arm | SDK switches (`runtime/application/CMakeLists.txt`) | meaning |
|---|---|---|
| T1 | `-DCAPSTONE_HEAP_LIFETIMES=OFF -DCAPSTONE_HEAP_SCRUB_FREE=OFF` | Sublet spatial safety only |
| T2 | defaults (both `ON`) | spatial + temporal |

Both SDKs are built with `CAPSTONE_VIRTUAL_MOVC_KEEP=0` (`sdk.json` codegen carries
`-movc-keeps-integer-source`, as board images must until the bitstream has Q-04 (b); C-32).

**No score or timing from this directory is a published result.** QEMU runs here are correctness
checks; under `--icount` the guest clock counts instructions and is used only to size iterations.
The CoreMark program is a *CoreMark-derived workload*: it is not a conforming CoreMark run (it
runs for well under 10 s, one core line is changed and its CRCs are not upstream's; see below), and
EEMBC's rules forbid publishing its numbers as CoreMark scores. Only arm-to-arm ratios are
reported.

Upstream sources are never copied into the repository: the existing fetch scripts
(`../{coremark,beebs,rv8}/fetch-*.sh`) put them in `/tmp/capstone/{coremark,beebs,rv8}-src`, and
`build.py` refuses a tree that is not clean at the pinned commit. Every source adaptation is an
exact substitution into a copy in the build directory, with its match count asserted.

## Files

| file | role |
|---|---|
| `build-sdks.sh OUT CACHE [SRC]` | the two SDKs (`OUT/sdk-T1`, `OUT/sdk-T2`) from one prepared musl tree, built from the source tree SRC (default: this one; recorded in `sdk-source.txt`); checks the `sdk.json` switches, runs `check-sdks.py` and `heap-layout.py` |
| `check-sdks.py T1 T2` | lists every file that differs between the two SDK trees; passes only if the content differs in `heap.c`'s object and the `sdk.json` heap switches alone |
| `build-all.sh SDK OUT` | every image of one arm (`BENCH_MODE=count` / `native` for the counting and native builds); a wrapper of `build.py` |
| `build.py` | the program table, source adaptations and compile/link lines; writes `OUT/images.tsv`, `build.log`, `warnings.log` |
| `sizes.tsv` | per-program size: BEEBS `REPEAT_FACTOR` (via `-DBOARD_REPEAT_FACTOR`), RV8 `N`, CoreMark `ITERATIONS` |
| `bench-harness.{c,h}` | the shared measurement harness (below) |
| `beebs/beebs-main.c` | BEEBS driver: one iteration is the body of BEEBS's own `support/main.c` |
| `beebs/state/*.c` | check values for the programs whose `verify_benchmark` returns -1 (upstream verifies nothing) |
| `beebs/beebs-libc-alloc.h` | the `<name>-libc` variants: `*_beebs` allocator calls go to libc |
| `rv8/*.c` | one driver per RV8 program, replaying upstream `main()` as the harness unit |
| `coremark/core_portme.{c,h}` | the CoreMark port layer |
| `heap-layout.py` | the allocator's fixed layout region for both arms (`heap-layout.json`) |
| `kvariants.txt` | programs built at K = 0..3 |
| `ctl/loop.c` | the `ctl/empty` and `ctl/loop` controls |
| `check-arms.py T1 T2` | the layout gate (objects, section headers, linker maps) |
| `qemu/run-qemu.py`, `qemu/walltime.c` | the QEMU functional pass, many images per boot |
| `qemu/console-audit.py` | every console line a process printed besides the harness's, per image |
| `nonreentrant.txt` | programs run one timed iteration per process (`0 1`) |
| `resize.py` | re-derives `sizes.tsv`, `kvariants.txt` and `nonreentrant.txt` from a sizing boot; `--check-only` lists sizes out of range |
| `make-manifest.py` | an arm's manifest TSV from its build and its QEMU runs |
| `functional-table.py` | the per-program QEMU functional table (T1 ok, T2 ok, native check equal, failures) |
| `pipeline.sh` | everything above, from the SDKs to the manifests, in one command |

## The harness

One process runs `W` warm-up and then `I` timed iterations (`argv[1]`, `argv[2]`; defaults 1 and 5)
of the program's whole benchmark unit. Each iteration is bracketed by a mark and one
`clock_gettime(CLOCK_MONOTONIC)` at each end. A mark only `pread`s the perf module's three
cumulative counters (`bench_cycles`, `bench_instret`, `bench_entries` under
`/sys/module/capstone_vm/parameters/`, opened once at start-up) into static buffers: no stdio, no
allocation, no parsing. The buffers are parsed after the end mark and the line is printed after
that:

    BENCH <suite>/<name> it=<k> cycles=<d> instret=<d> entries=<d> ns=<d> check=<v>

(`it=w0..` for warm-ups). Absent counter files read 0 and `BENCH-INFO <id> counters=absent` is
printed once (`present` when all three exist). The exit status is 0 only if every iteration
verified. A program whose static state is not re-initialised between iterations is
`reentrant=no` in the manifest and runs `0 1` (one timed iteration per process).

The unit: BEEBS `initialise_benchmark(); REPEAT_FACTOR x { initialise_benchmark(); benchmark(); };
verify_benchmark()`; RV8 the program's `main()` work; CoreMark one call of upstream `main()`, whose
own `start_time()`/`stop_time()` are the bracket (only the iterations are timed, as upstream).

Controls, built from one source (`ctl/loop.c`) so that they differ ONLY inside one asm statement:
`ctl/empty` (no instructions: the cost of one bracket) and `ctl/loop` (`li` of N = 10,000,000, then
`addi`/`bnez` N times), so instret(loop) - instret(empty) = 2N + 2 = 20,000,002 exactly. The
prediction is printed as `BENCH-INFO ctl/loop predicted_instret_minus_empty=`.

The counting build (`-DBENCH_COUNT_ALLOCS`, linked `--wrap=malloc,free,realloc,calloc`) is for the
QEMU functional pass only. It prints `ALLOC <id> it=<k> malloc= free= realloc= calloc=` per
iteration and for the whole run. Benchmark objects are identical to the measured build's; only
the harness object and the link differ.

## Layout control

capstone-cc links the runtime archive (heap.c included) before the program, so by default every
benchmark symbol moved between the arms by heap.c's size difference (+0x94, or +0xb8 where
`realloc` is linked). The measured images are therefore linked by `build.py` with capstone-cc's own
`ld.lld` command line and link-input checks (imported from the SDK's `capstone-cc`), but with the
SDK's `link.ld` rewritten (`OUT/layout/layout-K<k>.ld`, asserted substitutions):

* heap.c's object is LAST in `.text` and in `.bss`, inside a region of fixed size, the same in both
  arms. `heap-layout.py` (run by `build-sdks.sh`) reads heap.c's object out of each SDK, checks it
  has no allocated section other than `.text*`/`.bss*`, and records the region sizes and both
  objects' hashes in `heap-layout.json`; `build.py` refuses an SDK whose heap object is not
  recorded there. Text region = max of the two arms' heap text rounded to 64, plus 3 x 64.
* Programs that allocate inside the timed region (`kvariants.txt`, from the counting run) are built
  at K = 0..3: heap.c's text starts K x 64 bytes into its region. Images are `<name>.K<k>.dom`;
  other programs are built at K0 only.
* `check-arms.py` is the gate: benchmark objects byte-identical across arms; per image pair,
  identical allocated section headers (`llvm-readelf -S`) and every input section and symbol of the
  linker maps (`-Map`) at the same address and size, except heap.c's own and the region's
  bookkeeping. `--k-check` requires every K>0 image to equal K0 outside heap.c;
  `--negative-control` (a T2 build with `--layout default`) must FAIL every pair, which shows the
  gate can fire.

## Source adaptations (all arms, the -libc variants and the native build alike)

Compile flags for every program: `-O2 -ffp-contract=off`. BEEBS's own checks compare floats
exactly against values computed without fused multiply-add; contracted, `ludcmp`, `matmult-float`
and `st` fail their own verification on both compilers.

* CoreMark (known deviations, a *CoreMark-derived workload*): one line of `core_list_join.c` is
  changed, in a build-directory copy: `core_list_init`'s per-node size `16+sizeof(struct
  list_data_s)` becomes `CM_LIST_NODE_BYTES+sizeof(struct list_data_s)`, with
  `CM_LIST_NODE_BYTES=32` in EVERY build, the native reference included. Unchanged, the 32-byte
  purecap `list_head` makes the list overrun its third of the 2000-byte block by ~80% into the
  matrix block, and the matrix initialisation overwrites live list pointers: QEMU T1 faults with
  cause=24 on the first list walk (`build.py --coremark-unmodified` reproduces it). With 32-byte
  nodes the list has 16 nodes instead of 31, and ALL THREE first-iteration CRCs differ from
  upstream's table, not only the list CRC: the matrix and state CRCs are first computed from list
  data (`calc_func`). The port checks them against the values for this geometry, list 0x70a5,
  matrix 0xdb4b, state 0x905f, computed with the same geometry by x86-64 gcc -O2 and by the
  riscv64 buildroot gcc -O2 (QEMU), both LP64; the native reference reproduces them. `core_main.c`
  prints its own "ERROR! ... crc" lines against upstream's table, as expected. `align_mem` is
  pointer arithmetic, not an integer round trip; `ee_ptr_int` is an opaque pointer-sized holder.
  `-Dmain=coremark_main` on `core_main.c` lets one process run several iteration sets; nothing else
  in the core files changes. It runs far under CoreMark's 10 s minimum, which its report flags.
* BEEBS `qsort`, `select`: `float arr[20]` -> `arr[21]` (a 1-indexed sort of `arr[1..20]`; exact
  global bounds trap the access one past the end).
* BEEBS `rijndael`: `aes.h`'s `word` ("must be a 32-bit storage unit") becomes `unsigned int`
  (as `unsigned long` it reads 8 bytes at offset 12 of a 16-byte block); `fillrand`'s
  `static char r[4]`, written as an `unsigned long`, becomes `r[8]`.
* BEEBS `ctl-vector`: `VectorDelete`'s `memmove` copies `(size-begin)` elements, not `size`, which
  read past the live array (trapped in `ctl-vector-libc`).
* BEEBS `dtoa`: the private Bigint pool is 16-aligned and hands out 16-byte multiples (a Bigint
  holds a pointer; an 8-mod-16 block traps on its first capability store).
* The 14 bump-allocator programs: the arena is 16-aligned and each allocation rounded to 16 bytes,
  for the same reason. Their `-libc` variants rename the bump definitions and route
  `malloc_beebs`/`free_beebs`/`realloc_beebs`/`calloc_beebs` to libc (`beebs-libc-alloc.h`).
* RV8 and BEEBS console output: CoreMark's report is formatted and read for the verdict but not
  written (`ee_printf` in the portme); Dhrystone's report line is silenced by
  `-Dprintf=rv8_silent_printf` on its driver. Both printed after the end mark, once per iteration;
  on the board stdout is a UART and every write a delegated call. Every other program prints only
  the harness lines (`qemu/console-audit.py`).
* RV8: upstream `main()` is stripped from a build-directory copy and replayed by the driver with a
  build-time size; `dhrystone`'s `LOOPS` is rewritten and its two `gettimeofday` calls become the
  bracket. Deviations, each in the driver's header comment: buffers upstream leaves partly
  unwritten after `malloc` are zeroed (a process that reuses freed memory does not get fresh
  zero pages, and T1 and T2 would differ); `primes` shifts `1ULL` (upstream shifts an `int` by up
  to 63); `miniz` draws its pattern from a copy of musl's `rand()` so the glibc native build sees
  the same bytes; `dhrystone` does not check `BoolGlob` (rv8-bench never initialises
  `String1Loc`).

## Coverage

* BEEBS: the 80 programs upstream builds. `src/Makefile.am`'s SUBDIRS lists 85 directories; `ctl`,
  `template` and `trio` are support directories, and `matmult` and `sglib-arraysort` build nothing
  (their `Makefile.am` has only `EXTRA_DIST`). `src/matmult`, `src/matmult-int` and
  `src/matmult-float` hold one `matmult.c` (cmp-identical), built as matmult-int (`-DMATMULT_INT`)
  and matmult-float (`-DMATMULT_FLOAT`); `src/sglib-arraysort/arraysort.c` is
  sglib-arrayquicksort's. The physical-profile port's `matmult` and `sglib-arraysort` were those two
  programs under other names, so they are not built twice.
* The 14 programs with BEEBS's bump allocator also get a `-libc` variant; `qrduino-libc` is built
  but excluded from the measured set: upstream `verify_benchmark` reads `strinbuf` after
  `benchmark()` has freed it (`freeecc`), a use-after-free once `free_beebs` is a real free (QEMU T1
  cause=25; the native build fails its verification).
* RV8: the 7 C programs. `bigint.cc` is C++ and needs the C++ standard library (`<vector>`,
  `<string>`, `<iostream>`, `<algorithm>`): capstone-cc compiles C++, but the virtual SDK has no C++
  runtime and ships musl's C headers only (`#include <vector>`: file not found). Not ported, and not
  rewritten in C.
* CoreMark: the five core files and `coremark.h`, as above.

## Rebuild

One command runs everything, from the SDKs to the manifests (`pipeline.sh`, stages
`sdks build size rebuild gate functional report`):

    export CAPSTONE_LLVM_BIN=<cc>/bin CAPSTONE_LLVM_BUILD_DIR=<cc>
    export MUSL_CACHE=<dir with musl-1.2.5.tar.gz> CAPSTONE_QEMU_LOCK=$HOME/.capstone-locks/qemu.lock
    export KIT=<QEMU kit: adapter/, images/, qemu-system-riscv64>
    capstone/benchmarks/virtual/pipeline.sh <root> [<source tree for the SDKs>]

`size` boots the T1 images and the counting images under `-icount` and rewrites `sizes.tsv` (a
size is scaled to about 1.2e7 guest instructions per timed iteration, after subtracting the
`ctl/empty` bracket), `kvariants.txt` (programs with an allocator call inside a timed iteration)
and `nonreentrant.txt`; `rebuild` then builds with them. `functional` runs both arms and the native
references (`ARGS`, default `1 10`; non-reentrant programs `0 1`) and the counting images;
`report` writes `manifest-T1.tsv`, `manifest-T2.tsv`, `functional.tsv`, `sizing-T*.tsv` (internal)
and `size-check.txt`. The QEMU passes are correctness checks only.

The pieces, by hand:

    capstone/benchmarks/virtual/build-sdks.sh <root> <musl cache> [<source tree>]
    capstone/benchmarks/virtual/build-all.sh <root>/sdk-T1/sdk <out>/T1
    capstone/benchmarks/virtual/build-all.sh <root>/sdk-T2/sdk <out>/T2
    BENCH_MODE=count  capstone/benchmarks/virtual/build-all.sh <root>/sdk-T1/sdk <out>/T1-count
    BENCH_MODE=native capstone/benchmarks/virtual/build-all.sh - <out>/native
    python3 capstone/benchmarks/virtual/check-arms.py <out>/T1 <out>/T2 --k-check \
        --negative-control <out>/T2-default    # T2-default: build.py --layout default
