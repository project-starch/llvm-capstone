#!/usr/bin/env bash
# Cross-build the fixed-work mimalloc-bench programs for CheriBSD purecap.
#   build-mb.sh BENCH_DIR OUT_DIR
# BENCH_DIR is mimalloc-bench's bench/ directory (upstream daanx/mimalloc-bench).
# Prints "built NAME" or "FAILED NAME" per program; logs go to OUT_DIR/NAME.log.
set -uo pipefail
B=$1; O=$2
SDK=${SDK:-$HOME/cheri/output/sdk}
CC="$SDK/bin/clang --config $SDK/bin/cheribsd-riscv64-purecap.cfg"
CXX="$SDK/bin/clang++ --config $SDK/bin/cheribsd-riscv64-purecap.cfg"
W="-O3 -w -Wno-implicit-function-declaration -Wno-implicit-int -Wno-int-conversion -Wno-incompatible-pointer-types"
mkdir -p "$O"
build() { name=$1; shift
  if "$@" > "$O/$name.log" 2>&1; then echo "built $name"; else echo "FAILED $name (see $O/$name.log)"; fi; }

cfrac=(cfrac.c pops.c pconst.c pio.c pabs.c pneg.c pcmp.c podd.c phalf.c padd.c psub.c pmul.c
       pdivmod.c psqrt.c ppowmod.c atop.c ptoa.c itop.c utop.c ptou.c errorp.c pfloat.c pidiv.c
       pimod.c picmp.c primes.c pcfrac.c pgcd.c)
build cfrac $CC $W -DNOMEMOPT=1 -o "$O/cfrac" "${cfrac[@]/#/$B/cfrac/}" -lm
espresso=(cofactor.c cols.c compl.c contain.c cubestr.c cvrin.c cvrm.c cvrmisc.c cvrout.c
          dominate.c equiv.c espresso.c essen.c exact.c expand.c gasp.c getopt.c gimpel.c
          globals.c hack.c indep.c irred.c main.c map.c matrix.c mincov.c opo.c pair.c part.c
          primes.c reduce.c rows.c set.c setc.c sharp.c sminterf.c solution.c sparse.c unate.c
          utility.c verify.c)
build espresso $CC $W -o "$O/espresso" "${espresso[@]/#/$B/espresso/}" -lm
# barnes calls gets(), which FreeBSD's libc no longer provides.
printf '#include <stdio.h>\nstatic char *mq_gets(char *b, int n) { char *p; if (!fgets(b, n, stdin)) return 0; for (p = b; *p; p++) if (*p == 0x0a) { *p = 0; break; } return b; }\n#define gets(b) mq_gets(b, sizeof(b))\n' > "$O/mq-gets.h"
barnes=(code.c code_io.c load.c grav.c getparam.c util.c)
build barnes $CC $W -include "$O/mq-gets.h" -o "$O/barnes" "${barnes[@]/#/$B/barnes/}" -lm
build glibc-simple $CC $W -o "$O/glibc-simple" "$B/glibc-bench/bench-malloc-simple.c" -lpthread
# sh6bench knows only Linux among the BSD-like systems: give FreeBSD its POSIX-thread path.
mkdir -p "$O/sh6-src" && cp "$B/shbench/sh6bench-new.c" "$O/sh6-src/"
sed -i 's/|| defined(sgi) || defined(__DGUX__) || defined(__linux__)/& || defined(__FreeBSD__)/; s/|| defined(__DGUX__) || defined(__linux__)$/& || defined(__FreeBSD__)/' "$O/sh6-src/sh6bench-new.c"
build sh6bench $CC $W -DBENCH=1 -DSYS_MULTI_THREAD=1 -o "$O/sh6bench" "$O/sh6-src/sh6bench-new.c" -lpthread
build mstress $CC $W -o "$O/mstress" "$B/mstress/mstress.c" -lpthread
# sh8bench: the same OS switch as sh6bench (its second line also names __MVS__).
mkdir -p "$O/sh8-src" && cp "$B/shbench/sh8bench-new.c" "$O/sh8-src/"
sed -i 's/|| defined(__linux__)/& || defined(__FreeBSD__)/' "$O/sh8-src/sh8bench-new.c"
# Its doBench has a bare "return;" in a non-void function, an error by default in this clang.
build sh8bench $CC $W -Wno-return-type -DBENCH=1 -DSYS_MULTI_THREAD=1 -o "$O/sh8bench" "$O/sh8-src/sh8bench-new.c" -lpthread
build mleak $CC $W -o "$O/mleak" "$B/mleak/mleak.c" -lpthread
build malloc-large $CXX $W -std=c++17 -o "$O/malloc-large" "$B/malloc-large/malloc-large.cpp" -lpthread
# alloc-test reads __rdtsc() except on Apple/aarch64; take its clock_gettime path on RISC-V.
mkdir -p "$O/alloc-test-src" && cp "$B"/alloc-test/* "$O/alloc-test-src/"
sed -i 's/defined(__APPLE__) || defined(__aarch64__)/defined(__APPLE__) || defined(__aarch64__) || defined(__riscv)/' "$O/alloc-test-src/test_common.h"
# Its RSS probe reads /proc/self/statm (absent on FreeBSD; fread(NULL) traps).
# Take the getrusage branch instead; the program only prints that number.
sed -i 's/^#elif defined(__APPLE__)$/#elif defined(__APPLE__) || defined(__FreeBSD__)\n#include <sys\/resource.h>/' "$O/alloc-test-src/test_common.cpp"
build alloc-test $CXX $W -std=c++17 -DBENCH=4 -o "$O/alloc-test" "$O/alloc-test-src/test_common.cpp" "$O/alloc-test-src/allocator_tester.cpp" -lpthread
cp "$B/espresso/largest.espresso" "$B/barnes/input" "$O/" 2>/dev/null
