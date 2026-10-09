#!/usr/bin/env bash
# The fixed-work C programs of mimalloc-bench as ordinary programs for the delegated
# (virtual) runtime: compiled and linked by the application SDK's capstone-cc against musl.
# They are the Capstone side of the malloc-interface comparison with default CheriBSD
# (llvm-capstone experiments/cheribsd-malloc-quarantine, capstone/experiments/malloc-quarantine).
# Same sources and run arguments as mimalloc-bench 69c41ed (bench/CMakeLists.txt, bench.sh).
# Every program is compiled hosted with builtins (-fhosted -fbuiltin after the SDK's
# -ffreestanding -fno-builtin), which is how CheriBSD's compiler builds them: main returns 0
# when it falls off its end (espresso), and an undeclared library function gets its library
# type (barnes calls malloc and strchr undeclared). No source is changed except:
#   cfrac         -DNOMEMOPT=1, as mimalloc-bench builds it
#   barnes        a gets() shim: musl declares no gets (CheriBSD's side has the same shim)
#   sh6bench      -DBENCH=1 -DSYS_MULTI_THREAD=1, as mimalloc-bench builds it; its OS switch
#                 knows no Capstone, so __CAPSTONE__ is added where __linux__ selects POSIX
#   mstress, mleak  <stdatomic.h> is a compiler header the SDK's -nostdinc leaves out; only that
#                 header is added, from the SDK's own compiler
#   espresso, glibc-simple: unchanged
# Not built: glibc-thread and xmalloc-test run for a fixed time, not a fixed amount of work;
# sh8bench stores a pointer into a block of blockSizeHistogram[0].size = 8 bytes
# (sh8bench-new.c:347-352): with 16-byte capabilities that store overflows the block, and
# Capstone's exact bounds stop it (cause 28); CheriBSD's off arm aborts too. Not built.
# rptest needs two compiler features Capstone lacks: __sync_bool_compare_and_swap on a pointer
# lowers to __atomic_compare_exchange_16, which nothing provides (capability atomics reach only
# the generic __atomic_* calls, C-54), and it keeps tag bits in a pointer and recovers the
# pointer through uintptr_t, which on Capstone carries no capability. The C++ programs (alloc-test, malloc-large, larson, cache-scratch, cache-thrash) need C++.
# No allocator option is set.
#   usage: build-domain.sh <sdk dir> <mimalloc-bench bench/ dir> <out dir>
# Writes <out>/<name>.dom and <out>/<name>-traced.dom (the same program linked with
# mqtrace-cap.c through -Wl,--wrap, see there) and copies the two inputs (largest.espresso,
# barnes input), and builds mb-run, the guest-side runner that reports the launcher's peak RSS
# (native Linux, needs CROSS_COMPILE). MALLOCBENCH_OPT selects the optimisation level (default
# -O3, as on CheriBSD).
set -euo pipefail
SDK=${1:?usage: build-domain.sh <sdk> <bench dir> <out>}
B=${2:?usage: build-domain.sh <sdk> <bench dir> <out>}
OUT=${3:?usage: build-domain.sh <sdk> <bench dir> <out>}
CC="$SDK/capstone-cc"
OPT=${MALLOCBENCH_OPT:--O3}
HOSTED=(-fhosted -fbuiltin)
W=("${HOSTED[@]}" -Wno-everything -Wno-implicit-function-declaration -Wno-implicit-int -Wno-int-conversion -Wno-incompatible-pointer-types)
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRACE=("$HERE/mqtrace-cap.c"
       -Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free,--wrap=posix_memalign
       -Wl,--wrap=aligned_alloc
       -Wl,--wrap=__clone,--wrap=__capstone_delegate_thread_attach,--wrap=__capstone_signals_thread_detach)
mkdir -p "$OUT/src"
# dom <name> <compiler arguments...>: the plain and the traced program.
dom() {
  local name=$1; shift
  "$CC" "$OPT" "${W[@]}" "$@" -o "$OUT/$name.dom"
  "$CC" "$OPT" "${W[@]}" "$@" "${TRACE[@]}" -o "$OUT/$name-traced.dom"
}

cfrac=(cfrac.c pops.c pconst.c pio.c pabs.c pneg.c pcmp.c podd.c phalf.c padd.c psub.c pmul.c
       pdivmod.c psqrt.c ppowmod.c atop.c ptoa.c itop.c utop.c ptou.c errorp.c pfloat.c pidiv.c
       pimod.c picmp.c primes.c pcfrac.c pgcd.c)
dom cfrac -DNOMEMOPT=1 "${cfrac[@]/#/$B/cfrac/}" -lm

espresso=(cofactor.c cols.c compl.c contain.c cubestr.c cvrin.c cvrm.c cvrmisc.c cvrout.c
          dominate.c equiv.c espresso.c essen.c exact.c expand.c gasp.c getopt.c gimpel.c
          globals.c hack.c indep.c irred.c main.c map.c matrix.c mincov.c opo.c pair.c part.c
          primes.c reduce.c rows.c set.c setc.c sharp.c sminterf.c solution.c sparse.c unate.c
          utility.c verify.c)
dom espresso "${espresso[@]/#/$B/espresso/}" -lm

printf '#include <stdio.h>\nstatic char *mq_gets(char *b, int n) { char *p; if (!fgets(b, n, stdin)) return 0; for (p = b; *p; p++) if (*p == 0x0a) { *p = 0; break; } return b; }\n#define gets(b) mq_gets(b, sizeof(b))\n' > "$OUT/src/mq-gets.h"
barnes=(code.c code_io.c load.c grav.c getparam.c util.c)
# barnes's random number generator (util.c prand) relies on signed int overflow:
# randx = (A*randx+B) & MASK. clang 22 (this SDK's, for riscv64 as well as capstone64) infers
# from the overflow being undefined that the product is non-negative and drops the mask, so
# the generator produces negative values, the initial bodies coincide and barnes stops with
# "not enough levels in tree" after exit 0. gcc (native) and clang 17 (CheriBSD) keep the
# mask. -fwrapv defines the overflow as the two's complement those compilers produce.
dom barnes -fwrapv -include "$OUT/src/mq-gets.h" "${barnes[@]/#/$B/barnes/}" -lm

dom glibc-simple "$B/glibc-bench/bench-malloc-simple.c" -lpthread
# mstress takes one atomic pointer exchange from <stdatomic.h>, a compiler header that the SDK's
# -nostdinc leaves out; only that header is added, from the SDK's own compiler.
RES=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cc"])' "$SDK/sdk.json")
mkdir -p "$OUT/src/atomic-include"
cp "$("$RES" -print-resource-dir)/include/stdatomic.h" "$OUT/src/atomic-include/"
dom mstress -isystem "$OUT/src/atomic-include" "$B/mstress/mstress.c" -lpthread

for sh in sh6bench; do
  cp "$B/shbench/$sh-new.c" "$OUT/src/"
  sed -i 's/|| defined(__linux__)/& || defined(__CAPSTONE__)/' "$OUT/src/$sh-new.c"
  dom $sh -DBENCH=1 -DSYS_MULTI_THREAD=1 "$OUT/src/$sh-new.c" -lpthread
done
dom mleak -isystem "$OUT/src/atomic-include" "$B/mleak/mleak.c" -lpthread
# The tracer's positive control (see tracer-check.c), not a benchmark.
dom tracer-check "$HERE/tracer-check.c" -lpthread

cp "$B/espresso/largest.espresso" "$B/barnes/input" "$OUT/"
"${CROSS_COMPILE:?RISC-V Linux compiler prefix for mb-run}gcc" -O2 -static -Wall -Wextra -Werror \
  "$HERE/mb-run.c" -o "$OUT/mb-run"
sha256sum "$OUT"/*.dom
