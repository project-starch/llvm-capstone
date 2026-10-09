#!/usr/bin/env bash
# The unprotected reference for the Capstone side: the same programs, arguments and fixes as
# build-domain.sh, built natively and statically against musl 1.2.5 (musl-gcc), so the
# allocator is the same mallocng without revocation at free. Each program once plain and once
# with ../mqtrace-cap.c built with -DMQ_NATIVE. Hosted compiler with builtins, as on CheriBSD.
#   usage: build-native.sh <mimalloc-bench bench/ dir> <out dir>
set -euo pipefail
B=${1:?usage: build-native.sh <bench dir> <out>}
OUT=${2:?usage: build-native.sh <bench dir> <out>}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CC=(musl-gcc -O3 -static -w -std=gnu11 -fcommon -Wno-implicit-int -Wno-implicit-function-declaration
    -Wno-int-conversion -Wno-incompatible-pointer-types -Wno-return-mismatch)
# gcc 14+ rejects these old-C forms by default; -Wno-return-mismatch keeps sh8bench's
# "return;" in a non-void thread function a warning, as older compilers had it.
TRACE=(-DMQ_NATIVE "$HERE/../mqtrace-cap.c"
       -Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free,--wrap=posix_memalign
       -Wl,--wrap=aligned_alloc)
mkdir -p "$OUT/src"
dom() {
  local name=$1; shift
  "${CC[@]}" "$@" -o "$OUT/$name"
  "${CC[@]}" "$@" "${TRACE[@]}" -o "$OUT/$name-traced"
}
cfrac=(cfrac.c pops.c pconst.c pio.c pabs.c pneg.c pcmp.c podd.c phalf.c padd.c psub.c pmul.c
       pdivmod.c psqrt.c ppowmod.c atop.c ptoa.c itop.c utop.c ptou.c errorp.c pfloat.c pidiv.c
       pimod.c picmp.c primes.c pcfrac.c pgcd.c)
dom cfrac -DNOMEMOPT=1 "${cfrac[@]/#/$B/cfrac/}" -lm
# espresso: musl's static getopt collides with the program's own getopt.c; libc's is used.
espresso=(cofactor.c cols.c compl.c contain.c cubestr.c cvrin.c cvrm.c cvrmisc.c cvrout.c
          dominate.c equiv.c espresso.c essen.c exact.c expand.c gasp.c gimpel.c
          globals.c hack.c indep.c irred.c main.c map.c matrix.c mincov.c opo.c pair.c part.c
          primes.c reduce.c rows.c set.c setc.c sharp.c sminterf.c solution.c sparse.c unate.c
          utility.c verify.c)
dom espresso "${espresso[@]/#/$B/espresso/}" -lm
printf '#include <stdio.h>\nstatic char *mq_gets(char *b, int n) { char *p; if (!fgets(b, n, stdin)) return 0; for (p = b; *p; p++) if (*p == 0x0a) { *p = 0; break; } return b; }\n#define gets(b) mq_gets(b, sizeof(b))\n' > "$OUT/src/mq-gets.h"
barnes=(code.c code_io.c load.c grav.c getparam.c util.c)
dom barnes -include "$OUT/src/mq-gets.h" "${barnes[@]/#/$B/barnes/}" -lm
dom glibc-simple "$B/glibc-bench/bench-malloc-simple.c" -lpthread
dom mstress "$B/mstress/mstress.c" -lpthread
for sh in sh6bench sh8bench; do
  dom $sh -DBENCH=1 -DSYS_MULTI_THREAD=1 "$B/shbench/$sh-new.c" -lpthread
done
dom mleak "$B/mleak/mleak.c" -lpthread
dom tracer-check "$HERE/../tracer-check.c" -lpthread
cp "$B/espresso/largest.espresso" "$OUT/"
cp "$B/barnes/input" "$OUT/barnes.input"
