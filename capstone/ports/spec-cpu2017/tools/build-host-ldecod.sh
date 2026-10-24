#!/bin/sh
# Build 525.x264_r's input generator, the JM reference decoder ldecod, for the
# BUILD HOST, from a licensed SPEC tree. tools/capstone-run.sh runs this build in
# place of the Capstone ldecod.
#
# Why: ldecod initialises its field-coded context tables with
# IBIARI_CTX_INIT2(NUM_BLOCK_TYPES, ...) over INIT_FLD_MAP_I[1][8][15][2],
# 10 rows of a table that has 8 (inc/defines.h, inc/ctx_tables.h). Natively
# rows 8 and 9 read the table after it; on Capstone the bounds stop the read
# (cause 28 in biari_init_context). ldecod only decodes the reference .264
# into the .yuv that x264 encodes, so the measured program is unaffected.
# CHERI's SPEC runs also ran ldecod on the host.
#
# usage: build-host-ldecod.sh SPEC_TREE OUTPUT   (CC selects the compiler)
set -eu
spec=${1:?usage: build-host-ldecod.sh SPEC_TREE OUTPUT}
out=${2:?usage: build-host-ldecod.sh SPEC_TREE OUTPUT}
src=$spec/benchspec/CPU/525.x264_r/src
cc=${CC:-cc}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
# The source list and flags are those of SPEC's own simple-build script.
srcs=$(grep -o 'ldecod_src/[A-Za-z0-9_]*\.c$' "$src/simple-build-ldecod_r-525.sh")
for f in $srcs; do
  o=$work/$(basename "$f" .c).o
  "$cc" -std=c99 -O2 -fcommon -fno-strict-aliasing -c -o "$o" \
    -DSPEC -DSPEC_CPU -DNDEBUG -DSPEC_AUTO_SUPPRESS_OPENMP \
    -DSPEC_AUTO_BYTEORDER=0x12345678 -DSPEC_LP64 \
    -I"$src/ldecod_src/inc" -I"$src/x264_src" -I"$src/x264_src/extras" \
    -I"$src/x264_src/common" "$src/$f"
done
"$cc" -z muldefs -o "$out" "$work"/*.o -lm
echo "built $out from $(echo "$srcs" | wc -w) sources"
