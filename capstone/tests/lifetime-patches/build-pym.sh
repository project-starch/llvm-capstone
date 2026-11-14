#!/usr/bin/env bash
# build-pym.sh: the standalone pymalloc port's obmalloc.c, of the fixed (B) and of the original (O)
# patch 0003, with inject.h, linked with pym-test.c and the port's backing.c through the trap SDK.
# Expects $L/pytB and $L/pytO (Python-3.13.7 Objects and Include with 0001-0003 applied) and the
# port's src/shared in $L/pymsrc.
set -euo pipefail
L=$HOME/asr-work/life-c T=$L/test O=$L/pym-test SDK=$HOME/asr-work/eval/prebuilt/sdk
RT=$HOME/asr-work/eval/tree/capstone/runtime/include
mkdir -p "$O"
F="-w -O1 -DPYM_ARENA_BYTES=16777216 -DPYM_META_BYTES=4194304 -I$L/pymsrc -I$RT"
$SDK/capstone-cc -c $F -o "$O/backing.o" "$L/pymsrc/backing.c"
$SDK/capstone-cc -c $F -o "$O/pym-test.o" "$T/pym-test.c"
for v in O B; do
  $SDK/capstone-cc -c $F -DPYMALLOC_PORT -DPy_BUILD_CORE -DWITH_PYMALLOC -DSIZEOF_SIZE_T=8 -DSIZEOF_INT=4 \
    -DSIZEOF_VOID_P=16 -I"$L/pyt$v/Include/internal" -include "$T/inject.h" -o "$O/obmalloc-$v.o" \
    "$L/pyt$v/Objects/obmalloc.c"
  $SDK/capstone-cc -o "$O/pym-test-$v.dom" "$O/pym-test.o" "$O/obmalloc-$v.o" "$O/backing.o"
done
