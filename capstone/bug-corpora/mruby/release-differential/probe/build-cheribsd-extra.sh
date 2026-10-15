#!/usr/bin/env bash
# What the CheriBSD arm needs beyond ports/mruby/cheribsd/build.sh's interpreter, linked from
# that build's own objects so nothing is recompiled that the interpreter already has:
#
#   build-cheribsd-extra.sh <CheriBSD mruby tree: $ROOT/src> <OUT>
#
#   OUT/mruby-relink        mruby.o + libmruby.a relinked; must equal build/cheribsd/bin/mruby
#                           byte for byte, or the link line below is not the build's
#   OUT/mruby-qprobe        the same with the quarantine probe wrapped around malloc, calloc,
#                           realloc and free (-DQPROBE_WRAP; the binary is static)
#   OUT/capi-NN[-qprobe]    each C-API driver (case.json "trigger": "capi.c"), plain and probed
#   OUT/revocation-control  the platform's positive control: a stale read after a forced sweep
#
# CHERI_SDK and CHERI_SYSROOT as for build.sh. The probe is the one every CheriBSD quarantine
# reading of this project used (sha256 fa283f76...b97263554), copied in the result bundles.
set -euo pipefail
M=${1:?CheriBSD mruby tree}; OUT=${2:?OUT}
: "${CHERI_SDK:?}" "${CHERI_SYSROOT:?}"
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CORPUS=$(cd -- "$HERE/.." && pwd)
BC=$(cd -- "$CORPUS/../.." && pwd)
PROBE=$BC/cpython/pymalloc-repros/results/20261008-cheribsd/quarantine-probe.c
CONTROL=$BC/../ports/memcached/allocators/security-tests/cheribsd/revocation-control.c
[[ $(sha256sum "$PROBE" | cut -c1-16) == fa283f7686c36e18 ]] || { echo "probe source changed: $PROBE" >&2; exit 2; }
B=$M/build/cheribsd
CH=(--target=riscv64-unknown-freebsd13 --sysroot="$CHERI_SYSROOT" -march=rv64imafdcxcheri
    -mabi=l64pc128d -mno-relax -B"$CHERI_SDK/bin")
CC=("$CHERI_SDK/bin/clang" "${CH[@]}")
WRAP=-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free
mkdir -p "$OUT"
"${CC[@]}" -static -o "$OUT/mruby-relink" "$B/mrbgems/mruby-bin-mruby/tools/mruby/mruby.o" "$B/lib/libmruby.a" -lm
cmp -s "$OUT/mruby-relink" "$B/bin/mruby" || { echo "relink differs from $B/bin/mruby" >&2; exit 2; }
"${CC[@]}" -O1 -g -DQPROBE_WRAP -c "$PROBE" -o "$OUT/qprobe.o"
"${CC[@]}" -static $WRAP -o "$OUT/mruby-qprobe" "$B/mrbgems/mruby-bin-mruby/tools/mruby/mruby.o" \
  "$OUT/qprobe.o" "$B/lib/libmruby.a" -lm
for d in "$CORPUS"/[0-9][0-9]_*/; do
  trig=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["trigger"])' "$d/case.json")
  [[ $trig == *.c ]] || continue
  nn=$(basename "$d" | cut -c1-2)
  # the cross build's own flags (ports/mruby/cheribsd/build_config.rb)
  flags=(-O1 -g -std=gnu99 -DPOOL_ALIGNMENT=16 -DMRB_NO_DIRECT_THREADING -DMRB_NO_BOXING
         -I"$M/include" -I"$B/include" -I"$M/mrbgems/mruby-task/include")
  "${CC[@]}" "${flags[@]}" -c "$d/$trig" -o "$OUT/capi-$nn.o"
  "${CC[@]}" -static -o "$OUT/capi-$nn" "$OUT/capi-$nn.o" "$B/lib/libmruby.a" -lm
  "${CC[@]}" -static $WRAP -o "$OUT/capi-$nn-qprobe" "$OUT/capi-$nn.o" "$OUT/qprobe.o" "$B/lib/libmruby.a" -lm
done
"${CC[@]}" -O1 -g -static -o "$OUT/revocation-control" "$CONTROL"
(cd "$OUT" && sha256sum mruby-relink mruby-qprobe capi-* revocation-control | grep -v '\.o$')
