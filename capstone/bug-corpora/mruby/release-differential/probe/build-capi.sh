#!/usr/bin/env bash
# Build every case whose trigger is a C-API driver (case.json "trigger": "capi.c") against one
# virtual mruby build, as OUT/capi-NN.dom. The driver links that build's libmruby.a, so the
# virtual-malloc and virtual-nested-pools drivers differ exactly as their mruby images do.
#
#   build-capi.sh <mruby tree: $MRBD_ROOT/src/mruby> <virtual SDK> <OUT>
#
# The SDK must be the one the mruby build used (build-mruby-domain.sh MRBD_SDK), whose directory
# also holds the spawn-shell.o that libmruby's IO HAL links against.
set -euo pipefail
M=${1:?mruby tree}; SDK=${2:?SDK}; OUT=${3:?OUT}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CORPUS=$(cd -- "$HERE/.." && pwd)
[[ -f "$M/build/capstone/lib/libmruby.a" ]] || { echo "build-capi: no $M/build/capstone/lib/libmruby.a" >&2; exit 2; }
[[ -f "$SDK/spawn-shell.o" ]] || { echo "build-capi: no $SDK/spawn-shell.o (build mruby with this SDK first)" >&2; exit 2; }
mkdir -p "$OUT"
n=0
for d in "$CORPUS"/[0-9][0-9]_*/; do
  trig=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["trigger"])' "$d/case.json")
  [[ $trig == *.c ]] || continue
  nn=$(basename "$d" | cut -c1-2)
  # The cross build's own flags (build_config.rb) that change mruby's headers' meaning.
  "$SDK/capstone-cc" -O1 -std=gnu99 -DPOOL_ALIGNMENT=16 -DMRB_NO_DIRECT_THREADING -DMRB_NO_BOXING \
    -I"$M/include" -I"$M/build/capstone/include" -I"$M/mrbgems/mruby-task/include" \
    "$d/$trig" "$M/build/capstone/lib/libmruby.a" "$SDK/spawn-shell.o" -o "$OUT/capi-$nn.dom"
  echo "build-capi: $OUT/capi-$nn.dom"
  n=$((n + 1))
done
[[ $n -gt 0 ]] || { echo "build-capi: no case names a .c trigger" >&2; exit 2; }
