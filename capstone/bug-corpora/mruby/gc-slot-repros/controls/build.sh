#!/usr/bin/env bash
# Build controls.c against one virtual mruby build, as one image for all four modes.
#
#   build.sh <mruby tree: $MRBD_ROOT/src/mruby> <virtual SDK> release|protected <OUT.dom>
#
# The tree and SDK are the ones build-mruby-domain.sh used (MRBD_SDK; release: MRBD_SUBLET=0,
# protected: MRBD_SUBLET=1, patch 0008). The flags are probe/build-capi.sh's (the cross build's
# own flags that change mruby's headers), plus -g for file:line attribution of a fault pc, plus,
# for protected, -DMRB_CAPSTONE_SUBLET, which the driver's mirror of gc.c's heap page layout
# needs exactly when libmruby was built with it. The script refuses a pairing it can check: a
# protected libmruby carries CDERIVE words, a release one none.
set -euo pipefail
M=${1:?mruby tree}; SDK=${2:?SDK}; ARM=${3:?release|protected}; OUT=${4:?OUT.dom}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAP=$(cd -- "$HERE/../../../.." && pwd)
LIB=$M/build/capstone/lib/libmruby.a
[[ -f $LIB ]] || { echo "build: no $LIB" >&2; exit 2; }
[[ -f $SDK/spawn-shell.o ]] || { echo "build: no $SDK/spawn-shell.o" >&2; exit 2; }
grep -q 'MRB_CAPSTONE_SUBLET' "$M/src/gc.c" || { echo "build: $M/src/gc.c lacks patch 0008" >&2; exit 2; }
case $ARM in
  release) extra=() ;;
  protected) extra=(-DMRB_CAPSTONE_SUBLET -I"$CAP/runtime/include") ;;
  *) echo "build: arm is release or protected, not $ARM" >&2; exit 2 ;;
esac
"$SDK/capstone-cc" -O1 -g -std=gnu99 -DPOOL_ALIGNMENT=16 -DMRB_NO_DIRECT_THREADING -DMRB_NO_BOXING \
  "${extra[@]}" -I"$M/include" -I"$M/build/capstone/include" \
  "$HERE/controls.c" "$LIB" "$SDK/spawn-shell.o" -o "$OUT"
echo "build: $ARM $OUT $(sha256sum "$OUT" | cut -c1-64)"
