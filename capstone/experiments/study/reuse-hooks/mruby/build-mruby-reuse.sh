#!/usr/bin/env bash
# One mruby 4.0.0-rc2 interpreter with the reuse-distance observer at the GC's object slot
# hand-out and sweep free (pre-registration capstone/docs/plans/2026-10-11-nested-reuse-distance.md,
# 16d2171a3edc).
#
#   build-mruby-reuse.sh stock|protected VIRTUAL_SDK NEW_ROOT
#
# VIRTUAL_SDK: an application SDK built for the virtual profile, private to this build (the port
# script writes spawn-shell.o into it). protected = MRBD_SUBLET=1 (patch 0008, MRB_CAPSTONE_SUBLET).
#
# 1. ports/mruby/app/build-mruby-domain.sh, unchanged: clone at the pin, the port's patches
#    (0008 is applied in both arms; it is all under #ifdef MRB_CAPSTONE_SUBLET), a full build.
# 2. gc-reuse-gap.patch on that prepared source, the observer header beside gc.c, the build
#    directory removed, and the same script again from its mruby stage with
#    MRBD_DEFINES=-DEXP_REUSE_GAP, so every object is rebuilt with the one define.
# The patch is the same file for both arms: the hooked lines are the same text in both, outside
# 0008's #ifdef blocks (hand-out right after gc->live++ in mrb_obj_alloc, release in the sweep
# right after obj_free left the slot MRB_TT_FREE, before the slot is linked to the free list),
# with the GC's own slot pointer p in both.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAP=$(cd -- "$HERE/../../../.." && pwd)
[[ $# == 3 ]] || { echo 'usage: build-mruby-reuse.sh stock|protected VIRTUAL_SDK NEW_ROOT' >&2; exit 2; }
ARM=$1 SDK=$(realpath "$2") ROOT=$(realpath -m "$3")
case "$ARM" in
  stock) SUBLET=0 ;;
  protected) SUBLET=1 ;;
  *) echo "arm: stock or protected, not $ARM" >&2; exit 2 ;;
esac
[[ ! -e "$ROOT" ]] || { echo "use a fresh root: $ROOT" >&2; exit 2; }
SCRIPT=$CAP/ports/mruby/app/build-mruby-domain.sh
PATCH=$HERE/gc-reuse-gap.patch
M=$ROOT/src/mruby

MRBD_PIN=4.0.0-rc2 MRBD_SDK=$SDK MRBD_ROOT=$ROOT MRBD_SUBLET=$SUBLET bash "$SCRIPT"
patch -d "$M" --batch --forward --fuzz=0 -p1 < "$PATCH"
cp "$CAP/experiments/study/reuse-gap-observer.h" "$M/src/reuse-gap-observer.h"
rm -rf "$M/build"
echo "[reuse] $ARM: applied $(basename "$PATCH"), rebuilding with -DEXP_REUSE_GAP" >&2
MRBD_PIN=4.0.0-rc2 MRBD_SDK=$SDK MRBD_ROOT=$ROOT MRBD_SUBLET=$SUBLET MRBD_FROM=mruby \
  MRBD_DEFINES=-DEXP_REUSE_GAP bash "$SCRIPT"
IMG=$M/build/capstone/bin/mruby
grep -c 'REUSE-GAP gc-slots capacity=' "$IMG" >/dev/null \
  || { echo "the image has no REUSE-GAP report" >&2; exit 2; }
sha256sum "$PATCH" "$M/src/gc.c" "$M/src/reuse-gap-observer.h" "$IMG" > "$ROOT/reuse-build.sha256"
echo "[reuse] $ARM image $IMG" >&2
