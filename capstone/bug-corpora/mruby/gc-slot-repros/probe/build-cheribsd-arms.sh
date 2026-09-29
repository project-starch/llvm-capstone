#!/usr/bin/env bash
# Builds the two CheriBSD purecap mruby arms this corpus measures PoisonCap with.
#
#   baseline  the pin plus the purecap PORTING patches only (0001, 0002, 0003). This is
#             the CHERI control: upstream behaviour, one capability over the whole entry
#             array, libc revocation the only temporal mechanism present.
#   poison    the same plus 0009 (index-carrying walks), 0010 (sublet, compiled out here)
#             and 0012 (MRB_POISONCAP_HASH). Selected at run time by MRB_POISON_MODE.
#
# Both arms are needed. The baseline is what says whether a fault is the defect being
# caught or a path CHERI's bounds already fault on -- measured, four of the sub-cases
# fault on the baseline with no adapter present at all.
#
# Usage: CHERI_SDK=... CHERI_SYSROOT=... MRUBY_SRC=... ./build-cheribsd-arms.sh <outdir>
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PORT=$HERE/../../../../ports/mruby/app
PIN=${PIN:-9d523e2f74f2e63ca02840937523de61398a617d}
OUT=${1:?usage: build-cheribsd-arms.sh <outdir>}
: "${CHERI_SDK:?set CHERI_SDK to the PoisonCap SDK}"
: "${CHERI_SYSROOT:?set CHERI_SYSROOT to the purecap rootfs}"
: "${MRUBY_SRC:?set MRUBY_SRC to an mruby checkout or clone source}"
PORTING="0001-embed-len-bits-for-16-byte-pointers 0002-symbol-literal-flag-keeps-the-tag
         0003-gc-region-align-keeps-the-tag"
POISON="$PORTING 0009-hash-entry-walks-carry-an-index 0010-hash-entry-slots-under-sublet
         0012-hash-entry-slots-under-poison"
build_arm() {
  local name=$1 defines=$2; shift 2
  local tree=$OUT/$name
  rm -rf "$tree"
  git clone -q --no-checkout --shared "$MRUBY_SRC" "$tree"
  git -C "$tree" checkout -q "$PIN"
  for p in $@; do git -C "$tree" apply "$PORT/patches/4.0.0-rc2/$p.patch"; done
  sed "s/MRB_NO_BOXING/MRB_NO_BOXING $defines/" "$PORT/cheribsd_config.rb" \
    > "$tree/cheribsd_config.rb"
  ( cd "$tree" && MRUBY_CONFIG=cheribsd_config.rb ./minirake -j"${JOBS:-8}" \
      > "$OUT/$name-build.log" 2>&1 )
  echo "$name: $tree/build/cheribsd/bin/mruby"
}
mkdir -p "$OUT"
build_arm baseline ""                    $PORTING
build_arm poison   MRB_POISONCAP_HASH    $POISON
