#!/usr/bin/env bash
# One program per case, built against real libavutil and run twice. The control
# arm runs FIRST; if it does not hold, nothing below is a verdict. A sanitiser
# build runs too, and here it is expected to stay SILENT on both arms -- the read
# leaves the plane but not the allocation, so no redzone sits where it lands, and
# that silence is the row's finding rather than a gap.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$HERE/..
OUT=${1:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/ffmpeg-plane-repros}
FFSRC=${FFPLANE_SRC:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/ffmpeg-native-vidstab/src}
FFBUILD=${FFPLANE_BUILD:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/ffmpeg-native-vidstab/build}
mkdir -p "$OUT"
CC=${CC:-cc}
LIB=$FFBUILD/libavutil/libavutil.a
[ -f "$LIB" ] || { echo "CONTROL-FAILED no libavutil at $LIB" >&2; exit 75; }
# That build is compiled with ASan, so every binary here links it.
SAN=-fsanitize=address

status=0
for dir in "$ROOT"/[0-9][0-9]_*; do
  name=$(basename "$dir")
  n=${name%%_*}; n=${n#0}; n=${n:-0}
  "$CC" -O1 -g $SAN -o "$OUT/$name" "$dir/case.c" "$ROOT/shared/driver.c" \
    -I"$ROOT/shared" -I"$FFSRC" -I"$FFBUILD" "$LIB" -lm -lpthread \
    || { echo "CONTROL-FAILED build $name" >&2; exit 75; }

  fixed_out=$("$OUT/$name" fixed "$n" 2>&1); rc=$?
  if [ $rc -ne 0 ]; then echo "CONTROL-FAILED $name fixed arm rc=$rc" >&2; exit 75; fi
  case "$fixed_out" in *"VERDICT FIXED"*) ;; *) echo "CONTROL-FAILED $name control did not hold" >&2; exit 75 ;; esac

  buggy_out=$("$OUT/$name" buggy "$n" 2>&1); rc=$?
  if [ $rc -ne 0 ]; then echo "CONTROL-FAILED $name buggy arm rc=$rc" >&2; exit 75; fi

  san_quiet=1
  case "$fixed_out$buggy_out" in *AddressSanitizer*) san_quiet=0 ;; esac
  slack=$(printf '%s\n' "$buggy_out" | sed -n 's/.*plane_slack=\([0-9-]*\).*/\1/p' | head -1)

  printf '%s plain=%s asan=%s plane_slack=%s\n' "$name" \
    "$(case "$buggy_out" in *"VERDICT DEFECT-REPRODUCED"*) echo reproduced;; *) echo NO;; esac)" \
    "$([ $san_quiet -eq 1 ] && echo silent-as-expected || echo REPORTED)" "${slack:-?}"

  case "$buggy_out" in *"VERDICT DEFECT-REPRODUCED"*) ;; *) status=1 ;; esac
  [ $san_quiet -eq 1 ] || status=1
done
exit $status
