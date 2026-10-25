#!/usr/bin/env bash
# The native fix differential: every case, buggy then fixed, on the hosted build of the wmem port
# with wmem as released, as `program 0 N buggy|fixed`.
#
#   bash runners/run-native.sh <fresh outdir> [extra cmake args...]
#
# A case passes when the buggy sequence reaches storage outside its object (VERDICT
# DEFECT-REPRODUCED) and the fixed one does not (VERDICT FIXED). Temporal rows observe ALIASING:
# in this run only, the next dissection reoccupies the freed storage with a marker the stale read
# must return (shared/driver.c, wm_reoccupy). Spatial rows observe WHERE the access went. Exit 0
# when every case is two-sided, 1 when one is not, 75 on a build or control failure.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/.." && pwd)
PORT=$(cd -- "$ROOT/../../../ports/wireshark/wmem" && pwd)
OUT=${1:?usage: run-native.sh <fresh outdir> [cmake args...]}; shift
[ -e "$OUT" ] && { echo "CONTROL-FAILED $OUT exists" >&2; exit 75; }
cmake --preset native -S "$PORT" -B "$OUT" -DWM_CORPUS_DIR="$ROOT" "$@" > "$OUT.configure.log" 2>&1 \
  && cmake --build "$OUT" -j "${JOBS:-8}" > "$OUT.build.log" 2>&1 \
  || { echo "CONTROL-FAILED build (see $OUT.build.log)" >&2; exit 75; }
status=0 seen=0
for dir in "$ROOT"/[0-9][0-9]_*/; do
  nn=$(basename "$dir" | cut -c1-2); n=$((10#$nn))
  prog=$(ls "$OUT"/bin/"$nn"-* 2>/dev/null | head -1)
  [ -x "$prog" ] || { echo "CONTROL-FAILED case $nn not built" >&2; exit 75; }
  ln -sf "$prog" "$OUT/bin/case-$nn"   # a stable name for tools/run-native-asan.py
  buggy=$("$prog" 0 "$n" buggy 2>&1); brc=$?
  fixed=$("$prog" 0 "$n" fixed 2>&1); frc=$?
  grep -q 'CONTROL-FAILED' <<<"$buggy$fixed" && { echo "CONTROL-FAILED case $nn: $(grep -m1 CONTROL-FAILED <<<"$buggy$fixed")" >&2; exit 75; }
  b=$(grep -o -m1 'VERDICT [A-Z-]*' <<<"$buggy"); f=$(grep -o -m1 'VERDICT [A-Z-]*' <<<"$fixed")
  printf '%s buggy rc=%d %s | fixed rc=%d %s\n' "$(basename "$dir")" "$brc" "$b" "$frc" "$f"
  [ "$b" = "VERDICT DEFECT-REPRODUCED" ] && [ $brc -eq 0 ] && [ "$f" = "VERDICT FIXED" ] && [ $frc -eq 0 ] || status=1
  seen=$((seen + 1))
done
[ $seen -gt 0 ] || { echo "CONTROL-FAILED no case ran" >&2; exit 75; }
exit $status
