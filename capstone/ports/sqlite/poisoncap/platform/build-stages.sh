#!/usr/bin/env bash
# Build the published PoisonCap platform with the recipe in
# ports/ffmpeg/buffer-pool/host/cheribsd/poisoncap/ (prepare.py, platform.sh), one stage after
# another, each timed. build.sh runs it in the container: the recipe needs the host it was written
# for (Ubuntu 24.04, clang-18).
#
#   bash build-stages.sh [stage ...]     stages: prepare verify llvm qemu cheribsd image
#                                          (default: all, in that order)
#
# POISONCAP_PLATFORM (default ~/poisoncap-platform) holds work/, logs/ and the pinned cheribuild
# (cheribuild/, at platform.json's cheribuild_commit). Each stage logs to logs/<stage>.log and
# appends one line to logs/STAGES.tsv; logs/DRIVE_DONE gets the overall rc at the end.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
REC=$REPO/capstone/ports/ffmpeg/buffer-pool/host/cheribsd/poisoncap
P=${POISONCAP_PLATFORM:-$HOME/poisoncap-platform}
WORK=$P/work
LOG=$P/logs
mkdir -p "$LOG" "$WORK"
export BUILD_JOBS=${BUILD_JOBS:-64}
export CHERIBUILD=${CHERIBUILD:-$P/cheribuild/cheribuild.py}
want=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cheribuild_commit"])' "$REC/platform.json")
have=$(git -C "$(dirname "$CHERIBUILD")" rev-parse HEAD 2>/dev/null)
[ "$have" = "$want" ] || { echo "cheribuild at '$have', the pin is $want" >&2; exit 2; }

[ -f "$LOG/STAGES.tsv" ] || printf 'stage\trc\tseconds\tfinished\n' > "$LOG/STAGES.tsv"
rm -f "$LOG/DRIVE_DONE"
run() {
  local name=$1; shift
  local t0; t0=$(date +%s)
  echo "=== $name start $(date -u +%FT%TZ)" > "$LOG/$name.log"
  "$@" >> "$LOG/$name.log" 2>&1 < /dev/null
  local rc=$?
  printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "$(( $(date +%s) - t0 ))" "$(date -u +%FT%TZ)" >> "$LOG/STAGES.tsv"
  df -h "$P" | tail -1 >> "$LOG/$name.log"
  return $rc
}
STAGES=("$@")
[ ${#STAGES[@]} -gt 0 ] || STAGES=(prepare verify llvm qemu cheribsd image)
rc=0
for st in "${STAGES[@]}"; do
  case $st in
    prepare) run prepare python3 "$REC/prepare.py" "$WORK" ;;
    verify)  run verify python3 "$REC/prepare.py" "$WORK" --verify ;;
    llvm|qemu|cheribsd|image) run "$st" bash "$REC/platform.sh" "$WORK" "$st" ;;
    *) echo "unknown stage $st" >&2; false ;;
  esac || { rc=$?; break; }
done
echo "rc=$rc $(date -u +%FT%TZ)" > "$LOG/DRIVE_DONE"
exit $rc
