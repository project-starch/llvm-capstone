#!/usr/bin/env bash
# Build the published PoisonCap platform with the recipe in
# ports/ffmpeg/buffer-pool/host/cheribsd/poisoncap/ (prepare.py, platform.sh), one stage after
# another, each timed.
#
#   bash build-platform.sh [stage ...]     stages: prepare verify llvm qemu cheribsd image
#                                          (default: all, in that order)
#
# POISONCAP_PLATFORM (default ~/poisoncap-platform) holds work/, logs/ and the pinned cheribuild
# (cheribuild/, at platform.json's cheribuild_commit). Each stage logs to logs/<stage>.log and
# appends one line to logs/STAGES.tsv; logs/DRIVE_DONE gets the overall rc at the end.
#
# The supported way to run it is build-platform-docker.sh, which runs this script inside an Ubuntu
# 24.04 container with clang-18, the host the recipe was written for, and sets POISONCAP_HOST_COMPAT=0.
#
# POISONCAP_HOST_COMPAT=1 (the default when run directly) adds workarounds for a newer host, as far
# as they were taken on p13 (Ubuntu 26.04: clang-21, GCC 15 libstdc++, a glibc whose strchr returns
# const char * for a const argument) before the CheriBSD bootstrap failed in libstdc++'s thread
# headers and the build moved into the container. None of them changes a published source file:
#   - POISONCAP_HOST_CC/CXX name the host compiler for every stage (default clang-21 here);
#   - POISONCAP_HOST_CXXFLAGS="-include cstdint" for the LLVM build, whose SmallVector.h uses
#     uint64_t without including <cstdint>;
#   - the host compiler is a wrapper adding -Wno-error=incompatible-pointer-types-discards-qualifiers
#     after the build's own flags: CheriBSD's bootstrap libc (getopt_long.c) assigns strchr's
#     result to char * and builds with that warning as an error.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
REC=$REPO/capstone/ports/ffmpeg/buffer-pool/host/cheribsd/poisoncap
P=${POISONCAP_PLATFORM:-$HOME/poisoncap-platform}
WORK=$P/work
LOG=$P/logs
CC_REAL=${POISONCAP_REAL_CC:-clang-21}
CXX_REAL=${POISONCAP_REAL_CXX:-clang++-21}
mkdir -p "$LOG" "$WORK"
if [ "${POISONCAP_HOST_COMPAT:-1}" = 1 ]; then
  mkdir -p "$P/host-cc"
  for pair in "cc:$CC_REAL" "c++:$CXX_REAL"; do
    name=${pair%%:*}; real=$(command -v "${pair#*:}") || { echo "no ${pair#*:} on PATH" >&2; exit 2; }
    printf '#!/bin/sh\nexec %s "$@" -Wno-error=incompatible-pointer-types-discards-qualifiers\n' \
      "$real" > "$P/host-cc/$name"
    chmod 755 "$P/host-cc/$name"
  done
  export POISONCAP_HOST_CC=$P/host-cc/cc POISONCAP_HOST_CXX=$P/host-cc/c++
  export POISONCAP_HOST_CXXFLAGS=${POISONCAP_HOST_CXXFLAGS:--include cstdint}
else
  export POISONCAP_HOST_CC=$CC_REAL POISONCAP_HOST_CXX=$CXX_REAL
fi
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
