#!/usr/bin/env bash
# Build the Capstone toolchain the way the host rules require (2026-09-07, after the 5-6
# September outage; ~/bin/logs/AGREED-RESOURCE-RULES.md):
#
#   1. take the MACHINE-WIDE exclusive lock -- any job capped at 20 GiB or more, OR using 8 cores
#      or more sustained, OR any timing measurement holds it (the rule was broadened from memory
#      alone on 2026-09-07 after a -j56 build landed on a benchmark run); ninja -j90 qualifies on
#      both counts, so this build never coincides with another lane's big job or measurement
#      (enforcement, not a promise). The file keeps its historical name. A common wrapper with a
#      holder sidecar is being written by another lane; when it lands, this script calls it;
#   2. run ninja inside a systemd-run scope with MemoryMax (fail-fast; no MemoryHigh: the host has
#      no swap, so a high watermark turns a kill into a silent multi-hour stall);
#   3. record the scope's memory.peak, so the cap can be re-sized from a measured number.
#
# Usage: build-toolchain.sh [-C <build dir>] [targets...]      (default: llc clang lld)
#   JOBS=90 (never 112: the parallel debug-link storm hangs the box), MEMMAX=64G,
#   MEMLOCK=$HOME/bin/logs/machine-memory.lock (a persistent path: /tmp is emptied at boot).
# Never run while a QEMU suite or twin run is in flight: the rebuilt shared libraries swap under it
# (toolchain-fresh.py says whether a rebuild is needed at all).
set -uo pipefail
BUILD=${CAPSTONE_LLVM_BUILD_DIR:-}
if [ "${1:-}" = "-C" ]; then BUILD=$2; shift 2; fi
[ -n "$BUILD" ] || BUILD=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/llvm/cmake-build-debug
[ -f "$BUILD/build.ninja" ] || { echo "build-toolchain: no build.ninja under $BUILD" >&2; exit 2; }
TARGETS=("$@"); [ ${#TARGETS[@]} -gt 0 ] || TARGETS=(llc clang lld)
JOBS=${JOBS:-90}; MEMMAX=${MEMMAX:-64G}
MEMLOCK=${MEMLOCK:-$HOME/bin/logs/machine-memory.lock}
for t in flock systemd-run ninja; do command -v "$t" >/dev/null || { echo "build-toolchain: $t not found" >&2; exit 2; }; done

echo "build-toolchain: waiting for the machine memory lock ($MEMLOCK) ..." >&2
exec 8>>"$MEMLOCK" || { echo "build-toolchain: cannot open $MEMLOCK" >&2; exit 2; }
flock -x 8
echo "build-toolchain: lock held $(date -Is); ninja -j$JOBS ${TARGETS[*]} in $BUILD under MemoryMax=$MEMMAX" >&2
# The inner shell reads its own cgroup's memory.peak before the scope disappears.
systemd-run --user --scope --quiet -p MemoryMax="$MEMMAX" -- bash -c '
  ninja -j"$1" -C "$2" "${@:3}"; rc=$?
  cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
  peak=$(cat "$cg/memory.peak" 2>/dev/null || echo "?")
  ev=$(tr "\n" " " < "$cg/memory.events" 2>/dev/null)
  echo "build-toolchain: ninja rc=$rc; scope memory.peak=$peak bytes ($(( ${peak:-0} / 1073741824 )) GiB); memory.events: $ev" >&2
  exit $rc' _ "$JOBS" "$BUILD" "${TARGETS[@]}"
rc=$?
flock -u 8
echo "build-toolchain: done rc=$rc $(date -Is)" >&2
exit $rc
