#!/usr/bin/env bash
# S1 on silicon, the SQLite half: A1's six probes (capstone/experiments/a1-sqlite-reuse/probes.c) compiled
# into the MEASUREMENT harness (speedtest1_measure.c, -DSPEEDTEST1_PROBE_RT=1), one image per arm, the
# probe chosen at RUN time by `--s1-probe <n>`. Geometry is the P1 cells 5 and 6 (-O2, lookaside 1200,40,
# heap 2 MiB, stack declaration 385,024; the Sublet arm lent a 2 MiB arena and 1,750,285 B of tables).
# Each image earns its emulator pass record by running the BENCHMARK (run-speedtest1-measure.sh), because
# a protected probe that faults by design can never earn one.
#
#   CAPSTONE_LLVM_BIN=<toolchain with #119> bash build-images.sh <out-dir>
#
# Writes <out-dir>/{memsys5,sublet}/sqlite_silicon.dom; compare against SHA256SUMS here.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../.." && pwd)
OUT=${1:?usage: build-images.sh <out-dir>}
: "${CAPSTONE_LLVM_BIN:?set CAPSTONE_LLVM_BIN to the bin directory of the toolchain}"
export CAPSTONE_CLANG=${CAPSTONE_CLANG:-$CAPSTONE_LLVM_BIN/clang} CAPSTONE_LD_LLD=${CAPSTONE_LD_LLD:-$CAPSTONE_LLVM_BIN/ld.lld}
export SQLITE_OPT_LEVEL=-O2 SQLITE_LOOKASIDE=1200,40 SPEEDTEST1_STACK=385024 SPEEDTEST1_HEAP=2097152
export SPEEDTEST1_PROBE_SRC=$REPO/capstone/experiments/a1-sqlite-reuse/probes.c
mkdir -p "$OUT"
arm() {  # tag va [extra env...]
  local tag=$1 va=$2; shift 2
  echo "== $tag @ $va"
  ( export OUT_DIR=$OUT/$tag DOMAIN_BASE_VA=$va DOMAIN_EXTRA_DEFS="-DSPEEDTEST1_PROBE_RT=1" SPEEDTEST1_LOG_FILE=$OUT/$tag.bench.log "$@"
    bash "$REPO/capstone/ports/sqlite/run-speedtest1-measure.sh" ) > "$OUT/$tag.build.out" 2>&1
  echo "   $(sha256sum "$OUT/$tag/sqlite_silicon.dom" | cut -c1-16)  $(grep -a -o -E 'Verification Hash: [0-9a-f ]+' "$OUT/$tag.bench.log" | head -1)"
}
# 4 MiB apart, neither at k800's 0x10000 (R-3 / preflight C15).
arm memsys5 0x810000
arm sublet  0xc10000 SPEEDTEST1_SUBLET=1 SPEEDTEST1_SUBLET_ARENA=2097152 SPEEDTEST1_SUBLET_TABLES=1750285
