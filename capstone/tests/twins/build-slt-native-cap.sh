#!/usr/bin/env bash
# Build the native SQLLogicTest runner with SLT_MAX_REPORTED raised.
#
#   build-slt-native-cap.sh <cap> <output-binary>
#
# Same sources, same semantic defines and the same one-line sqlite3AtoF patch as
# benchmarks/sqlite/build-slt-native.sh (which it runs first, so the patched
# amalgamation exists); the only difference is the cap, which that script cannot take.
# Rebuilds only when the output is older than its inputs.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"
SQL="$CAPSTONE_REPO_ROOT/capstone/benchmarks/sqlite"

CAP=${1:?usage: build-slt-native-cap.sh <cap> <output>}
OUT=${2:?usage: build-slt-native-cap.sh <cap> <output>}
NATIVE_DIR=${SLT_NATIVE_DIR:-$CAPSTONE_TMP_ROOT/slt-native}
PATCHED="$NATIVE_DIR/sqlite3-slt-native.c"

if [[ ! -f "$PATCHED" ]]; then
  bash "$SQL/build-slt-native.sh" > /dev/null
fi
[[ -f "$PATCHED" ]] || { echo "ERROR: $PATCHED missing after build-slt-native.sh" >&2; exit 2; }

if [[ -x "$OUT" && "$OUT" -nt "$PATCHED" && "$OUT" -nt "$SQL/slt/slt_runner.h" && "$OUT" -nt "$SQL/slt/slt_native.c" ]]; then
  exit 0
fi

# Harvest the defines exactly as build-slt-native.sh does, so the two binaries differ
# in the cap alone.
EXCLUDE='SQLITE_OS_OTHER|SQLITE_OMIT_AUTOINIT|SQLITE_ZERO_MALLOC|SQLITE_ENABLE_MEMSYS5|SQLITE_DEFAULT_LOOKASIDE|SQLITE_UNTESTABLE'
mapfile -t DEFS < <(sed -n '/^SQLITE_DEFINES=(/,/^)/p' "$SQL/build-sqlite-capstone.sh" \
                    | grep -oE '\-D[A-Za-z0-9_]+(=[^ )]*)?' \
                    | grep -vE "^-D($EXCLUDE)")
(( ${#DEFS[@]} > 10 )) || { echo "ERROR: harvested only ${#DEFS[@]} defines -- the list moved" >&2; exit 2; }

SQLITE_SRC_DIR=$(dirname "$PATCHED")
cc -O1 -o "$OUT" -DSLT_MAX_REPORTED="${CAP}u" \
  -I"$SQLITE_SRC_DIR" -I"$SQL/slt" \
  "${DEFS[@]}" \
  "$SQL/slt/slt_native.c" "$PATCHED" -lm
echo "Built $OUT (SLT_MAX_REPORTED=$CAP)"
