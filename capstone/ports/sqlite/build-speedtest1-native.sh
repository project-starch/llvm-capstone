#!/usr/bin/env bash
# Build the NATIVE speedtest1 baseline, and the oracle the domain run is checked against.
#
# It is the same shape as build-slt-native.sh and for the same reason: the domain build carries a
# long list of SQLITE_OMIT_* flags, so a baseline built from a plain default configuration makes an
# ordinary configuration difference look exactly like a capability defect. The defines are HARVESTED
# from build-sqlite-capstone.sh rather than retyped, so the two cannot drift apart silently.
#
# TWO SOURCES, NOT ONE. The amalgamation comes from fetch-sqlite.sh (SHA3-pinned); speedtest1.c is
# NOT in the amalgamation and comes from fetch-sqlite-src.sh, which pins the full tarball at the
# same version. Building the benchmark against a different engine version than the domain runs would
# be exactly the kind of mismatch this whole comparison exists to detect.
#
# THIS BINARY IS ALSO THE MATCHED BASELINE'S SOURCE OF TRUTH for which testsets are runnable at all
# under our defines: main, orm and parsenumber run clean; cte and star die on decimal literals
# (SQLITE_OMIT_FLOATING_POINT's tokenizer rejects a bare '.'), fp on round(), json on OMIT_JSON, and
# app on state another testset builds. See the header of sqlite_hostcall.h.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/speedtest1-native}
mkdir -p "$OUT_DIR"

SQLITE_SRC_DIR=$(bash "$SCRIPT_DIR/fetch-sqlite.sh")
FULL_SRC_DIR=$(bash "$SCRIPT_DIR/fetch-sqlite-src.sh")
SPEEDTEST1="$FULL_SRC_DIR/test/speedtest1.c"
[[ -f "$SPEEDTEST1" ]] || { echo "ERROR: no speedtest1.c at $SPEEDTEST1" >&2; exit 1; }
echo "== amalgamation: $SQLITE_SRC_DIR"
echo "== speedtest1:   $SPEEDTEST1"

# Identical exclusion list to build-slt-native.sh, and identical reasoning: these six select an
# OS/allocator rather than SQL semantics, and the native side has facilities the domain does not.
EXCLUDE='SQLITE_OS_OTHER|SQLITE_OMIT_AUTOINIT|SQLITE_ZERO_MALLOC|SQLITE_ENABLE_MEMSYS5|SQLITE_DEFAULT_LOOKASIDE|SQLITE_UNTESTABLE'
_blocks='/^SQLITE_DEFINES=(/,/^)/p'
[[ "${SQLITE_FEATURE_SET:-deployed}" == restored ]] && _blocks="$_blocks;/^SQLITE_RESTORE=(/,/^)/p"
mapfile -t DEFS < <(sed -n "$_blocks" "$SCRIPT_DIR/build-sqlite-capstone.sh" \
                    | grep -oE '\-[DU][A-Za-z0-9_]+(=[^ )]*)?' \
                    | grep -vE "^-[DU]($EXCLUDE)")
EXPECT_DEFS=21   # one variable, because a count written twice can disagree with itself

# SQLITE_FLOAT=on removes SQLITE_OMIT_FLOATING_POINT, and SQLITE_FULL=on adds json and rtree on top.
# Both are read IDENTICALLY by every consumer of the define list -- the silicon domain, the native
# oracle and the heap sweep -- because they harvest the same block by text, and a knob honoured by
# one of them and not the others is how a domain and its oracle drift apart without a word. Both are
# off by default, so every recorded board result keeps its exact define set.
#
# MEASURED 2026-09-10 natively, with the domain's own argument shape: floating point alone takes
# speedtest1 from three runnable testsets to SEVEN. cte and star stop failing on decimal literals,
# fp gets round(), and app gets unixepoch, because the omission force-defines
# SQLITE_OMIT_DATETIME_FUNCS as a side effect.
#
# FULL implies FLOAT because they are not independent: json uses sqlite3_value_double
# unconditionally and would compute integer "reals" without it, and rtree fails to LINK without
# INCRBLOB and to COMPILE without floating point. FLOAT stays separate so the 3/9-to-7/9 step can
# still be bisected on its own.
SQLITE_FULL=${SQLITE_FULL:-off}
SQLITE_FLOAT=${SQLITE_FLOAT:-off}
[[ "$SQLITE_FULL" == "on" ]] && SQLITE_FLOAT=on
if [[ "$SQLITE_FLOAT" == "on" ]]; then
  DEFS+=(-USQLITE_OMIT_FLOATING_POINT)
  EXPECT_DEFS=$(( EXPECT_DEFS + 1 ))
fi
if [[ "$SQLITE_FULL" == "on" ]]; then
  DEFS+=(-USQLITE_OMIT_JSON -DSQLITE_ENABLE_RTREE=1 -USQLITE_OMIT_INCRBLOB)
  EXPECT_DEFS=$(( EXPECT_DEFS + 3 ))
fi

(( ${#DEFS[@]} == EXPECT_DEFS )) || {
  echo "ERROR: harvested ${#DEFS[@]} defines, expected $EXPECT_DEFS." >&2
  echo "  A FLOOR TEST USED TO LIVE HERE and it could not do this job: it required only more than" >&2
  echo "  ten, so sixteen defines could vanish and it still passed. It fired only on total sed" >&2
  echo "  failure, which is the one case that also breaks the build loudly downstream." >&2
  echo "  If SQLITE_DEFINES legitimately changed, re-bless this number DELIBERATELY -- both arms of" >&2
  echo "  every comparison read the same block, so a silent drop moves them together and no ratio" >&2
  echo "  would look wrong." >&2
  exit 1; }
echo "== semantic defines shared with the domain: ${#DEFS[@]}"

# The one upstream bug that -DSQLITE_OMIT_FLOATING_POINT trips over in 3.53.3, patched exactly as
# build-slt-native.sh and the domain build patch it, and gated the same way.
PATCHED="$OUT_DIR/sqlite3-speedtest1-native.c"
sed -e 's/sqlite3Atoi64(z, pResult, strlen(z), SQLITE_UTF8)/sqlite3Atoi64(zIn, pResult, strlen(zIn), SQLITE_UTF8)/' \
    "$SQLITE_SRC_DIR/sqlite3.c" > "$PATCHED"
grep -q 'sqlite3Atoi64(zIn, pResult, strlen(zIn), SQLITE_UTF8)' "$PATCHED" \
  || { echo "ERROR: the sqlite3AtoF fix did not apply -- upstream moved" >&2; exit 1; }

# -O1 to match the domain's SUPPORT_OPT and the SLT baseline, so the arms differ in the capability
# ABI and the clock, not in the optimiser.
cc -O1 -o "$OUT_DIR/speedtest1_native" \
  -I"$SQLITE_SRC_DIR" \
  "${DEFS[@]}" \
  "$SPEEDTEST1" "$PATCHED" \
  -lm

echo "Built $OUT_DIR/speedtest1_native"
