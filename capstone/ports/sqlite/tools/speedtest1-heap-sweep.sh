#!/usr/bin/env bash
# THE HEAP SWEEP THAT SETS THE BOARD GEOMETRY, committed so it can be re-derived and falsified.
#
# The figures it produces -- the smallest memsys5 arena in which each testset COMPLETES -- are what
# kill the size sweep, fix the board arena at 2 MiB, and bound the revoke-on-free arm. They were
# measured from an ad-hoc script under /tmp and nothing in the repo could reproduce them, which for
# numbers that decide how a boot is spent is the wrong side of the line.
#
#   usage: speedtest1-heap-sweep.sh [testset ...]     default: main orm parsenumber
#   out:   one line per (testset, size), naming the smallest arena that completed
#
# WHY IT NEEDS ITS OWN BINARY. build-speedtest1-native.sh is the correctness ORACLE and deliberately
# excludes SQLITE_ENABLE_MEMSYS5, SQLITE_ZERO_MALLOC and the other allocator-selecting defines,
# because they choose an OS/allocator rather than SQL semantics. Without memsys5,
# sqlite3_config(SQLITE_CONFIG_HEAP, ...) returns SQLITE_ERROR and speedtest1 prints
# "heap configuration failed: 1" -- which is how this sweep first read as a uniform failure. So this
# builds a SEPARATE binary that keeps memsys5, and it is a measurement instrument, never an oracle.
#
# NO POSITIONAL DATABASE NAME, and that is not a detail. speedtest1 REOPENS the database by name in
# testset_app's test 110, so passing `:memory:` positionally hands it a fresh EMPTY database and app
# fails with "no such table: config" at any arena size. This tool did exactly that until 2026-09-10
# and reported "app DOES NOT COMPLETE at any arena up to 33554432 bytes", which reads as a finding
# about memory and is a finding about the argument shape. The domain passes no positional name
# either (run-speedtest1-measure.sh), so omitting it is also what makes this sweep match the thing it
# is measuring.
#
# WHAT IT MEASURES AND WHAT IT DOES NOT. Empirical pass/fail per arena size, not a high-water
# reading: memsys5 is a buddy allocator and fragments, so "the peak was N bytes" does not imply an
# N-byte arena works. The two are separately interesting and the domain's own allocation census
# (SPEEDTEST1_ALLOCSTATS=1 on run-speedtest1-measure.sh) reports the peak.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PORT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$PORT_DIR/../../tests/capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/speedtest1-heap-sweep}
mkdir -p "$OUT_DIR"

SQLITE_SRC_DIR=$(bash "$PORT_DIR/fetch-sqlite.sh")
SPEEDTEST1=$(bash "$PORT_DIR/fetch-sqlite-src.sh")/test/speedtest1.c
[[ -f "$SPEEDTEST1" ]] || { echo "no speedtest1.c at $SPEEDTEST1" >&2; exit 1; }

# The domain's defines, keeping the allocator ones. Only the three that need a domain to exist are
# dropped, and SQLITE_ENABLE_MEMSYS5 is deliberately NOT among them.
EXCLUDE='SQLITE_OS_OTHER|SQLITE_OMIT_AUTOINIT|SQLITE_ZERO_MALLOC|SQLITE_DEFAULT_LOOKASIDE|SQLITE_UNTESTABLE'
mapfile -t DEFS < <(sed -n '/^SQLITE_DEFINES=(/,/^)/p' "$PORT_DIR/build-sqlite-capstone.sh" \
                    | grep -oE '\-[DU][A-Za-z0-9_]+(=[^ )]*)?' \
                    | grep -vE "^-[DU]($EXCLUDE)")
EXPECT_DEFS=22   # one variable, because a count written twice can disagree with itself

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
printf '%s\n' "${DEFS[@]}" | grep -q -- '-DSQLITE_ENABLE_MEMSYS5' || {
  echo "ERROR: memsys5 is not in the harvested set -- this sweep would measure the system allocator" >&2
  exit 1; }

PATCHED="$OUT_DIR/sqlite3-sweep.c"
sed -e 's/sqlite3Atoi64(z, pResult, strlen(z), SQLITE_UTF8)/sqlite3Atoi64(zIn, pResult, strlen(zIn), SQLITE_UTF8)/' \
    "$SQLITE_SRC_DIR/sqlite3.c" > "$PATCHED"
grep -q 'sqlite3Atoi64(zIn, pResult, strlen(zIn), SQLITE_UTF8)' "$PATCHED" \
  || { echo "ERROR: the sqlite3AtoF fix did not apply -- upstream moved" >&2; exit 1; }

cc -O1 -o "$OUT_DIR/st1_sweep" -I"$SQLITE_SRC_DIR" "${DEFS[@]}" "$SPEEDTEST1" "$PATCHED" -lm 2>/dev/null

# THE INSTRUMENT'S OWN CONTROL. An arena of 64 KiB must FAIL for every testset; if it passes, the
# --heap option is not reaching sqlite3_config and every "smallest arena" below is meaningless.
if "$OUT_DIR/st1_sweep" --testset main --size 1 --heap 65536 64 >/dev/null 2>&1; then
  echo "ERROR: main --size 1 completed in a 64 KiB arena -- --heap is not taking effect," >&2
  echo "       so every figure this sweep would print is void." >&2
  exit 1
fi
echo "control: a 64 KiB arena fails, so --heap reaches the allocator"

LADDER=${HEAP_LADDER:-262144 524288 1048576 1572864 2097152 3145728 4194304 6291456 8388608 16777216 33554432}
SIZES=${HEAP_SIZES:-1 5 20}
# "${@:-a b c}" expands the default as ONE word, so with no arguments the loop ran once for a
# testset literally named "main orm parsenumber" and every arm reported DOES NOT COMPLETE -- a
# uniform failure that looks like a finding about the allocator.
if (( $# )); then TESTSETS=("$@"); else TESTSETS=(main orm parsenumber); fi
for ts in "${TESTSETS[@]}"; do
  for size in $SIZES; do
    found=""
    for heap in $LADDER; do
      if "$OUT_DIR/st1_sweep" --testset "$ts" --size "$size" --heap "$heap" 64 >/dev/null 2>&1; then
        found=$heap; break
      fi
    done
    if [[ -n "$found" ]]; then
      printf '%-12s size=%-3s smallest arena that COMPLETES: %s bytes (%.2f MiB)\n' \
        "$ts" "$size" "$found" "$(echo "$found" | awk '{print $1/1048576}')"
    else
      printf '%-12s size=%-3s DOES NOT COMPLETE at any arena up to %s bytes\n' \
        "$ts" "$size" "${LADDER##* }"
    fi
  done
done
