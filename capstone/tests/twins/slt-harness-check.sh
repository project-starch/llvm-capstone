#!/usr/bin/env bash
# THE GATE FOR THE GATE: proves slt-compare can produce every verdict before any domain
# result is read through it.  No QEMU; runs in seconds.  Exact exit codes are asserted,
# not "something non-zero", for the reason check-negative-control.sh gives: a comparator
# that stops discriminating still produces some failures.
#
# Arms:
#   AGREE     native fixture vs native fixture
#   MISMATCH  the "domain" fails a record the "native" side passes (the native run of the
#             fixture as the domain, the native run of a copy with FAIL 3 corrected as
#             the reference) -- and the line number of that record is named
#   MISMATCH  the same pair the other way round (a domain that PASSES a record the native
#             fails is a disagreement too)
#   ERROR     the domain log has no SLT-SUMMARY at all (a wedge, a trap, a create_dom
#             failure): absence must never read as a match
#   ERROR     the domain summary says completed=0
#   ERROR     the domain summary reports zero records
#   ERROR     the reported failures reach the cap (run with --cap 4 against 6 failures)
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"
FIX="$CAPSTONE_REPO_ROOT/capstone/benchmarks/sqlite/slt/negative-control.test"
[[ -f "$FIX" ]] || { echo "ERROR: $FIX missing" >&2; exit 2; }

WORK=$(mktemp -d "${CAPSTONE_TMP_ROOT:-/tmp}/slt-harness-check.XXXXXX")
CAP=256
NATIVE="$WORK/slt_native_cap"
bash "$SCRIPT_DIR/build-slt-native-cap.sh" "$CAP" "$NATIVE" > /dev/null || { echo "ERROR: native build failed" >&2; exit 2; }

"$NATIVE" "$FIX" > "$WORK/orig.log" 2>&1 || true
# FAIL 3 is "one wrong value": 9999999 where 1 is right.  Correct it and the copy has
# five failures where the original has six.
sed 's/^9999999$/1/' "$FIX" > "$WORK/corrected.test"
grep -q '^9999999$' "$FIX" || { echo "ERROR: fixture no longer has the FAIL 3 arm" >&2; exit 2; }
"$NATIVE" "$WORK/corrected.test" > "$WORK/corrected.log" 2>&1 || true
grep -v '^SLT-SUMMARY' "$WORK/orig.log" > "$WORK/nosummary.log"
sed 's/ completed=1/ completed=0/' "$WORK/orig.log" > "$WORK/incomplete.log"
sed 's/ records=[0-9]*/ records=0/' "$WORK/orig.log" > "$WORK/zero.log"

fail=0
arm() {  # $1 = label, $2 = expected rc, $3.. = comparator args
  local label=$1 want=$2; shift 2
  local out rc=0
  out=$(python3 "$SCRIPT_DIR/slt-compare.py" "$@" 2>&1) || rc=$?
  if [[ "$rc" == "$want" ]]; then
    echo "  ok   $label (rc=$rc): $(printf '%s' "$out" | cut -f2,3 | cut -c1-140)"
  else
    echo "  FAIL $label: expected rc=$want got rc=$rc"; echo "       $out"; fail=1
  fi
}

echo "== slt-compare positive controls (cap=$CAP)"
arm "AGREE    native vs native"                 0 --native "$WORK/orig.log" --domain "$WORK/orig.log" --cap $CAP
arm "MISMATCH domain fails a record native passes" 1 --native "$WORK/corrected.log" --domain "$WORK/orig.log" --cap $CAP
arm "MISMATCH domain passes a record native fails" 1 --native "$WORK/orig.log" --domain "$WORK/corrected.log" --cap $CAP
arm "ERROR    no domain summary"                2 --native "$WORK/orig.log" --domain "$WORK/nosummary.log" --cap $CAP
arm "ERROR    domain completed=0"               2 --native "$WORK/orig.log" --domain "$WORK/incomplete.log" --cap $CAP
arm "ERROR    domain ran zero records"          2 --native "$WORK/orig.log" --domain "$WORK/zero.log" --cap $CAP
arm "ERROR    failures reach the cap"           2 --native "$WORK/orig.log" --domain "$WORK/orig.log" --cap 4

# The MISMATCH arm must name the corrected record, not merely differ: the reported
# line must fall inside the FAIL 3 record's span (between its comment and FAIL 4's).
LO=$(grep -n '^# FAIL 3' "$FIX" | cut -d: -f1)
HI=$(grep -n '^# FAIL 4' "$FIX" | cut -d: -f1)
out=$(python3 "$SCRIPT_DIR/slt-compare.py" --native "$WORK/corrected.log" --domain "$WORK/orig.log" --cap $CAP 2>&1 || true)
N=$(printf '%s' "$out" | grep -o 'fails only in the domain: SLT-FAIL line=[0-9]*' | grep -o '[0-9]*$' || true)
if [[ -n "$N" && -n "$LO" && -n "$HI" && "$N" -gt "$LO" && "$N" -lt "$HI" ]]; then echo "  ok   MISMATCH names the FAIL 3 record (line=$N in $LO..$HI)"
else echo "  FAIL MISMATCH does not name the FAIL 3 record (got line=$N, span $LO..$HI): $out"; fail=1; fi

rm -rf "$WORK"
if [[ "$fail" -eq 0 ]]; then echo "slt-harness-check: ALL ARMS OK"; exit 0; fi
echo "slt-harness-check: FAILED"; exit 1
