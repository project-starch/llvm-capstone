#!/usr/bin/env bash
# THE GATE. Asserts the exact verdict of every arm in negative-control.test.
#
# WHY AN EXACT TALLY AND NOT "SOME FAILURES APPEARED": the failure this guards against is a
# comparator that silently stops discriminating -- a rendering rule that quietly matches
# everything, a sort that is never applied, a skip bucket that starts counting as a pass.
# Each of those still produces "some failures", so only the exact numbers catch it. Every
# count below corresponds to a labelled arm in the fixture.
#
# It also asserts the CAPPED run, because skip_big is the one bucket that can turn a
# not-evaluated record into an apparent pass, and it is invisible in the uncapped run.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../../tests/capstone-test-env.sh"

BIN=${SLT_NATIVE_BIN:-$CAPSTONE_TMP_ROOT/slt-native/slt_native}
[[ -x "$BIN" ]] || { echo "ERROR: $BIN missing -- run build-slt-native.sh" >&2; exit 1; }
FIXTURE="$SCRIPT_DIR/negative-control.test"
[[ -f "$FIXTURE" ]] || { echo "ERROR: $FIXTURE missing" >&2; exit 1; }

# Never pipe the runner into a filter and read $? -- the pipe would replace it. Redirect.
run_summary() {  # $1 = value cap ("" for default)
  local out rc
  out=$(mktemp)
  if [[ -n "$1" ]]; then SLT_MAX_VALUES="$1" "$BIN" "$FIXTURE" > "$out" 2>&1 || rc=$?
  else "$BIN" "$FIXTURE" > "$out" 2>&1 || rc=$?; fi
  grep -m1 '^SLT-SUMMARY' "$out" || { echo "NO SUMMARY LINE" ; cat "$out"; }
  rm -f "$out"
}

fail=0
check() {  # $1 = label, $2 = expected summary tail, $3 = actual
  if [[ "$3" == *"$2"* ]]; then
    echo "  ok   $1"
  else
    echo "  FAIL $1"
    echo "       expected to contain: $2"
    echo "       got:                 $3"
    fail=1
  fi
}

echo "== negative control, default cap"
GOT=$(run_summary "")
# 7 setup statements + 1 `statement error` that correctly errors = 8 passing statements.
# FAIL 1 and FAIL 2 are the two statement arms that must fail.
# 6 query arms must pass (nosort/rowsort/valuesort x value-form/hash-form, NULL, (empty), %.3f).
# FAIL 3..6 are the four query arms that must fail: wrong value, wrong md5, wrong count,
# too few expected values. SKIP 1/2 must land in skip_cond, NOT in a pass bucket.
check "tally" \
  "records=20 stmt_pass=8 stmt_fail=2 query_pass=6 query_fail=4 skip_big=0 skip_cond=2 parse_err=1 completed=1" \
  "$GOT"

echo "== negative control, cap=100 -- the skip_big bucket must fire and must NOT read as a pass"
GOT=$(run_summary 100)
# The four 500-value arms (2 passing, 2 failing) all move into skip_big.
check "capped tally" \
  "records=20 stmt_pass=8 stmt_fail=2 query_pass=4 query_fail=2 skip_big=4 skip_cond=2 parse_err=1 completed=1" \
  "$GOT"

if [[ $fail -ne 0 ]]; then
  echo "NEGATIVE CONTROL FAILED -- the comparator is not discriminating as designed" >&2
  exit 1
fi
echo "negative control PASSED: the comparator fails on all six wrong arms and skips two"
