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
# not-evaluated record into an apparent pass, and it is invisible in the uncapped run. And it
# asserts a run in which every allocation the runner makes for itself fails: those records
# must land in oom, not in skip_big, which once hid every empty valuesort result.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../../tests/capstone-test-env.sh"

BIN=${SLT_NATIVE_BIN:-$CAPSTONE_TMP_ROOT/slt-native/slt_native}
[[ -x "$BIN" ]] || { echo "ERROR: $BIN missing -- run build-slt-native.sh" >&2; exit 1; }
FIXTURE="$SCRIPT_DIR/negative-control.test"
[[ -f "$FIXTURE" ]] || { echo "ERROR: $FIXTURE missing" >&2; exit 1; }

# Never pipe the runner into a filter and read $? -- the pipe would replace it. Redirect.
run_summary() {  # $1 = value cap ("" for default), $2 = "fail-alloc" to fail the runner's allocations
  local out rc
  out=$(mktemp)
  if [[ "${2:-}" == fail-alloc ]]; then SLT_FAIL_RUNNER_ALLOC=1 "$BIN" "$FIXTURE" > "$out" 2>&1 || rc=$?
  elif [[ -n "$1" ]]; then SLT_MAX_VALUES="$1" "$BIN" "$FIXTURE" > "$out" 2>&1 || rc=$?
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
# 7 setup statements + 2 `statement error` arms that correctly error = 9 passing statements.
# The second of those pins that a float literal is a SYNTAX ERROR under this build's
# SQLITE_OMIT_FLOATING_POINT; if it ever starts failing, the build gained floating point.
# FAIL 1 and FAIL 2 are the two statement arms that must fail.
# 7 query arms must pass (nosort/rowsort/valuesort x value-form/hash-form, NULL, (empty), %.3f,
# and an empty valuesort result). FAIL 3..7 are the five query arms that must fail: wrong
# value, wrong md5, wrong count, too few expected values, an empty result where one value is
# expected. SKIP 1/2 must land in skip_cond, NOT in a pass bucket.
check "tally" \
  "records=23 stmt_pass=9 stmt_fail=2 query_pass=7 query_fail=5 skip_big=0 oom=0 skip_cond=2 parse_err=1 completed=1" \
  "$GOT"

echo "== negative control, cap=100 -- the skip_big bucket must fire and must NOT read as a pass"
GOT=$(run_summary 100)
# The four 500-value arms (2 passing, 2 failing) all move into skip_big.
check "capped tally" \
  "records=23 stmt_pass=9 stmt_fail=2 query_pass=5 query_fail=3 skip_big=4 oom=0 skip_cond=2 parse_err=1 completed=1" \
  "$GOT"

echo "== negative control, the runner's own allocations fail -- they must land in oom"
GOT=$(run_summary "" fail-alloc)
# The ten query arms that store values (six passing, four failing) all move into oom: none
# may pass, and none may be counted as skipped for size. The two empty-result arms store
# nothing, need no allocation, and keep their verdicts. Statements are unaffected.
check "failed-allocation tally" \
  "records=23 stmt_pass=9 stmt_fail=2 query_pass=1 query_fail=1 skip_big=0 oom=10 skip_cond=2 parse_err=1 completed=1" \
  "$GOT"

if [[ $fail -ne 0 ]]; then
  echo "NEGATIVE CONTROL FAILED -- the comparator is not discriminating as designed" >&2
  exit 1
fi
echo "negative control PASSED: the comparator fails on all seven wrong arms, skips two, and counts a failed allocation as oom"
