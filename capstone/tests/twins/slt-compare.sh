#!/usr/bin/env bash
# Compare a domain SQLLogicTest run against the native run of the same file.
#
#   slt-compare.sh <file.test> <domain-console-log> [label...]
#
# Builds (once) a native runner with SLT_MAX_REPORTED raised to $SLT_TWIN_CAP so that
# every failing record is listed rather than the first eight, runs it on the file,
# and hands both logs to slt-compare.py.  The domain run must have been built with the
# same cap (run-slt-twin.sh does that through DOMAIN_EXTRA_DEFS).  Exit code is the
# comparator's: 0 AGREE, 1 MISMATCH, 2 ERROR.  Result lines are appended to
# capstone/tests/twins/results/<date>.tsv -- result lines only, never captures.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TEST=${1:?usage: slt-compare.sh <file.test> <domain-log> [label]}
DOMLOG=${2:?usage: slt-compare.sh <file.test> <domain-log> [label]}
shift 2
LABEL=${*:-$(basename "$TEST")}
[[ -f "$TEST" ]] || { echo "ERROR: no such test file: $TEST" >&2; exit 2; }
[[ -f "$DOMLOG" ]] || { echo "ERROR: no such domain log: $DOMLOG" >&2; exit 2; }

CAP=${SLT_TWIN_CAP:-256}
WORK=${SLT_TWIN_WORK:-$CAPSTONE_TMP_ROOT/twins}
mkdir -p "$WORK"
NATIVE="$WORK/slt_native_cap$CAP"
bash "$SCRIPT_DIR/build-slt-native-cap.sh" "$CAP" "$NATIVE"

NATLOG="$WORK/native-$(basename "$TEST").log"
"$NATIVE" "$TEST" > "$NATLOG" 2>&1 || true   # its exit status is the failure count, not an error

RESULTS=${SLT_TWIN_RESULTS:-$SCRIPT_DIR/results/$(date +%Y-%m-%d).tsv}
mkdir -p "$(dirname "$RESULTS")"
python3 "$SCRIPT_DIR/slt-compare.py" --native "$NATLOG" --domain "$DOMLOG" --cap "$CAP" \
  --label "$LABEL" --tsv "$RESULTS"
