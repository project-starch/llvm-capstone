#!/usr/bin/env bash
# Aggregate gate for the RV8 C benchmarks. Runs each run-rv8-<name>.sh, retries
# once on a transient QEMU boot flake, and reports a summary. Exits non-zero if
# any benchmark fails. `bigint` is C++ and deferred (no C++ runtime on the
# domain), so it is intentionally excluded.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
LOG_DIR=${LOG_DIR:-$CAPSTONE_TMP_ROOT/run-all-rv8-logs}
mkdir -p "$LOG_DIR"

BENCHES="dhrystone qsort sha512 aes primes norx miniz"

pass=0
fail=0
failed=""

for b in $BENCHES; do
  marker="__RV8_$(printf '%s' "$b" | tr '[:lower:]' '[:upper:]')_PASSED__"
  ok=0
  for attempt in 1 2; do
    log="$LOG_DIR/$b.attempt-$attempt.log"
    if bash "$SCRIPT_DIR/run-rv8-$b.sh" >"$log" 2>&1 && grep -q "$marker" "$log"; then
      ok=1
      break
    fi
    [ "$attempt" -eq 1 ] && echo "run-all-rv8: $b failed (attempt 1); retrying once (possible QEMU boot flake)" >&2
  done
  if [ "$ok" -eq 1 ]; then
    echo "PASS  $b"
    pass=$((pass + 1))
  else
    echo "FAIL  $b   (see $LOG_DIR/$b.attempt-*.log)"
    fail=$((fail + 1))
    failed="$failed $b"
  fi
done

echo "run-all-rv8: $pass passed, $fail failed.${failed:+ Failed:$failed}"
if [ "$fail" -eq 0 ]; then
  echo "__RV8_ALL_PASSED__"
  exit 0
fi
exit 1
