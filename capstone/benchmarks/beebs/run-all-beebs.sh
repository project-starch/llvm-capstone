#!/usr/bin/env bash
set -euo pipefail

# Serial aggregate gate for all validated BEEBS wrappers. Individual
# run-beebs-*.sh scripts remain the authoritative diagnostic entry points.
# By default this wrapper keeps child output in per-benchmark logs to avoid
# flooding agent transcripts. Set RUN_ALL_BEEBS_VERBOSE=1 to stream child output.
# Pre-login QEMU boot hangs are retried by default; set
# RUN_ALL_BEEBS_LOGIN_TIMEOUT to cap only the boot-to-login wait during aggregate
# runs without changing per-benchmark guest execution timeouts.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

mapfile -t BEEBS_RUNNERS < <(
  find "$SCRIPT_DIR" -maxdepth 1 -name 'run-beebs-*.sh' \
    ! -name 'run-beebs-simple-common.sh' \
    | sort
)

if [[ ${#BEEBS_RUNNERS[@]} -eq 0 ]]; then
  echo "run-all-beebs.sh: no BEEBS runners found" >&2
  exit 1
fi

LOG_DIR=${RUN_ALL_BEEBS_LOG_DIR:-$CAPSTONE_TMP_ROOT/run-all-beebs}
VERBOSE=${RUN_ALL_BEEBS_VERBOSE:-0}
BOOT_RETRIES=${RUN_ALL_BEEBS_BOOT_RETRIES:-1}
LOGIN_TIMEOUT=${RUN_ALL_BEEBS_LOGIN_TIMEOUT:-90}
mkdir -p "$LOG_DIR"

passed=0
for runner in "${BEEBS_RUNNERS[@]}"; do
  name=$(basename "$runner" .sh)
  attempt=1
  max_attempts=$((BOOT_RETRIES + 1))

  while :; do
    log_file=$LOG_DIR/$name.attempt-$attempt.log
    if [[ $attempt -eq 1 ]]; then
      echo "run-all-beebs.sh: running $name"
    else
      echo "run-all-beebs.sh: retrying $name after pre-login QEMU boot failure (attempt $attempt/$max_attempts)"
    fi

    set +e
    if [[ "$VERBOSE" == 1 ]]; then
      CAPSTONE_QEMU_LOGIN_TIMEOUT="$LOGIN_TIMEOUT" bash "$runner" 2>&1 | tee "$log_file"
      status=${PIPESTATUS[0]}
    else
      if CAPSTONE_QEMU_LOGIN_TIMEOUT="$LOGIN_TIMEOUT" bash "$runner" >"$log_file" 2>&1; then
        status=0
      else
        status=$?
      fi
    fi
    set -e

    if [[ $status -eq 0 ]]; then
      break
    fi

    if [[ $attempt -lt $max_attempts ]] &&
       grep -q "QEMU stopped before the guest login prompt appeared" "$log_file"; then
      attempt=$((attempt + 1))
      continue
    fi

    echo "run-all-beebs.sh: FAIL $name (exit=$status, log=$log_file)" >&2
    echo "run-all-beebs.sh: recent log tail:" >&2
    tail -80 "$log_file" >&2 || true
    exit "$status"
  done

  marker=$(grep -Eo '__BEEBS_[A-Z0-9_]+_PASSED__' "$log_file" | tail -1 || true)
  if [[ -n "$marker" ]]; then
    echo "run-all-beebs.sh: PASS $name ($marker, log=$log_file)"
  else
    echo "run-all-beebs.sh: PASS $name (log=$log_file)"
  fi
  passed=$((passed + 1))
done

echo "run-all-beebs.sh: all $passed BEEBS runners completed (logs=$LOG_DIR)"
