#!/usr/bin/env bash
set -euo pipefail

# Aggregate gate for all validated BEEBS wrappers. Individual run-beebs-*.sh
# scripts remain the authoritative diagnostic entry points. The default is still
# serial; set RUN_ALL_BEEBS_JOBS=N for opt-in parallelism.
# By default this wrapper keeps child output in per-benchmark logs to avoid
# flooding agent transcripts. Set RUN_ALL_BEEBS_VERBOSE=1 to stream child output.
# QEMU infra flakes before benchmark execution are retried by default; set
# RUN_ALL_BEEBS_LOGIN_TIMEOUT to cap only the boot-to-login wait.

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
BOOT_RETRIES=${RUN_ALL_BEEBS_BOOT_RETRIES:-2}
LOGIN_TIMEOUT=${RUN_ALL_BEEBS_LOGIN_TIMEOUT:-90}
JOBS=${RUN_ALL_BEEBS_JOBS:-1}
WORK_ROOT=${RUN_ALL_BEEBS_WORK_ROOT:-$LOG_DIR/work}
SHARED_BEEBS_SRC_DIR=${BEEBS_SRC_DIR:-$CAPSTONE_TMP_ROOT/beebs-src}
mkdir -p "$LOG_DIR"

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; then
  echo "run-all-beebs.sh: RUN_ALL_BEEBS_JOBS must be a positive integer, got '$JOBS'" >&2
  exit 1
fi

if ! [[ "$BOOT_RETRIES" =~ ^[0-9]+$ ]]; then
  echo "run-all-beebs.sh: RUN_ALL_BEEBS_BOOT_RETRIES must be a non-negative integer, got '$BOOT_RETRIES'" >&2
  exit 1
fi

"$SCRIPT_DIR/fetch-beebs.sh" >/dev/null

run_one() {
  local runner=$1
  local name
  name=$(basename "$runner" .sh)
  local attempt=1
  local max_attempts=$((BOOT_RETRIES + 1))
  local marker=""
  local result_file=$LOG_DIR/$name.result
  local status=0

  rm -f "$result_file"

  while :; do
    local attempt_work_dir=$WORK_ROOT/$name/attempt-$attempt
    local attempt_share_dir=$attempt_work_dir/beebs-build
    local log_file=$LOG_DIR/$name.attempt-$attempt.log
    local qemu_log_file=$attempt_work_dir/capstone-runtime-qemu-$name.log
    mkdir -p "$attempt_share_dir"

    if [[ $attempt -eq 1 ]]; then
      echo "run-all-beebs.sh: running $name"
    else
      echo "run-all-beebs.sh: retrying $name after QEMU infra flake (attempt $attempt/$max_attempts)"
    fi

    set +e
    if [[ "$VERBOSE" == 1 ]]; then
      TMP_ROOT="$attempt_work_dir" \
      OUT_DIR="$attempt_share_dir" \
      SHARE_DIR="$attempt_share_dir" \
      LOG_FILE="$qemu_log_file" \
      BEEBS_SRC_DIR="$SHARED_BEEBS_SRC_DIR" \
      BEEBS_FETCH_READONLY=1 \
      CAPSTONE_QEMU_LOGIN_TIMEOUT="$LOGIN_TIMEOUT" \
        bash "$runner" 2>&1 | tee "$log_file"
      status=${PIPESTATUS[0]}
    else
      if TMP_ROOT="$attempt_work_dir" \
         OUT_DIR="$attempt_share_dir" \
         SHARE_DIR="$attempt_share_dir" \
         LOG_FILE="$qemu_log_file" \
         BEEBS_SRC_DIR="$SHARED_BEEBS_SRC_DIR" \
         BEEBS_FETCH_READONLY=1 \
         CAPSTONE_QEMU_LOGIN_TIMEOUT="$LOGIN_TIMEOUT" \
           bash "$runner" >"$log_file" 2>&1; then
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
       grep -q "__CAPSTONE_INFRA_FLAKE__" "$log_file"; then
      attempt=$((attempt + 1))
      continue
    fi

    echo "run-all-beebs.sh: FAIL $name (exit=$status, log=$log_file)" >&2
    echo "run-all-beebs.sh: recent log tail:" >&2
    tail -80 "$log_file" >&2 || true
    printf 'status=FAIL\nattempt=%s\nlog=%s\nexit=%s\n' "$attempt" "$log_file" "$status" >"$result_file"
    exit "$status"
  done

  marker=$(grep -Eo '__BEEBS_[A-Z0-9_]+_PASSED__' "$log_file" | tail -1 || true)
  local retry_suffix=""
  if [[ $attempt -gt 1 ]]; then
    retry_suffix="_WITH_INFRA_RETRY"
  fi

  if [[ -n "$marker" ]]; then
    echo "run-all-beebs.sh: PASS$retry_suffix $name ($marker, log=$log_file)"
  else
    echo "run-all-beebs.sh: PASS$retry_suffix $name (log=$log_file)"
  fi
  printf 'status=PASS\nattempt=%s\nlog=%s\nmarker=%s\n' "$attempt" "$log_file" "$marker" >"$result_file"
}

if [[ "$JOBS" -eq 1 ]]; then
  for runner in "${BEEBS_RUNNERS[@]}"; do
    run_one "$runner"
  done
else
  echo "run-all-beebs.sh: running with RUN_ALL_BEEBS_JOBS=$JOBS"
  next=0
  active=0
  failure_seen=0

  while [[ $next -lt ${#BEEBS_RUNNERS[@]} || $active -gt 0 ]]; do
    while [[ $failure_seen -eq 0 && $next -lt ${#BEEBS_RUNNERS[@]} && $active -lt $JOBS ]]; do
      run_one "${BEEBS_RUNNERS[$next]}" &
      next=$((next + 1))
      active=$((active + 1))
    done

    if [[ $active -gt 0 ]]; then
      if wait -n; then
        :
      else
        failure_seen=1
      fi
      active=$((active - 1))
    elif [[ $failure_seen -ne 0 ]]; then
      break
    fi
  done

  if [[ $failure_seen -ne 0 ]]; then
    echo "run-all-beebs.sh: failed; no new jobs were scheduled after the first non-recovered failure" >&2
    exit 1
  fi
fi

passed=0
retried=0
for runner in "${BEEBS_RUNNERS[@]}"; do
  name=$(basename "$runner" .sh)
  result_file=$LOG_DIR/$name.result
  if [[ ! -f "$result_file" ]]; then
    echo "run-all-beebs.sh: missing result file for $name" >&2
    exit 1
  fi
  if grep -q '^status=PASS$' "$result_file"; then
    passed=$((passed + 1))
    attempt=$(sed -n 's/^attempt=//p' "$result_file")
    if [[ ${attempt:-1} -gt 1 ]]; then
      retried=$((retried + 1))
    fi
  else
    echo "run-all-beebs.sh: $name did not pass; result=$result_file" >&2
    exit 1
  fi
done

echo "run-all-beebs.sh: all $passed BEEBS runners completed (infra-retried=$retried, logs=$LOG_DIR)"
