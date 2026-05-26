#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNTIME_QEMU_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$RUNTIME_QEMU_DIR/../capstone-test-env.sh"

SHARE_DIR=${SHARE_DIR:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-typed-load-repro-share}
LOG_DIR=${LOG_DIR:-$CAPSTONE_TMP_ROOT/static-cap-typed-load-repro-logs}

mkdir -p "$SHARE_DIR" "$LOG_DIR"
rm -f "$SHARE_DIR"/*.dom "$SHARE_DIR"/capstone-test.user

bash "$RUNTIME_QEMU_DIR/build-capstone-test-user.sh" \
  "$SHARE_DIR/capstone-test.user"

bash "$SCRIPT_DIR/build.sh" "$SHARE_DIR"

run_expect_success() {
  local domain_name=$1
  local expected_value=$2
  local log_file="$LOG_DIR/${domain_name}.log"

  python3 "$RUNTIME_QEMU_DIR/run-domain-smoke.py" \
    --share-dir "$SHARE_DIR" \
    --log-file "$log_file" \
    --timeout-multiplier 2 \
    --guest-command "/mnt/host/capstone-test.user /mnt/host/${domain_name}.dom" \
    --success-marker 'Created domain ID = 0' \
    --success-marker "Called dom (1-th time) retval = ${expected_value}"
}

run_expect_failure() {
  local domain_name=$1
  local expected_marker=$2
  local log_file="$LOG_DIR/${domain_name}.log"
  local wrapper_file="$LOG_DIR/${domain_name}.wrapper.log"
  local status

  set +e
  python3 "$RUNTIME_QEMU_DIR/run-domain-smoke.py" \
    --share-dir "$SHARE_DIR" \
    --log-file "$log_file" \
    --timeout-multiplier 2 \
    --guest-command "/mnt/host/capstone-test.user /mnt/host/${domain_name}.dom" \
    --success-marker 'Created domain ID = 0' \
    --success-marker 'Called dom (1-th time) retval =' \
    > "$wrapper_file" 2>&1
  status=$?
  set -e

  if [ "$status" -eq 0 ]; then
    echo "${domain_name}: unexpectedly succeeded" >&2
    exit 1
  fi

  if ! grep -qF "$expected_marker" "$log_file" "$wrapper_file"; then
    echo "${domain_name}: failed, but not with the expected marker" >&2
    echo "  expected marker: $expected_marker" >&2
    echo "  log: $log_file" >&2
    echo "  wrapper: $wrapper_file" >&2
    exit 1
  fi
}

run_expect_failure fail_fn_struct_load '[CAPSTONE] cs.cjalr requires capability in rs1'
run_expect_failure fail_str_struct_load '[CAPSTONE] Cap mem access requires capability'
run_expect_success fix_fn_runtime_materialize 305419896
run_expect_success fix_str_runtime_materialize 111
run_expect_success descriptor_fn_runtime_materialize 305419896
run_expect_success descriptor_str_runtime_materialize 111
run_expect_success autogen_fn_runtime_materialize 305419896
run_expect_success autogen_str_runtime_materialize 111
run_expect_success consume_emitted_gct_string_domain 111

echo 'static-cap-typed-load-repro: reproduced both minimal failing capability-typed loads and verified manual, descriptor-driven, LLVM-IR-generated, and emitted-.gct-consumer runtime-materialization fixes'
echo '__STATIC_CAP_TYPED_LOAD_REPRO_OK__'
printf 'Logs: %s\n' "$LOG_DIR"



