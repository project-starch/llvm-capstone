#!/usr/bin/env bash
# Q-12 acceptance includes the real Linux register-save path, not just probes.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CAPSTONE_REPO_ROOT=$(cd -- "$script_dir/../../.." && pwd)
source "$CAPSTONE_REPO_ROOT/capstone/tests/capstone-test-env.sh"

image_dir=
compiler=
qemu=$CAPSTONE_QEMU_BINARY
guest_timeout=150
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image-dir) image_dir=$2; shift 2 ;;
        --cc) compiler=$2; shift 2 ;;
        --qemu) qemu=$2; shift 2 ;;
        --timeout) guest_timeout=$2; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ -z $image_dir || -z $compiler ]]; then
    echo "Usage: $0 --image-dir DIR --cc COMPILER [--qemu BINARY] [--timeout SECONDS]" >&2
    exit 2
fi
if [[ ${CAPSTONE_QEMU_LOCK_HELD:-0} != 1 ]]; then
    exec env CAPSTONE_QEMU_LOCK_HELD=1 flock -w "${CAPSTONE_QEMU_LOCK_WAIT:-3600}" \
        "$CAPSTONE_QEMU_LOCK" "$0" --image-dir "$image_dir" --cc "$compiler" \
        --qemu "$qemu" --timeout "$guest_timeout"
fi

mkdir -p "$CAPSTONE_TMP_ROOT/virtual-capstone-q12"
work_dir=$(mktemp -d "$CAPSTONE_TMP_ROOT/virtual-capstone-q12/run.XXXXXX")
qemu_tests=$CAPSTONE_REPO_ROOT/capstone/capstone-qemu/tests
VIRTUAL_CAPSTONE_QEMU_BINARY=$qemu "$qemu_tests/virtual-capstone-m1/run.sh" \
    > "$work_dir/m1.log" 2>&1
tail -n 1 "$work_dir/m1.log"
TRUSTED_LINUX_QEMU_BINARY=$qemu "$qemu_tests/trusted-linux-u-access/run.sh" \
    > "$work_dir/u-access.log" 2>&1
tail -n 1 "$work_dir/u-access.log"

common=(--image-dir "$image_dir" --cc "$compiler" --qemu "$qemu"
        --protected --timeout "$guest_timeout")
python3 "$script_dir/run.py" "${common[@]}" \
    --log "$work_dir/protected.log" --record "$work_dir/protected-result.json"
python3 "$script_dir/run.py" "${common[@]}" --control-strip-protected-tag \
    --log "$work_dir/strip.log" --record "$work_dir/strip-result.json"
printf 'PASS Q-12 QEMU and Linux acceptance (artifacts: %s)\n' "$work_dir"
