#!/usr/bin/env bash
# Boot Capstone QEMU once and run the musl-hello domain.
#
# Same two rules as yield-probe/run-yield-probe.sh, both learned the expensive
# way: the boot control shares ONE guest command with the program (every
# --success-marker is checked against every --guest-command), and the timeout
# multiplier defaults to 8 because boot-to-login on this host straddles the
# 480 s that a multiplier of 4 allows.
#
# Both writes are required markers. The first proves only that the outward path
# works; the second can only be reached by RETURNING from the first write, so it
# is the one that shows a syscall completed rather than merely fired.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/musl-capstone-hello}
SHARE_DIR=${SHARE_DIR:-$OUT_DIR/share}
LOG_FILE=${LOG_FILE:-$CAPSTONE_TMP_ROOT/capstone-musl-hello.log}

mkdir -p "$OUT_DIR" "$SHARE_DIR"
rm -f "$SHARE_DIR"/musl_hello*.dom "$SHARE_DIR/musl_hello.user"

OUT_DIR="$OUT_DIR" OUT_DOM="$SHARE_DIR/musl_hello.dom" \
  OUT_HOST="$SHARE_DIR/musl_hello.user" \
  bash "$SCRIPT_DIR/build-musl-hello.sh"

# QUIET THE KERNEL CONSOLE. Measured 2026-08-14: this image prints "remote fence
# extension is not available in SBI v1.0" 842 times during boot -- 61% of all
# console traffic -- and under TCG with an emulated 8250 the console is the
# bottleneck. Booting with loglevel=1 (a second -append overrides the driver's
# first one) cut console traffic from 71125 to 2708 bytes and time-to-login from
# 8-16 minutes, frequently timing out at `Running sysctl:` right after klogd
# starts draining the ring, to under 2.5 minutes. This is not a workaround for a
# flake; the flood WAS the flake.
# RETRY ON INFRA FLAKE. run-domain-smoke.py exits 75 with
# `__CAPSTONE_INFRA_FLAKE__ phase=boot-login` when the guest never reaches the
# login prompt. Characterised 2026-08-14: the vCPU thread spins at 99.9%, so it
# is a guest-side busy loop, and it is neither our domain (not loaded yet) nor
# the qemu arguments (the identical command line reaches login when its output
# goes to a file instead of a pexpect pty). Most likely a getty/respawn race in
# the buildroot image. Not ours to fix, but cheap to survive: a boot is ~3
# minutes now, so retry rather than lose the run.
run_smoke() {
  local attempt
  for attempt in 1 2 3; do
    if "$@"; then
      return 0
    fi
    local status=$?
    if [[ $status -ne 75 ]]; then
      return $status
    fi
    echo "run: infra flake on attempt $attempt (exit 75), retrying" >&2
  done
  echo "run: 3 consecutive infra flakes, giving up" >&2
  return 75
}

# DETECT THE FLAKE FAST. A healthy boot reaches the login prompt in ~3 minutes
# with the quiet console; the driver's default is 120 * multiplier = 960 s, so a
# stalled boot used to burn sixteen minutes before anyone noticed. 300 s is
# comfortably above a healthy boot and turns a flake into a five-minute loss.
# Retries then bound the worst case at ~15 minutes instead of ~48.
CAPSTONE_QEMU_LOGIN_TIMEOUT=${CAPSTONE_QEMU_LOGIN_TIMEOUT:-300} \
run_smoke python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --qemu-extra-arg=-append --qemu-extra-arg="root=/dev/vda ro loglevel=1" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier "${TIMEOUT_MULTIPLIER:-8}" \
  --guest-command \
    'echo __CAPSTONE_QEMU_BOOT_CONTROL_OK__; cp /mnt/host/musl_hello.user /tmp/h && chmod 0755 /tmp/h && /tmp/h /mnt/host/musl_hello.dom' \
  --success-marker '__CAPSTONE_QEMU_BOOT_CONTROL_OK__' \
  --success-marker 'S1: hostcall.c direct write ok' \
  --success-marker 'S2: musl write() RETURNED' \
  --success-marker 'musl-hello: write #1 through musl in a domain' \
  --success-marker 'musl-hello: write #2, so write() RETURNED' \
  --success-marker '__CAPSTONE_HOSTCALL_HOST_DONE__ status=0'

echo "run-musl-hello.sh completed. Full serial log: $LOG_FILE"
