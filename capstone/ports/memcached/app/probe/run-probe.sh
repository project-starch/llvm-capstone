#!/usr/bin/env bash
# M0 threads probe: build the domain probe and the native guest client, boot one capstone-vm, run
# the probe in the background, drive it with the client, end it with SIGTERM, report.
#   CAPSTONE_SDK=<sdk dir with capstone-cc>  CAPSTONE_VM_UP_ARGS="--qemu ... --firmware ... --module ...
#     --launcher ... --job-helper ... --kernel ... --rootfs ... --ssh-server ..."  run-probe.sh <out-dir>
# Predictions P1-P4: docs/plans/2026-10-01-memcached-full-app-port.md.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:?out dir}; mkdir -p "$OUT/share"
: "${CAPSTONE_SDK:?}" "${CAPSTONE_VM_UP_ARGS:?}"
XCC=${GUEST_CC:-${CAPSTONE_BUILDROOT_DIR:?}/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
"$CAPSTONE_SDK/capstone-cc" -O1 -o "$OUT/share/mc-threads-probe.dom" "$HERE/mc-threads-probe.c" -lpthread \
  || { echo "probe build FAILED"; exit 2; }
"$XCC" -O1 -o "$OUT/share/mc-probe-client" "$HERE/mc-probe-client.c" || { echo "client build FAILED"; exit 2; }
sha256sum "$OUT/share/mc-threads-probe.dom" "$OUT/share/mc-probe-client" | sed "s#$OUT/share/##"
VM=$OUT/vm
trap 'capstone-vm --state "$VM" down > /dev/null 2>&1' EXIT
for attempt in $(seq 1 2400); do
  rm -rf "$VM"
  # shellcheck disable=SC2086
  capstone-vm --state "$VM" up $CAPSTONE_VM_UP_ARGS --share "$OUT/share" --boot-timeout 600 > "$OUT/up.log" 2>&1 && break
  grep -q "Another Capstone VM owns" "$OUT/up.log" || { tail -3 "$OUT/up.log"; exit 3; }
  sleep 3
done
capstone-vm --state "$VM" exec sh -c '
  CAPSTONE_DELEGATE_STATS=1 /usr/bin/capstone-exec /mnt/host/mc-threads-probe.dom > /tmp/probe.out 2>&1 &
  P=$!
  /mnt/host/mc-probe-client; echo "client rc=$?"
  sleep 1
  t0=$(cut -d" " -f1 /proc/uptime); kill -TERM $P; wait $P; rc=$?; t1=$(cut -d" " -f1 /proc/uptime)
  echo "probe rc=$rc stop_seconds=$(echo "$t0 $t1" | awk "{printf \"%.2f\", \$2-\$1}")"
  echo "--- probe output"; cat /tmp/probe.out' > "$OUT/run.txt" 2>&1
echo "exec rc=$?"; cat "$OUT/run.txt"
