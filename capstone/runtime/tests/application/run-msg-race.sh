#!/usr/bin/env bash
# sendmsg and recvmsg from four contexts at once, under two launchers in one boot: the arms differ
# only in the capstone-exec binary that runs pthread-probe.dom (pthread-probe.c, msg_concurrent).
#   run-msg-race.sh <out-dir> <pthread-probe.dom> <stock capstone-exec> <fixed capstone-exec>
# Env: CAPSTONE_VM_UP_ARGS (capstone-vm up platform arguments), capstone-vm on PATH; MSG_RACE_ARMS
# (launcher:mode:rounds ...) replaces the default arms. Arms alternate
# stock and fixed, so boot history (what ran earlier in the boot) is not what separates them.
set -uo pipefail
OUT=${1:?out dir}; DOM=${2:?probe}; STOCK=${3:?stock launcher}; FIXED=${4:?fixed launcher}
: "${CAPSTONE_VM_UP_ARGS:?}"
mkdir -p "$OUT/share"
cp "$DOM" "$OUT/share/pthread-probe.dom"; cp "$STOCK" "$OUT/share/exec-stock"; cp "$FIXED" "$OUT/share/exec-fixed"
( cd "$OUT/share" && sha256sum pthread-probe.dom exec-stock exec-fixed ) | tee "$OUT/inputs.txt"
VM=$OUT/vm
trap 'capstone-vm --state "$VM" down > /dev/null 2>&1' EXIT
for attempt in $(seq 1 2400); do
  rm -rf "$VM"
  # shellcheck disable=SC2086
  capstone-vm --state "$VM" up $CAPSTONE_VM_UP_ARGS --share "$OUT/share" --boot-timeout 600 > "$OUT/up.log" 2>&1 && break
  grep -q "Another Capstone VM owns" "$OUT/up.log" || { tail -3 "$OUT/up.log"; exit 3; }
  sleep 3
done
ARMS=${MSG_RACE_ARMS:-"stock:sendmsg:300 fixed:sendmsg:300 stock:recvmsg:300 fixed:recvmsg:300 stock:sendmsg:1000 fixed:sendmsg:1000 stock:recvmsg:1000 fixed:recvmsg:1000"}
capstone-vm --state "$VM" exec sh -c "
  for arm in $ARMS; do
    l=\${arm%%:*}; rest=\${arm#*:}; mode=\${rest%%:*}; rounds=\${rest#*:}
    t0=\$(cut -d' ' -f1 /proc/uptime)
    # a watchdog, not timeout(1): the guest's busybox has no timeout applet
    /mnt/host/exec-\$l /mnt/host/pthread-probe.dom \$mode-concurrent \$rounds > /tmp/arm.out 2>&1 & p=\$!
    ( sleep 900; kill -KILL \$p 2>/dev/null ) & w=\$!
    wait \$p; rc=\$?; kill \$w 2>/dev/null
    t1=\$(cut -d' ' -f1 /proc/uptime)
    echo \"ARM \$arm rc=\$rc seconds=\$(echo \"\$t0 \$t1\" | awk '{printf \"%.1f\", \$2-\$1}')\"
    sed 's/^/  /' /tmp/arm.out
  done" > "$OUT/run.txt" 2>&1
echo "exec rc=$?"; cat "$OUT/run.txt"
