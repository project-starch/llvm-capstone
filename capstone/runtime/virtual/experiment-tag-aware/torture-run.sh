#!/bin/bash
# torture-run.sh TAG QEMU IMAGE MODULE LAUNCHER FORK_SECS ROUNDS [MODE]
# Boots one guest, runs the domain torture test while the host loop compacts guest memory and
# samples the domain's physical frames; result in ~/lanes/laneM/tt/TAG/.
set -uo pipefail
BIN=${BIN:-torture}
TAG=$1 QEMU=$2 IMAGE=$3 MODULE=$4 LAUNCHER=$5 FORK=$6 ROUNDS=$7 MODE=${8:-run}
L=~/lanes/laneM; D=~/prof-dev; PL=$D/platform-rf3; OUT=$L/tt/$TAG
[ -e "$OUT" ] && { echo "exists $OUT"; exit 2; }
mkdir -p "$OUT"
vm() { env CAPSTONE_QEMU_LOCK=$OUT/qemu.lock PYTHONPATH=$D/host python3 -m capstone_vm --state "$OUT/vm" "$@"; }
mkdir -m 700 -p ~/.capstone-spec-vm/lm-$TAG; ln -sfn ~/.capstone-spec-vm/lm-$TAG "$OUT/vm"
sha256sum "$QEMU" "$IMAGE" "$MODULE" "$LAUNCHER" $L/torture/$BIN $L/torture/pfns > "$OUT/inputs.sha256"
vm up --profile virtual --exact-bounds --memory 2G --share-model mapped-xattr \
  --cpu-prop x-capstone-test-csrs=true \
  --qemu "$QEMU" --kernel "$IMAGE" --firmware $PL/images/fw_jump.elf \
  --rootfs $PL/images/rootfs.ext2 --share $L \
  --launcher "$LAUNCHER" --job-helper $PL/adapter/capstone-job \
  --module "$MODULE" --ssh-server $PL/ssh-server/dropbearmulti > "$OUT/vm-up.log" 2>&1 || { echo "vm up failed"; tail -n 20 "$OUT/vm-up.log"; exit 3; }
trap 'vm down > /dev/null 2>&1 || true' EXIT
vm exec -- sh -c 'cp /mnt/host/torture/pfns /tmp/pfns && chmod 755 /tmp/pfns; killall syslogd klogd 2>/dev/null; uname -a; cat /proc/cmdline; echo swaps:; cat /proc/swaps; for p in /sys/module/captag/parameters/* /sys/module/capstone_vm/parameters/floor /sys/module/capstone_vm/parameters/unpinned; do [ -f $p ] && echo "$p=$(cat $p)"; done; grep -E "pgmigrate_success|compact_migrate_scanned|thp_fault_alloc" /proc/vmstat' > "$OUT/guest-before.txt" 2>&1
ENVS=(); [ "$FORK" != 0 ] && ENVS=(-e CAPSTONE_VM_TORTURE_FORK=$FORK); [ -n "${HUGE:-}" ] && ENVS+=(-e CAPSTONE_VM_HUGE=1)
if [ -n "${MODPARAMS:-}" ]; then vm exec -- sh -c "rmmod capstone_vm && insmod /mnt/control/module $MODPARAMS && cat /sys/module/capstone_vm/parameters/thp /sys/module/capstone_vm/parameters/unpinned" > "$OUT/modparams.txt" 2>&1; fi
if [ "$MODE" = stale ]; then ARGS=(stale "$ROUNDS"); else ARGS=("$ROUNDS" 1500); fi
  vm run --cwd /mnt/host/torture "${ENVS[@]}" --result "$OUT/result.json" -- ./$BIN "${ARGS[@]}" > "$OUT/run.out" 2> "$OUT/run.err" &
  RUN=$!
  while kill -0 $RUN 2>/dev/null; do
    if grep -q READY "$OUT/run.out" 2>/dev/null; then
      line=$(grep -m1 READY "$OUT/run.out")
      pid=$(echo "$line" | sed 's/.*pid=\([0-9]*\).*/\1/'); heap=$(echo "$line" | sed 's/.* heap=\(0x[0-9a-f]*\).*/\1/')
      hb=$(echo "$line" | sed 's/.*heap_bytes=\([0-9]*\).*/\1/'); bss=$(echo "$line" | sed 's/.* bss=\(0x[0-9a-f]*\).*/\1/')
      bb=$(echo "$line" | sed 's/.*bss_bytes=\([0-9]*\).*/\1/')
      { date +%T; vm exec -- sh -c "echo 1 > /proc/sys/vm/compact_memory; echo heap \$(/tmp/pfns $pid $heap $hb /tmp/st-heap); echo bss \$(/tmp/pfns $pid $bss $bb /tmp/st-bss); grep -E 'pgmigrate_success|compact_migrate_scanned' /proc/vmstat | tr '\n' ' '; echo"; } >> "$OUT/compact.log" 2>&1
    fi
    sleep 1
  done
  wait $RUN; echo "rc=$?" > "$OUT/rc.txt"
vm exec -- sh -c 'for p in /sys/module/captag/parameters/* /sys/module/capstone_vm/parameters/unpinned_faults; do [ -f $p ] && echo "$p=$(cat $p)"; done; grep -E "pgmigrate_success|pgmigrate_fail|compact_migrate_scanned|thp_fault_alloc" /proc/vmstat; dmesg | grep -i -E "capstone|captag|BUG|Oops" | tail -n 30' > "$OUT/guest-after.txt" 2>&1
vm down > /dev/null 2>&1; trap - EXIT
cat "$OUT/rc.txt"; tail -n 3 "$OUT/run.out"; awk '/^heap|^bss/' "$OUT/compact.log" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i ~ /^moved=/){split($i,a,"=");m[$1]+=a[2]}} END{for(k in m) print k, "moved_total", m[k]}'
