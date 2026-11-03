#!/bin/bash
# torture-run.sh TAG QEMU IMAGE MODULE LAUNCHER FORK_SECS ROUNDS [MODE]
# Boots one guest, runs the domain torture test while the host loop compacts guest memory and
# samples physical frames; all writable state stays beneath LANE_ROOT.
set -uo pipefail
BIN=${BIN:-torture}
TAG=$1 QEMU=$2 IMAGE=$3 MODULE=$4 LAUNCHER=$5 FORK=$6 ROUNDS=$7 MODE=${8:-run}
L=${LANE_ROOT:?Set a private lane root}; PL=${PLATFORM:?Set a private platform}; OUT=$L/tt/$TAG
[ -e "$OUT" ] && { echo "exists $OUT"; exit 2; }
mkdir -p "$OUT"
vm() { env CAPSTONE_QEMU_LOCK=$OUT/qemu.lock PYTHONPATH=${VM_HOST:-$L/vmhost} python3 -m capstone_vm --state "$OUT/vm" "$@"; }
mkdir -m 700 -p "$L/vm/t-$TAG"; ln -s "$L/vm/t-$TAG" "$OUT/vm"
sha256sum "$QEMU" "$IMAGE" "$MODULE" "$LAUNCHER" $L/torture/$BIN $L/torture/pfns > "$OUT/inputs.sha256"
vm up --profile virtual --exact-bounds --memory 2G --share-model mapped-xattr \
  --cpu-prop x-capstone-test-csrs=${TEST_CSRS:-true} --cpu-prop x-capstone-raw-moves=true --cpu-prop x-capstone-supervisor-slots=2 \
  --qemu "$QEMU" --kernel "$IMAGE" --firmware $PL/images/fw_jump.elf \
  --rootfs $PL/images/rootfs.ext2 --share $L \
  --launcher "$LAUNCHER" --job-helper $PL/adapter/capstone-job \
  --module "$MODULE" --ssh-server $PL/ssh-server/dropbearmulti > "$OUT/vm-up.log" 2>&1 || { echo "vm up failed"; tail -n 20 "$OUT/vm-up.log"; exit 3; }
trap 'vm down > /dev/null 2>&1 || true' EXIT
vm exec -- sh -c 'cp /mnt/host/torture/pfns /tmp/pfns && chmod 755 /tmp/pfns; cp /mnt/host/torture/hog /tmp/hog && chmod 755 /tmp/hog; killall syslogd klogd 2>/dev/null; uname -a; cat /proc/cmdline; echo swaps:; cat /proc/swaps; for p in /sys/module/captag/parameters/* /sys/module/capstone_vm/parameters/floor /sys/module/capstone_vm/parameters/unpinned; do [ -f $p ] && echo "$p=$(cat $p)"; done; grep -E "pgmigrate_success|compact_migrate_scanned|thp_fault_alloc" /proc/vmstat' > "$OUT/guest-before.txt" 2>&1
ENVS=(); [ "$FORK" != 0 ] && ENVS=(-e CAPSTONE_VM_TORTURE_FORK=$FORK)
if [ -n "${MODPARAMS:-}" ]; then vm exec -- sh -c "rmmod capstone_vm && insmod /mnt/control/module $MODPARAMS && cat /sys/module/capstone_vm/parameters/unpinned" > "$OUT/modparams.txt" 2>&1; fi
if [ "$MODE" = stale ]; then ARGS=(stale "$ROUNDS"); else ARGS=("$ROUNDS" 1500); fi
if [ -n "${HOG:-}" ]; then
  vm exec -- sh -c "nohup /tmp/hog $HOG > /tmp/hog.out 2>&1 < /dev/null & for i in \$(seq 1 120); do grep -q READY /tmp/hog.out && break; sleep 1; done; cat /tmp/hog.out; free -m" > "$OUT/hog.txt" 2>&1
fi
HOGKILL=${HOG:+1}
  vm run --cwd /mnt/host/torture "${ENVS[@]}" --result "$OUT/result.json" -- ./$BIN "${ARGS[@]}" > "$OUT/run.out" 2> "$OUT/run.err" &
  RUN=$!
  RUN2=
  if [ -n "${TWOPROC:-}" ]; then
    vm run --cwd /mnt/host/torture "${ENVS[@]}" --result "$OUT/result2.json" -- ./$BIN "${ARGS[@]}" > "$OUT/run2.out" 2> "$OUT/run2.err" &
    RUN2=$!
  fi
  while kill -0 $RUN 2>/dev/null; do
    if grep -q READY "$OUT/run.out" 2>/dev/null; then
      line=$(grep -m1 READY "$OUT/run.out")
      pid=$(echo "$line" | sed 's/.*pid=\([0-9]*\).*/\1/'); heap=$(echo "$line" | sed 's/.* heap=\(0x[0-9a-f]*\).*/\1/')
      hb=$(echo "$line" | sed 's/.*heap_bytes=\([0-9]*\).*/\1/'); bss=$(echo "$line" | sed 's/.* bss=\(0x[0-9a-f]*\).*/\1/')
      bb=$(echo "$line" | sed 's/.*bss_bytes=\([0-9]*\).*/\1/')
      if [ -n "$HOGKILL" ]; then vm exec -- sh -c "killall hog; sleep 1; free -m" >> "$OUT/hog.txt" 2>&1; HOGKILL=; fi
      { date +%T; vm exec -- sh -c "echo heap \$(/tmp/pfns $pid $heap $hb /tmp/st-heap); echo bss \$(/tmp/pfns $pid $bss $bb /tmp/st-bss); echo 1 > /proc/sys/vm/compact_memory; grep -E 'pgmigrate_success|compact_migrate_scanned' /proc/vmstat | tr '\n' ' '; echo"; } >> "$OUT/compact.log" 2>&1
    fi
    sleep 1
  done
  if [ -n "$RUN2" ]; then wait "$RUN2"; echo "rc2=$?" > "$OUT/rc2.txt"; fi
  wait $RUN; echo "rc=$?" > "$OUT/rc.txt"
vm exec -- sh -c 'for p in /sys/module/captag/parameters/* /sys/module/capstone_vm/parameters/unpinned_faults; do [ -f $p ] && echo "$p=$(cat $p)"; done; grep -E "pgmigrate_success|pgmigrate_fail|compact_migrate_scanned|thp_fault_alloc" /proc/vmstat; dmesg | grep -i -E "capstone|captag|BUG|Oops" | tail -n 30' > "$OUT/guest-after.txt" 2>&1
vm down > /dev/null 2>&1; trap - EXIT
cat "$OUT/rc.txt"; tail -n 3 "$OUT/run.out"; awk '/^heap|^bss/' "$OUT/compact.log" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i ~ /^moved=/){split($i,a,"=");m[$1]+=a[2]}} END{for(k in m) print k, "moved_total", m[k]}'
