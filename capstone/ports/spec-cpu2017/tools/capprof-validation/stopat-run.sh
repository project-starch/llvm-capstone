#!/bin/bash
# stopat-run.sh TAG: capprof's work-bounded windows (stop_vaddr/stop_count, start_*) against the
# same program's full run, plain and Capstone (tools/stop-at.py resolves stopat_f in each build;
# the Capstone image is placed by the launcher, so its site is matched with slide_align). One VM
# per configuration, because the plugin's arguments are fixed at boot and a stop ends the VM:
#   full        no stop: stopat-plain then stopat-cap (windows 0-1 plain, 2-3 Capstone)
#   plain-stop  stop at entry 500 of stopat_f              cap-stop    the same, Capstone
#   plain-start start at entry 100, stop at 500             cap-start   start at 400, stop at 500
#   plain-fault `stopat fault`: stop at entry 32 of stopat_g, whose first instruction faults and
#               is retried across a block boundary          cap-fault   the same, Capstone
#   plain-faultcount, cap-faultcount: stop at 1000, never reached: the footer's entry count is 64
#   plain-fault-old (only with OLD_PLUGIN=path, and without the anchor arguments that plugin does
#               not know): the negative control for the retry rule
# Then stopatcheck.py. Environment: PLATFORM (default ~/prof-dev/platform) with
# plugins/libcapprof.so, SDK (capstone-cc), RVCC (plain driver), NM (llvm-nm), SHARE (a directory
# below which TAG/ is made; it is the guest's /mnt/host).
set -euo pipefail
TAG=$1
D=~/prof-dev; PL=${PLATFORM:-$D/platform}; V=${SHARE:-$D/vt}; OUT=$V/$TAG
HERE=$(cd "$(dirname "$0")" && pwd -P)
NM=${NM:-$D/cap/build/bin/llvm-nm}
[ -e "$OUT" ] && { echo "exists $OUT"; exit 2; }
mkdir -p "$OUT"
cp "$HERE/stopat.c" "$OUT/"
"${RVCC:-$D/rv/rv-cc}" -O2 -o "$OUT/stopat-plain" "$OUT/stopat.c"
"${SDK:-$D/cap/sdk}/capstone-cc" -O2 -o "$OUT/stopat-cap" "$OUT/stopat.c"
for k in plain cap; do
  kind=$([ $k = cap ] && echo capstone || echo plain)
  python3 "$HERE/../stop-at.py" resolve --nm "$NM" --kind $kind --exe "$OUT/stopat-$k" --symbol stopat_f --count 500 > "$OUT/$k-stop.json"
  start=$([ $k = cap ] && echo 400 || echo 100)
  python3 "$HERE/../stop-at.py" resolve --nm "$NM" --kind $kind --exe "$OUT/stopat-$k" --symbol stopat_f --count 500 \
    --start-symbol stopat_f --start-count $start > "$OUT/$k-start.json"
  for n in fault:32 faultcount:1000; do
    python3 "$HERE/../stop-at.py" resolve --nm "$NM" --kind $kind --exe "$OUT/stopat-$k" --symbol stopat_g --count ${n#*:} > "$OUT/$k-${n%:*}.json"
  done
done
if [ -n "${OLD_PLUGIN:-}" ]; then
  python3 -c "import json,sys;d=json.load(open(sys.argv[1]));d['plugin_args']=[a for a in d['plugin_args'] if not a.startswith('anchor_')];print(json.dumps(d))" "$OUT/plain-fault.json" > "$OUT/plain-fault-old.json"
fi
boot() { # NAME PROGRAM... (PROGRAM = plain|cap), plugin arguments from NAME.json when it exists
  local name=$1; shift
  local o=$OUT/$name; mkdir -p "$o/capprof"
  local args=() plugin=$PL/plugins/libcapprof.so
  case $name in *-old) plugin=$OLD_PLUGIN ;; esac
  if [ -f "$OUT/$name.json" ]; then
    mapfile -t pa < <(python3 -c "import json,sys;print('\n'.join(json.load(open(sys.argv[1]))['plugin_args']))" "$OUT/$name.json")
    for a in "${pa[@]}"; do args+=(--plugin-arg "$a"); done
  fi
  echo "${args[*]:-no stop}" > "$o/plugin-args.txt"
  vm() { env CAPSTONE_QEMU_LOCK=$o/qemu.lock PYTHONPATH=$D/host python3 -m capstone_vm --state "$o/vm" "$@"; }
  mkdir -m 700 -p ~/.capstone-spec-vm/stopat-$TAG-$name; ln -sfn ~/.capstone-spec-vm/stopat-$TAG-$name "$o/vm"
  vm up --profile virtual --exact-bounds --memory 8G --share-model mapped-xattr --no-pcrel --icount 0 \
    --cpu-prop x-capstone-test-csrs=true \
    --plugin "$plugin" --plugin-arg out=$o/capprof "${args[@]}" \
    --qemu $PL/qemu-system-riscv64 --kernel $PL/images/Image --firmware $PL/images/fw_jump.elf \
    --rootfs $PL/images/rootfs.ext2 --share $V \
    --launcher $PL/adapter/capstone-vexec --job-helper $PL/adapter/capstone-job \
    --module $PL/adapter/module/capstone_vm.ko --ssh-server $PL/ssh-server/dropbearmulti > "$o/vm-up.log" 2>&1
  vm exec -- sh -c "echo 0 > /proc/sys/kernel/randomize_va_space; killall syslogd klogd 2>/dev/null; cat /proc/sys/kernel/randomize_va_space" > "$o/guest-setup.txt"
  for p in "$@"; do
    rc=0
    arg=; case $p in *f) arg=fault ;; esac
    if [ "${p%f}" = cap ]; then
      vm run --cwd /mnt/host/$TAG -e CAPSTONE_PROFILE_RECORD=/mnt/host/$TAG/$name/image-records.txt -- ./stopat-cap $arg > "$o/$p.out" 2> "$o/$p.err" || rc=$?
    else
      vm exec -- sh -c "cp /mnt/host/$TAG/stopat-plain /tmp/ && chmod 0755 /tmp/stopat-plain && /tmp/stopat-plain $arg" > "$o/$p.out" 2> "$o/$p.err" || rc=$?
    fi
    echo "$p rc=$rc" >> "$o/order.txt"
  done
  vm status > "$o/status-after.txt" 2>&1 || true   # "stopped" after a slice: capprof ended QEMU
  vm down > /dev/null 2>&1 || true
}
boot full plain cap
boot plain-stop plain
boot cap-stop cap
boot plain-start plain
boot cap-start cap
boot plain-fault plainf
boot cap-fault capf
boot plain-faultcount plainf
boot cap-faultcount capf
[ -z "${OLD_PLUGIN:-}" ] || boot plain-fault-old plainf
python3 "$HERE/stopatcheck.py" "$OUT" | tee "$OUT/stopatcheck.txt"
