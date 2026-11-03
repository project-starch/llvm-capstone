#!/bin/bash
# stopat-run.sh TAG: capprof's work-bounded windows (stop/start sites) against the same program's
# full run, plain and Capstone (tools/stop-at.py resolves the sites in each build; the Capstone
# image is placed by the launcher, so its sites are matched at the base its anchor arms). One VM per
# configuration, because the plugin's arguments are fixed at boot and a stop ends the VM:
#   full         no stop: stopat-plain then stopat-cap (windows 0-1 plain, 2-3 Capstone)
#   plain-stop   stop at entry 500 of stopat_f             cap-stop    the same, Capstone
#   plain-start  start at entry 100, stop at 500           cap-start   start at 400, stop at 500
#   plain-k1     stop at entry 1 of stopat_f: the site has executed once, its successor not
#   plain-fault  `stopat fault`: stop at entry 32 of stopat_g, whose first instruction faults and
#                is retried across a block boundary        cap-fault   the same, Capstone
#   plain-faultcount, cap-faultcount: stop at 1000, never reached: the footer's entry count is 64
#   plain-handler `stopat handler`: stop at entry 5 of stopat_h, whose SIGSEGV handler reaches the
#                site's successor by another path before the retry; plain-handlercount: 1000
#   cap-inter    two domain images at different bases (stopat-cap, stopat-cap2) started together in
#                ONE window opened by profrun, stop at entry 500 of stopat_f: both bases are armed
#                and their entries add up. They do not interleave: the module admits one live
#                domain (supervisor_slots=1, the processor's one saved-context slot), and with two
#                slots this platform halts the guest (cause 24 in the firmware), so the second
#                domain enters only when the first has ended
#   refusals     the resolver and the plugin refuse an anchor that is the site
# Negative controls, each only when its plugin is given (the arguments that plugin does not know
# are dropped): OLD_PLUGIN (671eacb23e, entries = arrivals at a block end): plain-fault-old;
# OLD2_PLUGIN (136068aae7, completion flag cleared by any successor arrival): plain-handler-old2.
# Then stopatcheck.py and stopat-kit-test.py. Environment: PLATFORM (default ~/prof-dev/platform)
# with plugins/libcapprof.so and profrun, SDK (capstone-cc), RVCC (plain driver), NM (llvm-nm),
# SHARE (a directory below which TAG/ is made; it is the guest's /mnt/host).
set -euo pipefail
TAG=$1
D=~/prof-dev; PL=${PLATFORM:-$D/platform}; V=${SHARE:-$D/vt}; OUT=$V/$TAG
HERE=$(cd "$(dirname "$0")" && pwd -P)
NM=${NM:-$D/cap/build/bin/llvm-nm}
[ -e "$OUT" ] && { echo "exists $OUT"; exit 2; }
mkdir -p "$OUT"
cp "$HERE/stopat.c" "$OUT/"; cp "$PL/profrun" "$OUT/profrun"
"${RVCC:-$D/rv/rv-cc}" -O2 -DSTOPAT_PLAIN -o "$OUT/stopat-plain" "$OUT/stopat.c"
"${SDK:-$D/cap/sdk}/capstone-cc" -O2 -o "$OUT/stopat-cap" "$OUT/stopat.c"
"${SDK:-$D/cap/sdk}/capstone-cc" -O2 -DSTOPAT_BIG -o "$OUT/stopat-cap2" "$OUT/stopat.c"
res() { # NAME KIND EXE ARGS...
  local name=$1 kind=$2 exe=$3; shift 3
  python3 "$HERE/../stop-at.py" resolve --nm "$NM" --kind $kind --exe "$OUT/$exe" "$@" > "$OUT/$name.json"
}
for k in plain cap; do
  kind=$([ $k = cap ] && echo capstone || echo plain)
  res $k-stop $kind stopat-$k --symbol stopat_f --count 500
  res $k-start $kind stopat-$k --symbol stopat_f --count 500 --start-symbol stopat_f --start-count $([ $k = cap ] && echo 400 || echo 100)
  res $k-fault $kind stopat-$k --symbol stopat_g --count 32
  res $k-faultcount $kind stopat-$k --symbol stopat_g --count 1000
done
res plain-k1 plain stopat-plain --symbol stopat_f --count 1
res plain-handler plain stopat-plain --symbol stopat_h --count 5
res plain-handlercount plain stopat-plain --symbol stopat_h --count 1000
res cap-inter capstone stopat-cap --symbol stopat_f --count 500
res cap2-inter capstone stopat-cap2 --symbol stopat_f --count 500   # its offsets must equal stopat-cap's
strip() { # SRC DST PREFIX...: a plugin's arguments without the ones an older plugin does not know
  local src=$1 dst=$2; shift 2
  python3 -c "import json,sys;d=json.load(open(sys.argv[1]));d['plugin_args']=[a for a in d['plugin_args'] if not a.startswith(tuple(sys.argv[2:]))];print(json.dumps(d))" "$OUT/$src.json" "$@" > "$OUT/$dst.json"
}
[ -z "${OLD_PLUGIN:-}" ] || strip plain-fault plain-fault-old anchor_ stop_next_bytes text_
[ -z "${OLD2_PLUGIN:-}" ] || strip plain-handler plain-handler-old2 stop_next_bytes text_

# Refusals: the resolver refuses the image's entry as a site, and the plugin an anchor at a site.
{
  for k in plain:_start cap:domain_main; do
    rc=0; python3 "$HERE/../stop-at.py" resolve --nm "$NM" --kind $([ ${k%:*} = cap ] && echo capstone || echo plain) \
      --exe "$OUT/stopat-${k%:*}" --symbol ${k#*:} --count 1 > /dev/null 2> "$OUT/refuse-${k%:*}.err" || rc=$?
    echo "resolver ${k#*:} rc=$rc $(tail -1 "$OUT/refuse-${k%:*}.err")"
  done
  mkdir -p "$OUT/refuse-plugin"
  rc=0; timeout 30 "$PL/qemu-system-riscv64" -M virt -display none -monitor none -serial none -S \
    -plugin "$PL/plugins/libcapprof.so,out=$OUT/refuse-plugin,stop_vaddr=10000,stop_count=1,stop_bytes=13,stop_next_bytes=13,anchor_vaddr=10000,anchor_bytes=13,text_vaddr=0,text_size=100000" \
    > "$OUT/refuse-plugin.err" 2>&1 || rc=$?
  echo "plugin anchor=site rc=$rc $(grep '^capprof:' "$OUT/refuse-plugin.err" | tail -1)"
} > "$OUT/refusals.txt"

boot() { # NAME PROGRAM... (plain|cap|plainf|capf|plainh|inter), plugin arguments from NAME.json
  local name=$1; shift
  local o=$OUT/$name; mkdir -p "$o/capprof"
  local args=() plugin=$PL/plugins/libcapprof.so
  case $name in *-old) plugin=$OLD_PLUGIN ;; *-old2) plugin=$OLD2_PLUGIN ;; esac
  if [ -f "$OUT/$name.json" ]; then
    mapfile -t pa < <(python3 -c "import json,sys;print('\n'.join(json.load(open(sys.argv[1]))['plugin_args']))" "$OUT/$name.json")
    for a in "${pa[@]}"; do args+=(--plugin-arg "$a"); done
  fi
  echo "$plugin ${args[*]:-no stop}" > "$o/plugin-args.txt"
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
    case $p in
      cap|capfault) vm run --cwd /mnt/host/$TAG -e CAPSTONE_PROFILE_RECORD=/mnt/host/$TAG/$name/image-records.txt -- ./stopat-cap ${p#cap} > "$o/$p.out" 2> "$o/$p.err" || rc=$? ;;
      inter) vm exec -- sh -c "for f in stopat-cap stopat-cap2 profrun; do cp /mnt/host/$TAG/\$f /tmp/ && chmod 0755 /tmp/\$f || exit 125; done; /tmp/profrun sh -c '/tmp/stopat-cap inter 300 & /tmp/stopat-cap2 inter 300 & wait'" > "$o/$p.out" 2> "$o/$p.err" || rc=$? ;;
      *) vm exec -- sh -c "cp /mnt/host/$TAG/stopat-plain /tmp/ && chmod 0755 /tmp/stopat-plain && /tmp/stopat-plain ${p#plain}" > "$o/$p.out" 2> "$o/$p.err" || rc=$? ;;
    esac
    echo "$p rc=$rc" >> "$o/order.txt"
  done
  vm status > "$o/status-after.txt" 2>&1 || true   # "stopped" after a slice: capprof ended QEMU
  vm down > /dev/null 2>&1 || true
}
# plainf/capf pass "f..." as argv[1] (fault mode), plainh "h..." (handler mode)
boot full plain cap
boot plain-stop plain
boot cap-stop cap
boot plain-start plain
boot cap-start cap
boot plain-k1 plain
boot plain-fault plainfault
boot cap-fault capfault
boot plain-faultcount plainfault
boot cap-faultcount capfault
boot plain-handler plainhandler
boot plain-handlercount plainhandler
boot cap-inter inter
[ -z "${OLD_PLUGIN:-}" ] || boot plain-fault-old plainfault
[ -z "${OLD2_PLUGIN:-}" ] || boot plain-handler-old2 plainhandler
rc=0; python3 "$HERE/stopatcheck.py" "$OUT" > "$OUT/stopatcheck.txt" 2>&1 || rc=$?
cat "$OUT/stopatcheck.txt"
rc2=0; python3 "$HERE/stopat-kit-test.py" "$OUT" "$PL" "$NM" > "$OUT/stopat-kit-test.txt" 2>&1 || rc2=$?
cat "$OUT/stopat-kit-test.txt"
[ $rc = 0 ] && [ $rc2 = 0 ]
