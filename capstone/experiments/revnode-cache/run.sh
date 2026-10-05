#!/bin/bash
# run.sh <workload> : boot capstone-qemu with the revocation-node trace on, run one
# workload, and simulate the node caches while the trace streams through a FIFO.
#
# Workloads: boot-only (control: boot, run `true`), sqlite-O2 (SQLite 3.22 work +
# again + speedtest1 size 1, checked against native), mruby-<arm> (the mruby port's
# smoke.rb in that heap arm: fx-level0, fx-sublet, fx-sublet-gc), and
# mix-<par|seq>-<label>: the programs in $MIX (mruby benchmarks scaled by
# prepare-bench.py, in heap arm $ARM, and `speedtest1`), all at once as separate
# processes (par) or one after another in the same boot (seq), each checked against
# its native output by run-mix.py.
#
# Output: $K/reports/<workload>{,.npl4,.nosup}.json -- one node per cache line, four
# nodes per line, and with the emulated supervisor's own reads left out.
#
# Everything this host-specific run needs is a variable; the defaults are the kits
# the SQLite and mruby lanes built (see README.md).
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
W=$(cd -- "$HERE/../.." && pwd)                       # .../capstone
K=${K:-/tmp/capstone/revnode-cache}
QEMU=${QEMU:?set QEMU to a capstone-qemu build with CAPSTONE_REVNODE_TRACE}
KERNEL=${KERNEL:-/tmp/capstone/delegation-binfmt-linux/arch/riscv/boot/Image}
FIRMWARE=${FIRMWARE:-/tmp/capstone/step-one-ecall/new2/opensbi/platform/generic/firmware/fw_jump.elf}
ROOTFS=${ROOTFS:-/tmp/capstone/cheap/rootfs.ext2}
MODULE=${MODULE:-/tmp/capstone/launch-cost/module/module/capstone.ko}
SQLITE_KIT=${SQLITE_KIT:-/tmp/capstone/sqlite322}
MRUBY_KIT=${MRUBY_KIT:-/tmp/capstone/mruby-arms}
MRUBY_LAUNCHER=${MRUBY_LAUNCHER:-/tmp/capstone/v0-removal-stack/guest/capstone-exec}
PY=${PY:-/tmp/capstone/venv/bin/python3}
BENCH=${BENCH:-$K/bench}                 # prepare-bench.py's output
ARM=${ARM:-fx-sublet-gc}
MIX=${MIX:-}
# CMA the guest reserves for domains, and how much of it the module may keep cached;
# 1024/768 runs one or two mruby sublet-gc domains at once (each takes a 160 MiB heap); the
# kernel places CMA below 4 GiB physical, so about 1.9 GiB is the most it can reserve
CMA_MIB=${CMA_MIB:-1024}
CACHE_MIB=${CACHE_MIB:-768}
name=${1:?usage: run.sh <workload>}

mkdir -p $K/logs $K/reports $K/fifo
SIM=$K/cachesim
cc -O2 -o $SIM $HERE/cachesim.c || exit 1
$PY $HERE/selftest.py $SIM > $K/logs/selftest.txt || { cat $K/logs/selftest.txt; echo "selftest FAILED"; exit 75; }

export PYTHONPATH=$W/runtime/host CAPSTONE_QEMU_LOCK=$K/qemu.lock
# the revoking arms spend far more nodes than the 65536 default
export CAPSTONE_REV_NODES=${CAPSTONE_REV_NODES:-16777216}
FIFO=$K/fifo/$name; rm -f $FIFO; mkfifo $FIFO
export CAPSTONE_REVNODE_TRACE=$FIFO
S=$K/vm-$name
vm() { $PY -m capstone_vm --state $S "$@"; }
case $name in
  sqlite*) SHARE=$SQLITE_KIT/share; LAUNCHER=$SQLITE_KIT/guest/capstone-exec ;;
  mruby-*|boot-only) SHARE=$MRUBY_KIT/share; LAUNCHER=$MRUBY_LAUNCHER ;;
  mix-par-*|mix-seq-*)
    [ -n "$MIX" ] && [ -f $BENCH/native/speedtest1.out ] || { echo "mix needs MIX and BENCH"; exit 2; }
    SHARE=$K/share-mix; LAUNCHER=$MRUBY_LAUNCHER
    mkdir -p $SHARE/bench && cp $BENCH/*.rb $SHARE/bench/ \
      && cp $MRUBY_KIT/share/mruby-$ARM.dom $SQLITE_KIT/app-O2/speedtest1.dom $SHARE/ || exit 1 ;;
  *) echo "unknown workload $name"; exit 2 ;;
esac
vm down > /dev/null 2>&1; rm -rf $S
R=$K/reports/$name
rm -f $R.json $R.npl4.json $R.nosup.json
# the simulators read the trace as QEMU writes it; nothing is stored
( tee >($SIM --nodes-per-line 4 - > $R.npl4.json 2> $R.npl4.err) \
      >($SIM --exclude gc,supervisor - > $R.nosup.json 2> $R.nosup.err) \
    < $FIFO | $SIM - > $R.json 2> $R.err ) &
SIMPID=$!
vm up --qemu $QEMU --kernel $KERNEL --firmware $FIRMWARE --rootfs $ROOTFS --share $SHARE \
  --module $MODULE --launcher $LAUNCHER --cma-mib $CMA_MIB --process-cache-mib $CACHE_MIB \
  > $K/logs/up-$name.log 2>&1 || { echo "up FAILED"; tail -5 $K/logs/up-$name.log; exit 75; }
# the kernel boots without CMA when it cannot reserve it, and every domain then fails
if grep -q "cma: Failed to reserve" $S/console.log; then
  grep "cma: Failed" $S/console.log; vm down > /dev/null 2>&1; echo "RUN FAILED $name: no CMA"; exit 75
fi
start=$(date +%s)
case $name in
  boot-only) vm exec true > $K/logs/$name.txt 2>&1; rc=$? ;;
  sqlite-O2) timeout 10800 $PY $W/ports/sqlite/app/run-app.py --state $S --build $SQLITE_KIT/app-O2 \
      --native $SQLITE_KIT/native-sqlite --share $SHARE --label O2 --size 1 \
      --report $K/logs/$name.json > $K/logs/$name.txt 2>&1; rc=$? ;;
  mix-*) mode=${name#mix-}; mode=${mode%%-*}
      $PY $HERE/run-mix.py --state $S --bench $BENCH --mode $mode --mruby /mnt/host/mruby-$ARM.dom \
        --speedtest1 /mnt/host/speedtest1.dom $MIX > $K/logs/$name.txt 2>&1; rc=$?
      cat $K/logs/$name.txt ;;
  mruby-*) timeout 1800 $PY -m capstone_vm --state $S exec /mnt/host/mruby-${name#mruby-}.dom \
      /mnt/host/files/smoke.rb > $K/logs/$name.txt 2>&1; rc=$?
      grep -q SMOKE_DONE $K/logs/$name.txt || rc=1 ;;
esac
echo "workload rc=$rc wall=$(( $(date +%s) - start ))s"
vm down > /dev/null 2>&1
wait $SIMPID
for i in $(seq 1800); do
  grep -qx "}" $R.npl4.json 2>/dev/null && grep -qx "}" $R.nosup.json 2>/dev/null && break; sleep 1
done
rm -f $FIFO
bad=0
for f in $R.err $R.npl4.err $R.nosup.err; do [ -s $f ] && { echo "SIM ERROR $f"; cat $f; bad=1; }; done
for f in $R.json $R.npl4.json $R.nosup.json; do grep -qx "}" $f 2>/dev/null || { echo "no report $f"; bad=1; }; done
[ $rc = 0 ] && [ $bad = 0 ] || { echo "RUN FAILED $name"; exit 1; }
echo "RUN-DONE $name"
