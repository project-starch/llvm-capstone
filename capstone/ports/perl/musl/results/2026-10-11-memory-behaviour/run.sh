#!/usr/bin/env bash
# Memory-behaviour runs (pre-registration 2026-10-11-protected-memory-behaviour.md), Perl and FFmpeg rows.
# 12 VMs in parallel: {perl,ffmpeg} x {stock,protected} x {1,2,3}; each VM boots once and runs ONE program on ONE arm.
set -u
K=$HOME/pg-virtual-kit M=$HOME/sa-mem-pf R=$HOME/sa-perl/repo V=$R/capstone/runtime/virtual
export LD_LIBRARY_PATH=$K/toolchain/hostlib
TAG=$(date +%Y%m%d-%H%M%S)
declare -A IMG=(
  [perl-stock]=$HOME/sa-perl/stage-smoke2k/perl-stock.dom
  [perl-protected]=$HOME/sa-perl/stage-smoke2k/perl-protected.dom
  [ffmpeg-stock]=$K/out/ffapp/vstock/image/ffmpeg.dom
  [ffmpeg-protected]=$K/out/ffapp/vpools/image/ffmpeg.dom )
sha256sum ${IMG[@]} $K/kit-sublet/adapter/capstone-vexec $K/kit-sublet/adapter/module/capstone_vm.ko \
  $K/kit-sublet/qemu-system-riscv64 $K/kit-sublet/images/{fw_jump.elf,Image,rootfs.ext2} $V/run-staged.py \
  $HOME/sa-perl/stage-smoke2k/smoke.pl $K/ffapp-ref/input.mkv $K/ffapp-ref/native.out
pids=()
for prog in perl ffmpeg; do for arm in stock protected; do for rep in 1 2 3; do
  id=$prog-$arm-$rep ST=$M/stage-$TAG-$id
  mkdir -p $ST
  cp $K/kit-sublet/adapter/capstone-vexec $K/kit-sublet/adapter/module/capstone_vm.ko $ST/
  cp ${IMG[$prog-$arm]} $ST/prog.dom
  if [ $prog = perl ]; then
    cp $HOME/sa-perl/stage-smoke2k/smoke.pl $ST/
    PREP='mkdir -p /tmp/run; cp /mnt/vm/smoke.pl /tmp/run/smoke.pl; cd /tmp/run; export CAPSTONE_PERL_SMOKE_OBJECTS=2000'
    ARGS='smoke.pl /tmp/run'
  else
    cp $K/ffapp-ref/input.mkv $ST/
    PREP='cd /mnt/vm'
    ARGS='/mnt/vm/input.mkv 1'
  fi
  cat > $ST/gate.sh <<G
#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export CAPSTONE_EXEC_DIAGNOSTICS=1 CAPSTONE_VM_STATS=1
up() { read u _ < /proc/uptime; echo \$u; }
echo "MB_IMAGE:\$(sha256sum /mnt/vm/prog.dom | cut -d ' ' -f 1)"
printf 'echo \$\$ > /tmp/mb.pid\nexec "\$@"\n' > /tmp/mb-wrap.sh
$PREP
echo "MB_STATS_BEFORE:\$(/mnt/vm/capstone-vexec --stats)"
s=\$(up)
time -o /tmp/mb.time -v sh /tmp/mb-wrap.sh /mnt/vm/capstone-vexec /mnt/vm/prog.dom $ARGS > /tmp/mb.out 2> /tmp/mb.err &
tp=\$!
while [ ! -s /tmp/mb.pid ]; do usleep 10000; done
pid=\$(cat /tmp/mb.pid)
: > /tmp/mb.samples
while kill -0 \$pid 2>/dev/null; do
  h=; r=
  while read k v x; do case \$k in VmHWM:) h=\$v;; VmRSS:) r=\$v;; esac; done < /proc/\$pid/status
  t=\$(up)
  [ -n "\$h" ] && echo "\$t \$r \$h" >> /tmp/mb.samples
  usleep 100000
done
wait \$tp; rc=\$?
e=\$(up)
echo "MB_STATS_AFTER:\$(/mnt/vm/capstone-vexec --stats)"
echo "MB_RUN:rc=\$rc:start=\$s:end=\$e:pid=\$pid:stdout_sha256=\$(sha256sum /tmp/mb.out | cut -d ' ' -f 1)"
echo MB_SAMPLES_BEGIN; cat /tmp/mb.samples; echo MB_SAMPLES_END
echo MB_TIME_BEGIN; cat /tmp/mb.time; echo MB_TIME_END
echo MB_STDERR_BEGIN; cat /tmp/mb.err; echo MB_STDERR_END
echo MB_STDOUT_BEGIN; cat /tmp/mb.out; echo MB_STDOUT_END
cd /mnt/vm
rmmod capstone_vm
echo MB_CLEANUP:\$?
echo VIRTUAL_STAGED_DONE
G
  chmod +x $ST/gate.sh
  ( s=$(date +%s)
    CAPSTONE_QEMU_LOCK=$M/qemu-$id.lock python3 $V/run-staged.py --qemu $K/kit-sublet/qemu-system-riscv64 \
      --images $K/kit-sublet/images --stage $ST --work $M/guest-$TAG-$id --timeout 3600 > $M/log-$TAG-$id.txt 2>&1
    echo "STAGED $id rc=$? wall=$(( $(date +%s) - s ))s work=$M/guest-$TAG-$id" ) &
  pids+=($!)
done; done; done
date -u +"LAUNCHED %Y-%m-%dT%H:%M:%SZ tag=$TAG"
wait "${pids[@]}"
date -u +"ALL-DONE %Y-%m-%dT%H:%M:%SZ"
