#!/usr/bin/env bash
# Reuse-distance runs (pre-registration docs/plans/2026-10-11-nested-reuse-distance.md, 16d2171a3edc),
# Perl and FFmpeg rows. 8 VMs in parallel: perl x {stock,protected} x {1,2,3} and
# ffmpeg x {stock,protected} x {1}; each VM boots once and runs ONE program on ONE arm.
# Images: experiments/study/reuse-hooks/{perl,ffmpeg}/build.sh (EXP_REUSE_GAP).
set -u
K=$HOME/pg-virtual-kit M=$HOME/sa-reuse-pf R=$M/repo V=$R/capstone/runtime/virtual
export LD_LIBRARY_PATH=$K/toolchain/hostlib
TAG=$(date +%Y%m%d-%H%M%S)
SMOKE=$R/capstone/ports/perl/musl/scripts/smoke.pl
declare -A IMG=(
  [perl-stock]=$M/perl-stock/perl.dom
  [perl-protected]=$M/perl-protected/perl.dom
  [ffmpeg-stock]=$M/ffmpeg-stock/image/ffmpeg.dom
  [ffmpeg-protected]=$M/ffmpeg-protected/image/ffmpeg.dom )
sha256sum ${IMG[@]} $K/kit-sublet/adapter/capstone-vexec $K/kit-sublet/adapter/module/capstone_vm.ko \
  $K/kit-sublet/qemu-system-riscv64 $K/kit-sublet/images/{fw_jump.elf,Image,rootfs.ext2} $V/run-staged.py \
  $SMOKE $K/ffapp-ref/input.mkv $K/ffapp-ref/native.out
pids=()
for id in perl-stock-1 perl-stock-2 perl-stock-3 perl-protected-1 perl-protected-2 perl-protected-3 \
          ffmpeg-stock-1 ffmpeg-protected-1; do
  prog=${id%%-*} arm=${id#*-}; arm=${arm%-*}
  ST=$M/stage-$TAG-$id
  mkdir -p $ST
  cp $K/kit-sublet/adapter/capstone-vexec $K/kit-sublet/adapter/module/capstone_vm.ko $ST/
  cp ${IMG[$prog-$arm]} $ST/prog.dom
  if [ $prog = perl ]; then
    cp $SMOKE $ST/
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
export CAPSTONE_EXEC_DIAGNOSTICS=1
up() { read u _ < /proc/uptime; echo \$u; }
echo "RG_IMAGE:\$(sha256sum /mnt/vm/prog.dom | cut -d ' ' -f 1)"
$PREP
s=\$(up)
/mnt/vm/capstone-vexec /mnt/vm/prog.dom $ARGS > /tmp/rg.out 2> /tmp/rg.err
rc=\$?
e=\$(up)
echo "RG_RUN:rc=\$rc:start=\$s:end=\$e:stdout_sha256=\$(sha256sum /tmp/rg.out | cut -d ' ' -f 1)"
echo RG_STDERR_BEGIN; cat /tmp/rg.err; echo RG_STDERR_END
echo RG_STDOUT_BEGIN; cat /tmp/rg.out; echo RG_STDOUT_END
cd /mnt/vm
rmmod capstone_vm
echo RG_CLEANUP:\$?
echo VIRTUAL_STAGED_DONE
G
  chmod +x $ST/gate.sh
  ( s=$(date +%s)
    CAPSTONE_QEMU_LOCK=$M/qemu-$id.lock python3 $V/run-staged.py --qemu $K/kit-sublet/qemu-system-riscv64 \
      --images $K/kit-sublet/images --stage $ST --work $M/guest-$TAG-$id --timeout 3600 > $M/log-$TAG-$id.txt 2>&1
    echo "STAGED $id rc=$? wall=$(( $(date +%s) - s ))s work=$M/guest-$TAG-$id" ) &
  pids+=($!)
done
date -u +"LAUNCHED %Y-%m-%dT%H:%M:%SZ tag=$TAG"
wait "${pids[@]}"
date -u +"ALL-DONE %Y-%m-%dT%H:%M:%SZ"
