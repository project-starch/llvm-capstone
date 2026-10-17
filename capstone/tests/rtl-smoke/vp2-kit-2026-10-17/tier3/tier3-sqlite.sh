#!/bin/bash
# Tier 3: sqlite/engine-repros, both arms, on the fixed v6 revision (pre-registered at lane b30bbc54228b).
# Builds first (no QEMU), then one bounded capstone_vm session; the VM always comes down.
K=/tmp/capstone/vp2-prep/v6-fix/out/kit ST=/tmp/capstone/t3vm SH=/tmp/capstone/t3vm-share T=/tmp/capstone/vp2-prep/tier3-fix/sqlite
SRC=/tmp/capstone/vp2-prep/v6-fix/src O=/tmp/capstone/vp2-prep/v6-fix
main() {
  local C=$SRC/capstone/bug-corpora/sqlite/engine-repros
  mkdir -p $T $SH; export CAPSTONE_QEMU_LOCK=$HOME/.capstone-locks/qemu.lock
  [ -n "${SKIP_BUILD:-}" ] || for arm in virtual-malloc virtual-nested-pools; do
    local extra=""; [ $arm = virtual-nested-pools ] && extra="--sublet"
    rm -rf $T/build-$arm
    nice -n 10 python3 $SRC/capstone/ports/sqlite/repro322/build-virtual.py --sdk $O/out/sdk --amalgamation $O/out/sqlite-source \
      --out $T/build-$arm --memsys5 $extra \
      --observe $C/controls/control_uaf_mem5.c --observe $C/controls/control_bounds_mem5.c > $T/build-$arm.log 2>&1
    echo "SQ build $arm rc=$? doms=$(ls $T/build-$arm/*.dom 2>/dev/null | wc -l) $(date +%T)"
  done
  cd $SRC/capstone/runtime/host || return 1
  trap 'python3 -m capstone_vm --state $ST down > $T/vm-down.log 2>&1; echo "VM down rc=$? $(date +%T)"' EXIT
  rm -rf $ST; local end=$(( $(date +%s) + 1800 )) rc
  while :; do
    timeout 600 python3 -m capstone_vm --state $ST up --qemu $K/qemu-system-riscv64 --kernel $K/images/Image --firmware $K/images/fw_jump.elf \
      --rootfs $K/images/rootfs.ext2 --share $SH --profile virtual --launcher $K/adapter/capstone-vexec --module $K/adapter/module/capstone_vm.ko \
      --job-helper $K/adapter/capstone-job --ssh-server /tmp/capstone/b0/platform/guest/ssh_server ${EXACT:+--exact-bounds} > $T/vm-up.log 2>&1; rc=$?
    grep -q 'Another Capstone VM owns' $T/vm-up.log && [ $(date +%s) -lt $end ] && { sleep 30; continue; }; break
  done
  echo "VM up rc=$rc $(date +%T) exact=${EXACT:-0}"; [ $rc -eq 0 ] || return 1
  for arm in ${ARMS:-virtual-malloc virtual-nested-pools}; do
    timeout 3600 python3 $C/shared/run-virtual.py $T/out-$arm${EXACT:+-exact} --state $ST --arm $arm --build $T/build-$arm \
      --llvm-bin /tmp/capstone/vp2-prep/llvm-bin-v6 --raw $T/raw-$arm${EXACT:+-exact} > $T/run-$arm${EXACT:+-exact}.log 2>&1
    echo "SQ $arm rc=$? $(date +%T) $(grep -E '^---' $T/run-$arm${EXACT:+-exact}.log | cut -c1-140)"
  done
  echo SQ_DONE
}
main "$@"
