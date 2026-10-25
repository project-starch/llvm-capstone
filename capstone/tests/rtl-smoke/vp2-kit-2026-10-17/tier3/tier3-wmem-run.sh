#!/bin/bash
# Tier 3: wireshark/wmem-repros, both arms, on a capstone_vm VM from the v6-fix kit (pre-registered at lane 9c512e9bad72).
# One bounded VM session; the VM always comes down (it holds the host-wide QEMU lock while up).
main() {
  K=/tmp/capstone/vp2-prep/v6-fix/out/kit ST=/tmp/capstone/t3vm SH=/tmp/capstone/t3vm-share T=/tmp/capstone/vp2-prep/tier3-fix/wmem   # globals: the EXIT trap reads them
  local SRC=/tmp/capstone/vp2-prep/v6-fix/src C=/tmp/capstone/vp2-prep/v6-fix/src/capstone/bug-corpora/wireshark/wmem-repros
  export CAPSTONE_QEMU_LOCK=$HOME/.capstone-locks/qemu.lock
  cd $SRC/capstone/runtime/host || return 1
  down(){ python3 -m capstone_vm --state $ST down > $T/vm-down.log 2>&1; echo "VM down rc=$? $(date +%T)"; }
  trap down EXIT
  rm -rf $ST; mkdir -p $SH
  local end=$(( $(date +%s) + 1800 )) rc
  while :; do
    timeout 600 python3 -m capstone_vm --state $ST up --qemu $K/qemu-system-riscv64 --kernel $K/images/Image --firmware $K/images/fw_jump.elf \
      --rootfs $K/images/rootfs.ext2 --share $SH --profile virtual --launcher $K/adapter/capstone-vexec --module $K/adapter/module/capstone_vm.ko \
      --job-helper $K/adapter/capstone-job --ssh-server /tmp/capstone/b0/platform/guest/ssh_server ${EXACT:+--exact-bounds} > $T/vm-up.log 2>&1; rc=$?
    grep -q 'Another Capstone VM owns' $T/vm-up.log && [ $(date +%s) -lt $end ] && { sleep 30; continue; }; break
  done
  echo "VM up rc=$rc $(date +%T): $(tail -1 $T/vm-up.log | cut -c1-160)"
  python3 -m capstone_vm --state $ST status 2>&1 | head -2
  [ $rc -eq 0 ] || return 1
  for s in ${ARMS:-OFF ON}; do
    local arm=virtual-malloc; [ $s = ON ] && arm=virtual-nested-pools
    echo "WMEM $arm START $(date +%T)"
    timeout 3600 python3 $C/shared/run-defects.py $T/out-$arm${EXACT:+-exact} --state $ST --arm $arm --hosted-build $T/cases-$s \
      --controls-hosted-build $T/controls-$s --llvm-bin /tmp/capstone/vp2-prep/llvm-bin-v6 --raw $T/raw-$arm${EXACT:+-exact} > $T/run-$arm${EXACT:+-exact}.log 2>&1
    echo "WMEM $arm END rc=$? $(date +%T) $(tail -2 $T/run-$arm${EXACT:+-exact}.log | head -1 | cut -c1-140)"
  done
  echo WMEM_RUN_DONE
}
main "$@"
