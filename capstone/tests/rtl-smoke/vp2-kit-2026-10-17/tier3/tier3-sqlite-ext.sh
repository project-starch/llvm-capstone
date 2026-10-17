#!/bin/bash
# Tier 3, sqlite/engine-repros, the rerun with --ext-src (SQLite 3.22.0 full source, SHA3-256 f8695d69... as the kit's
# prepare-sources.sh pins it), so cases 06 (expert) and 22 (spellfix) build. Same kit, tree and controls as the
# 10:46-11:02 rerun; all 33 cells run again, the 31 already read are its reproduction check. Non-exact both arms, then
# --exact-bounds for nested-pools. Each VM session is bounded and always comes down.
K=/tmp/capstone/vp2-prep/v6-fix/out/kit ST=/tmp/capstone/t3vm SH=/tmp/capstone/t3vm-share T=/tmp/capstone/vp2-prep/tier3-fix/sqlite-ext
SRC=/tmp/capstone/vp2-prep/v6-fix/src O=/tmp/capstone/vp2-prep/v6-fix EXT=/tmp/capstone/sqlite322-src/sqlite-src-3220000
vm_up() {
  rm -rf $ST; local end=$(( $(date +%s) + 1800 )) rc
  while :; do
    timeout 600 python3 -m capstone_vm --state $ST up --qemu $K/qemu-system-riscv64 --kernel $K/images/Image --firmware $K/images/fw_jump.elf \
      --rootfs $K/images/rootfs.ext2 --share $SH --profile virtual --launcher $K/adapter/capstone-vexec --module $K/adapter/module/capstone_vm.ko \
      --job-helper $K/adapter/capstone-job --ssh-server /tmp/capstone/b0/platform/guest/ssh_server $1 > $T/vm-up$1.log 2>&1; rc=$?
    grep -q 'Another Capstone VM owns' $T/vm-up$1.log && [ $(date +%s) -lt $end ] && { sleep 30; continue; }; break
  done
  echo "VM up rc=$rc $(date +%T) $1"; return $rc
}
vm_down() { python3 -m capstone_vm --state $ST down > $T/vm-down.log 2>&1; echo "VM down rc=$? $(date +%T)"; }
run_arm() {  # arm suffix
  timeout 3600 python3 $C/shared/run-virtual.py $T/out-$1$2 --state $ST --arm $1 --build $T/build-$1 \
    --llvm-bin /tmp/capstone/vp2-prep/llvm-bin-v6 --raw $T/raw-$1$2 > $T/run-$1$2.log 2>&1
  echo "SQ $1$2 rc=$? $(date +%T) $(grep -E '^---' $T/run-$1$2.log | cut -c1-160)"
}
main() {
  C=$SRC/capstone/bug-corpora/sqlite/engine-repros
  mkdir -p $T $SH; export CAPSTONE_QEMU_LOCK=$HOME/.capstone-locks/qemu.lock
  for arm in virtual-malloc virtual-nested-pools; do
    local extra=""; [ $arm = virtual-nested-pools ] && extra="--sublet"
    rm -rf $T/build-$arm
    nice -n 10 python3 $SRC/capstone/ports/sqlite/repro322/build-virtual.py --sdk $O/out/sdk --amalgamation $O/out/sqlite-source \
      --out $T/build-$arm --memsys5 $extra --ext-src $EXT \
      --observe $C/controls/control_uaf_mem5.c --observe $C/controls/control_bounds_mem5.c > $T/build-$arm.log 2>&1
    echo "SQ build $arm rc=$? doms=$(ls $T/build-$arm/*.dom 2>/dev/null | wc -l) $(date +%T)"
  done
  cd $SRC/capstone/runtime/host || return 1
  trap vm_down EXIT
  vm_up || return 1
  run_arm virtual-malloc; run_arm virtual-nested-pools
  vm_down; trap - EXIT
  trap vm_down EXIT
  vm_up --exact-bounds || return 1
  run_arm virtual-nested-pools -exact
  echo SQ_EXT_DONE
}
main "$@"; exit
