#!/bin/bash
# B0.6 controls (docs/plans/b0-silicon-delegated-runtime.md): A = fw-b0 with fabrication ON (legacy image must run, b0-hello must still pass);
# B = fw-ctl (no B0.1) with fabrication OFF (b0-hello must be REFUSED). One VM at a time.
main() {
  set -u
  # CAPSTONE_BUILDROOT_DIR: the QEMU buildroot (capstone-test-env.sh sets it); B0_PLATFORM: the frozen platform
  # (the plan's B0.6 section lists each file and its sha256).
  local BR=${CAPSTONE_BUILDROOT_DIR:?} P=${B0_PLATFORM:-/tmp/capstone/b0/platform}
  local G=$P/guest
  local VM=/tmp/capstone/b0/tools/bin/capstone-vm
  local ARGS=(--qemu $P/qemu-system-riscv64 --kernel $BR/build/images/Image --rootfs $BR/build/images/rootfs.ext2
              --launcher $G/capstone-exec --module $G/capstone.ko --ssh-server $G/ssh_server
              --share /tmp/capstone/b0/share --boot-timeout 600)
  cd /tmp/capstone/b0
  $VM --state /tmp/capstone/b0/vm down > /dev/null 2>&1
  for arm in A B; do
    local S=/tmp/capstone/b0/vm-$arm FW FAB
    if [ $arm = A ]; then FW=$P/fw-b0/opensbi/build/platform/generic/firmware/fw_jump.elf; FAB=1
    else FW=$P/fw-ctl/opensbi/build/platform/generic/firmware/fw_jump.elf; FAB=0; fi
    rm -rf $S
    echo "== arm $arm: firmware $(sha256sum $FW | cut -c1-12) GP_FABRICATE=$FAB"
    CAPSTONE_GP_FABRICATE=$FAB CAPSTONE_MOVC_NULL_SCALAR=1 taskset -c 0-7,32-39 nice -n 10 timeout 900 \
      $VM --state $S up "${ARGS[@]}" --firmware $FW > $S.up.log 2>&1
    echo "   up rc=$? $(tail -1 $S.up.log | cut -c1-100)"
    for img in b0-hello legacy-ctl; do
      [ $arm = B ] && [ $img = legacy-ctl ] && continue
      timeout 300 $VM --state $S run --result $S.$img.json /mnt/host/$img.dom > $S.$img.out 2> $S.$img.err
      echo "   $img rc=$? out=[$(head -1 $S.$img.out | cut -c1-70)] err=[$(tail -1 $S.$img.err | cut -c1-150)]"
    done
    $VM --state $S down > /dev/null 2>&1
  done
  echo CONTROLS_DONE
}
main "$@"; exit $?
