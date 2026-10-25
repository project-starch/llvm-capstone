#!/bin/bash
# The QEMU kit injection's behavioural test (the runtime lane's handover rule): run.py on QEMU 3b8040d5ff6e (binary
# 809d1fc107e64515, one context slot by default) with the v9 images, once with the CEPC injection firmware (c170e49) and
# once with f5f1bbe's as the control. Prediction for c170e49: at `threads` (the second CSRUNV refused, 29) the monitor
# prints INJS 0x1d and the kernel takes the trap at stvec (its own trap report; no INJL, no spin). For f5f1bbe: INJS,
# then INJL with the stale mepc and a spin until the timeout.
main() {
  set -u
  local H=${CAPSTONE_HDD:?the scratch volume of the run host}
  local B=$H/qinj-run-0343 K=$H/v9-kit R=$H/v9-kit/src/capstone/runtime/virtual/run.py
  [ "$(sha256sum < $B/qemu-system-riscv64 | cut -c1-16)" = 809d1fc107e64515 ] || { echo "STOP: qemu binary"; return 1; }
  local arm t
  for arm in cepc f5f1bbe; do
    t=300; [ $arm = f5f1bbe ] && t=180
    echo "ARM $arm fw $(sha256sum < $B/$arm/images/fw_jump.elf | cut -c1-16) $(date +%T)"
    CAPSTONE_QEMU_LOCK=$HOME/.capstone-locks/qemu.lock CAPSTONE_LLVM_BIN=/tmp/capstone/llvm-dev-current-20261017/bin \
      timeout $((t + 300)) taskset -c 16-31,48-63 python3 $R --qemu $B/qemu-system-riscv64 --images $B/$arm/images \
      --adapter $K/out/adapter-flat --application $K/out/contract.dom --sqlite $K/out/sqlite/sqlite3.dom \
      --mruby $K/out/mruby/src/mruby/build/capstone/bin/mruby --record $B/$arm/record.json --timeout $t > $B/$arm/runpy.log 2>&1
    echo "  run.py rc=$? $(date +%T); serial: $(grep -a -o 'serial[^ ]*log' $B/$arm/runpy.log | head -1)"
  done
  echo PAIR-DONE
}
main "$@"; exit
