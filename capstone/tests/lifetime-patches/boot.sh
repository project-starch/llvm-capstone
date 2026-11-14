#!/usr/bin/env bash
# boot.sh STAGE WORK: one trap-route boot of STAGE (its gate.sh runs in the guest) on the life-c kit
# (kernel Image-0133b9e21cd, qemu-1b, firmware f-keep); prints the serial lines that start with TEST
# and the kernel's Sublet fault lines.
set -euo pipefail
L=$HOME/asr-work/life-c K=$L/kit
cp "$K/adapter/capstone-vexec" "$1/capstone-vexec"
export CAPSTONE_QEMU_LOCK=$L/qemu.lock
python3 $HOME/asr-work/eval/tree/capstone/runtime/virtual/run-staged.py --qemu "$K/qemu-system-riscv64" \
  --images "$K/images" --stage "$1" --work "$2" --timeout "${BOOT_TIMEOUT:-1200}" > "$2.run.log" 2>&1 || echo "run-staged rc=$?"
tr -d '\r' < "$2/serial.log" | grep -aE '^(TEST|CPYTEST|MCTEST|WMTEST)|sublet: fault|capstone-exec: domain fault' || true
