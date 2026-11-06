#!/usr/bin/env bash
# run.sh QEMU IMAGES ADAPTER CONTRACT ABI_PROBE OUT [pinned|unpinned]
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
source "$here/../../../../tests/capstone-test-env.sh"
qemu=${1:?QEMU}; images=${2:?IMAGES}; adapter=${3:?ADAPTER}
contract=${4:?CONTRACT}; probe=${5:?ABI_PROBE}; out=${6:?OUT}; mode=${7:-pinned}
case "$mode" in pinned) unpinned=0;; unpinned) unpinned=1;; *) exit 2;; esac
mkdir "$out"
mkdir "$out/stage"
cp "$adapter/capstone-vexec" "$adapter/module/capstone_vm.ko" "$out/stage/"
cp "$contract" "$out/stage/contract.dom"
cp "$probe" "$out/stage/abi-probe"
cp "$here/gate.sh" "$out/stage/"
printf '%s\n' "$unpinned" > "$out/stage/unpinned"
firmware=${COPYABLE_FIRMWARE:-$CAPSTONE_REPO_ROOT/capstone/capstone-qemu/pc-bios/opensbi-riscv64-generic-fw_dynamic.bin}
sha256sum "$qemu" "$images/Image" "$images/rootfs.ext2" "$firmware" \
    "$adapter/capstone-vexec" "$adapter/module/capstone_vm.ko" "$contract" "$probe" \
    > "$out/inputs.sha256"
export CAPSTONE_QEMU_CPU_EXTRA=x-capstone-supervisor-slots=32,x-pcrel=false,x-capstone-raw-moves=true,x-capstone-test-csrs=false
python3 "$here/../../run-staged.py" --qemu "$qemu" --images "$images" \
    --firmware "$firmware" --stage "$out/stage" --work "$out/run" \
    --disk-mib 64 --timeout "${COPYABLE_TIMEOUT:-180}"
python3 "$here/check.py" "$out/run/serial.log" --output "$out/qualification.json"
