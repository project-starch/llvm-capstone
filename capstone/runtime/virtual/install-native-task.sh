#!/usr/bin/env bash
# Apply once to a disposable Linux 6.1 tree with the tag-aware memory patch.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../tests/capstone-test-env.sh"
kernel=${1:?Prepared Linux source/build tree}
[[ -f "$kernel/Makefile" && -f "$kernel/arch/riscv/kernel/entry.S" ]]
[[ ! -e "$kernel/drivers/misc/capstone" ]] || {
    echo 'Use a fresh disposable kernel tree; capstone driver already exists.' >&2
    exit 1
}
patch --dry-run -d "$kernel" -p1 < "$here/kernel-native-task.patch"
# Stage before patching so missing source inputs fail without changing Linux.
stage=$(mktemp -d "${CAPSTONE_TMP_ROOT:-/tmp/capstone}/native-task-source.XXXXXX")
trap 'rm -rf -- "$stage"' EXIT
bash "$here/stage-adapter.sh" "$stage"
sed -i 's/^obj-m += capstone_vm.o$/obj-y += capstone_vm.o/' "$stage/Makefile"
patch -d "$kernel" -p1 < "$here/kernel-native-task.patch"
mv "$stage" "$kernel/drivers/misc/capstone"
printf '\nobj-$(CONFIG_RISCV_CAPSTONE_TASK) += capstone/\n' >> "$kernel/drivers/misc/Makefile"
"$kernel/scripts/config" --file "$kernel/.config" --disable SMP \
    --disable VIRT_CPU_ACCOUNTING_NATIVE --disable TICK_CPU_ACCOUNTING \
    --enable VIRT_CPU_ACCOUNTING_GEN --enable CONTEXT_TRACKING_USER_FORCE \
    --enable RISCV_CAPSTONE_TASK
echo 'Sources installed. Run make ARCH=riscv olddefconfig, then build Image.'
