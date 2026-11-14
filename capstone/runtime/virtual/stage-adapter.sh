#!/usr/bin/env bash
# Stage one matching source set for the kernel's built-in Sublet code.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../tests/capstone-test-env.sh"
out=${1:?Output directory}
mkdir -p "$out/capstone"
cp "$here/module/Makefile" "$here/module/capstone_vm.c" "$here/module/binfmt-dom.h" "$here/module/delegate-dom.h" "$here/vm-abi.h" "$out/"
cp "$here/module/sublet-format.h" "$here/module/sublet-table.h" "$out/"
# install-native-task also installs this exact interface into include/linux.
cp "$here/kernel-capstone-helpers.h" "$out/capstone_native_helpers.h"
# The table ABI of the QEMU the kernel runs on; sublet-format.h refuses any other format.
cp "${CAPSTONE_QEMU_SOURCE:-$CAPSTONE_REPO_ROOT/capstone/capstone-qemu}/target/riscv/cap_rev_table_abi.h" "$out/"
# The representability predicate, shared with the allocator, from the tree being built.
cp "$here/../include/capstone/representable.h" "$out/"
cp "$here/../include/capstone/delegate.h" "$out/capstone/"
cp "$here/module/resource-dom.h" "$out/"
cp "$here/module/native-files-dom.h" "$here/module/native-gcc-dom.h" "$here/module/native-syscall-dom.h" "$out/"
cp "$here/../common/native-syscall-table.c" "$out/"
cp "$here/../include/capstone/"{native-syscall,native-syscall-abi,native-context}.h "$out/capstone/"
cp "$here/../common/delegate.c" "$out/delegate-wire.c"

cp "$here/module/native-spawn-dom.h" "$out/"
cp "$here/../include/capstone/spawn.h" "$out/capstone/"
cp "$here/../common/spawn.c" "$out/spawn-wire.c"

cp "$here/module/native-directory-dom.h" "$here/module/native-time-dom.h" "$out/"
