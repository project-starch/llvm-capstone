#!/usr/bin/env bash
# Stage one matching runtime source set for an external module or in-tree build.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../tests/capstone-test-env.sh"
out=${1:?Output directory}
mkdir -p "$out/capstone"
cp "$here/module/Makefile" "$here/module/capstone_vm.c" "$here/module/binfmt-dom.h" "$here/module/delegate-dom.h" "$here/wire.h" "$here/vm-abi.h" "$out/"
cp "$here/native-grant.h" "$out/"
cp "${CAPSTONE_QEMU_SOURCE:-$CAPSTONE_REPO_ROOT/capstone/capstone-qemu}/target/riscv/cap_rev_table_abi.h" "$out/"
# The representability predicate, SHARED rather than copied: the module's arena test, grant() in
# the launcher and the allocator must agree on one rule, and that rule is cross-checked against
# the bounds encoder by tests/exact-bounds-iii/repr-cross-check.py. Copied from the tree being
# built, like the ABI header above, so a kit cannot pair a module and an SDK across revisions.
cp "$here/../include/capstone/representable.h" "$out/"
cp "$here/../include/capstone/delegate.h" "$out/capstone/"
cp "$here/../linux/delegate-policy.h" "$out/"
cp "$here/module/resource-dom.h" "$here/module/checked-copy-dom.h" "$out/"
cp "$here/module/rt-dom.h" "$out/"
cp "$here/module/native-files-dom.h" "$here/module/native-gcc-dom.h" "$here/module/native-syscall-dom.h" "$out/"
cp "$here/../common/native-syscall.c" "$out/native-syscall-core.c"
cp "$here/../common/native-syscall-table.c" "$out/"
cp "$here/../include/capstone/"{native-syscall,native-syscall-abi}.h "$out/capstone/"
cp "$here/checked-copy.c" "$out/checked-copy-core.c"
cp "$here/../include/capstone/"{checked-copy,native-context}.h "$out/capstone/"
cp "$here/../common/delegate.c" "$out/delegate-wire.c"
