#!/usr/bin/env bash
# apply.sh LINUX: turn a fresh Linux 6.1 tree into the kernel that runs Sublet programs.
#
# Applies patches/series in order, copies the files the kernel shares with QEMU and the libc
# next to the Sublet sources, and enables the option. The series is the kernel's Sublet code
# in arch/riscv (arch/riscv/mm/sublet-table.c and arch/riscv/kernel/sublet/) and the hooks it
# needs in Linux: capability tags that follow physical pages, the task skeleton, spawn and the
# helpers of the system calls the kernel implements for a protected program.
#
# The shared files are copied, not patched, so each has one source of truth:
#   cap_rev_table_abi.h  from $CAPSTONE_QEMU_SOURCE (default ../capstone-qemu), the QEMU the
#                        kernel runs on; sublet-format.h refuses any format but the 63-bit one
#   representable.h, capstone/*.h, vm-abi.h, delegate.c, spawn.c, native-syscall-table.c
#                        from ../runtime, shared with the libc
# The caller provides LINUX/.config. Then: make ARCH=riscv olddefconfig, and build Image.
# Plain bash 3.2 is enough.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime=$here/../runtime
qemu=${CAPSTONE_QEMU_SOURCE:-$here/../capstone-qemu}
kernel=${1:?Linux 6.1 source tree}
[[ -f $kernel/Makefile && -f $kernel/arch/riscv/kernel/entry.S ]] || { echo "not a Linux tree: $kernel" >&2; exit 1; }
[[ ! -e $kernel/arch/riscv/kernel/sublet ]] || { echo "use a fresh tree: the series is already applied" >&2; exit 1; }
[[ -f $qemu/target/riscv/cap_rev_table_abi.h ]] || { echo "no QEMU tree at $qemu (set CAPSTONE_QEMU_SOURCE)" >&2; exit 1; }
# The caller provides the kernel configuration; this script only adds the Sublet option to it.
[[ -f $kernel/.config ]] || { echo "no $kernel/.config: provide the platform's configuration first" >&2; exit 1; }

# Check the whole series on copies of the files it touches before Linux changes, so a patch
# that does not apply leaves the tree as it was.
stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT
patches=()
while IFS= read -r p; do
    [[ -z $p || $p == \#* ]] || patches+=("$p")
done < "$here/patches/series"
for p in "${patches[@]}"; do
    sed -n 's|^--- a/||p' "$here/patches/$p"
done | sort -u | while read -r file; do
    if [[ -f $kernel/$file ]]; then
        mkdir -p "$stage/$(dirname -- "$file")"
        cp "$kernel/$file" "$stage/$file"
    fi
done
for p in "${patches[@]}"; do
    patch -s -d "$stage" -p1 --forward < "$here/patches/$p" || { echo "does not apply: $p" >&2; exit 1; }
done

# Check the shared files too, so a missing one also leaves the tree as it was.
shared=("$qemu/target/riscv/cap_rev_table_abi.h" "$runtime/include/capstone/representable.h"
        "$runtime/virtual/vm-abi.h" "$runtime/common/delegate.c" "$runtime/common/spawn.c"
        "$runtime/common/native-syscall-table.c")
for h in delegate native-syscall native-syscall-abi native-context spawn; do
    shared+=("$runtime/include/capstone/$h.h")
done
for f in "${shared[@]}"; do
    [[ -r $f ]] || { echo "missing shared file: $f" >&2; exit 1; }
done

for p in "${patches[@]}"; do
    patch -s -d "$kernel" -p1 --forward < "$here/patches/$p"
done

mm=$kernel/arch/riscv/mm
sublet=$kernel/arch/riscv/kernel/sublet
cp "$qemu/target/riscv/cap_rev_table_abi.h" "$mm/"
cp "$runtime/include/capstone/representable.h" "$runtime/virtual/vm-abi.h" "$mm/"
mkdir -p "$sublet/capstone"
cp "$runtime/virtual/vm-abi.h" "$sublet/"
cp "$runtime/include/capstone/"{delegate,native-syscall,native-syscall-abi,native-context,spawn}.h "$sublet/capstone/"
cp "$runtime/common/delegate.c" "$sublet/delegate-wire.c"
cp "$runtime/common/spawn.c" "$sublet/spawn-wire.c"
cp "$runtime/common/native-syscall-table.c" "$sublet/"

"$kernel/scripts/config" --file "$kernel/.config" --disable SMP --enable RISCV_CAPSTONE_TASK
echo "Series applied. Run make ARCH=riscv olddefconfig, then build Image."
