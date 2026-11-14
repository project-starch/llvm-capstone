#!/usr/bin/env bash
# The binder's generated table and core against a Linux 6.1 tree, on the host, then controls
# that must fail.
#   tests/syscall-bind/run.sh LINUX [KERNEL_SOURCES]
# LINUX is a Linux 6.1 source tree (uapi headers and SYSCALL_DEFINE sources). KERNEL_SOURCES is
# the directory that holds bind.h and bind-core.c (arch/riscv/kernel/sublet of a tree with the
# series applied); without it they come from the series patch. The table is generated afresh
# by tools/gen-syscall-pointers.py and must match the one in the series. Exit 0 only if every
# check passes and every control fails.
set -euo pipefail
linux=$(cd -- "${1:?Linux tree}" && pwd)
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
kernel_dir=$here/../..
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
status=0
if (($# >= 2)); then
    src=$(cd -- "$2" && pwd)
else
    src=$work/series
    python3 - "$kernel_dir/patches/0032-riscv-sublet-binder.patch" "$kernel_dir/patches/0031-riscv-sublet-syscall-pointer-table.patch" "$src" <<'PYTHON'
import os, sys
out = sys.argv[-1]
os.makedirs(out)
for patch in sys.argv[1:-1]:
    lines, name = open(patch).read().split('\n'), None
    for i, line in enumerate(lines + ['']):
        if line.startswith('+++ b/') and lines[i - 1] == '--- /dev/null':
            name, body = os.path.basename(line[6:]), []
        elif name and line.startswith('@@'):
            continue
        elif name and line.startswith('+'):
            body.append(line[1:])
        elif name:
            open(os.path.join(out, name), 'w').write('\n'.join(body) + '\n')
            name = None
PYTHON
fi

python3 "$kernel_dir/tools/gen-syscall-pointers.py" "$linux" > "$work/syscall-pointers.c"
if cmp -s "$work/syscall-pointers.c" "$src/syscall-pointers.c"; then
    echo "PASS the series' table is the generator's output"
else
    echo "FAIL the series' table differs from the generator's output"; status=1
fi

# The riscv64 call numbers: every __NR_ macro of the tree's unistd.h, through the preprocessor.
${CPP:-cpp} -dM -nostdinc -D__riscv_xlen=64 -I"$linux/arch/riscv/include/uapi" -I"$linux/include/uapi" \
    "$linux/arch/riscv/include/uapi/asm/unistd.h" | grep -E '^#define __NR(3264)?_' > "$work/sublet-nr.h"
build() {   # build <output> <generated table> [cflags...]
    local out=$1 table=$2; shift 2
    ${CC:-cc} -std=gnu11 -O2 -Wall -Werror -Wno-unused-function -D_GNU_SOURCE -I"$src" -I"$work" \
        "$@" "$here/checks.c" "$src/bind-core.c" "$table" -o "$out"
}
if build "$work/checks" "$work/syscall-pointers.c" > "$work/build.log" 2>&1 && "$work/checks" > "$work/checks.log"; then
    echo "PASS checks ($(grep -c '^PASS' "$work/checks.log") of them)"
else
    echo "FAIL checks"; cat "$work/build.log"; status=1
fi
cat "$work/checks.log" 2>/dev/null || true

control() {   # control <name> <expected failing check> <table> [cflags...]
    local name=$1 expect=$2 table=$3; shift 3
    if build "$work/$name" "$table" "$@" > "$work/$name.log" 2>&1; then
        "$work/$name" > "$work/$name.out" || true
        if grep -q "^FAIL $expect" "$work/$name.out"; then
            echo "PASS control $name fails: $expect"
        else
            echo "FAIL control $name passes \"$expect\""; status=1
        fi
    else
        echo "FAIL control $name does not compile"; cat "$work/$name.log"; status=1
    fi
}
# The binder that keeps the program's upper bits lets the forged integer select a3's reference.
control keep-upper "forged: the integer does not select" "$work/syscall-pointers.c" -DSUBLET_MUTANT_KEEP_UPPER
# A generator that counts argument positions from 1 gives read's buffer the wrong bit.
sed 's/mask = sum(1 << i for i, t/mask = sum(1 << (i + 1) for i, t/' "$kernel_dir/tools/gen-syscall-pointers.py" > "$work/gen-off-by-one.py"
if cmp -s "$work/gen-off-by-one.py" "$kernel_dir/tools/gen-syscall-pointers.py"; then
    echo "FAIL control generator-off-by-one: the mutation did not apply"; status=1
else
    python3 "$work/gen-off-by-one.py" "$linux" > "$work/off-by-one.c"
    control generator-off-by-one "table: read" "$work/off-by-one.c"
fi
exit $status
