#!/usr/bin/env bash
# The kernel's sublet-format.h and sublet-table.h against QEMU's table and capability code, on
# the host (checks.c), then mutants of sublet-format.h that must each fail.
#   tests/sublet-format/run.sh QEMU_TARGET_RISCV KERNEL_SOURCES
# QEMU_TARGET_RISCV is capstone-qemu's target/riscv at the revision the kernel runs on (it supplies
# cap_rev_table_abi.h too). KERNEL_SOURCES is the directory that holds sublet-format.h and
# sublet-table.h. Exit 0 only if the checks pass and every mutant fails.
set -euo pipefail
qemu=$(cd -- "${1:?capstone-qemu target/riscv}" && pwd)
kernel=$(cd -- "${2:?directory with sublet-format.h and sublet-table.h}" && pwd)
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
status=0

build() {   # build <header dir> <output>
    ${CC:-cc} -std=gnu11 -O2 -Wall -Wextra -Werror -Wno-unused-function -I"$1" -I"$qemu" "$here/checks.c" \
        "$qemu/cap.c" "$qemu/cap_compress.c" "$qemu/cap_rev_tree.c" "$qemu/cap_rev_table.c" -o "$2" \
        $(pkg-config --cflags --libs glib-2.0)
}

mkdir "$work/kernel"
cp "$kernel/sublet-format.h" "$kernel/sublet-table.h" "$work/kernel/"
if build "$work/kernel" "$work/checks" > "$work/build.log" 2>&1 && "$work/checks" > "$work/checks.log"; then
    echo "PASS checks ($(grep -c '^PASS' "$work/checks.log") of them)"
else
    echo "FAIL checks"; cat "$work/build.log"; status=1
fi
cat "$work/checks.log" 2>/dev/null || true

# mutant <name> <old> <new>: sublet-format.h with <old> (found exactly once) replaced must fail,
# either at compile time or in a check.
mutant() {
    local dir="$work/$1"
    mkdir "$dir"
    cp "$kernel/sublet-table.h" "$dir/"
    python3 - "$kernel/sublet-format.h" "$dir/sublet-format.h" "$2" "$3" <<'PYTHON'
import sys
text = open(sys.argv[1]).read()
assert text.count(sys.argv[3]) == 1, sys.argv[3]
open(sys.argv[2], "w").write(text.replace(sys.argv[3], sys.argv[4]))
PYTHON
    if build "$dir" "$dir/checks" > "$dir/build.log" 2>&1; then
        if "$dir/checks" > "$dir/checks.log"; then
            echo "FAIL mutant $1 passes the checks"; status=1
        else
            echo "PASS mutant $1 fails $(grep -c '^FAIL' "$dir/checks.log") checks"
        fi
    else
        echo "PASS mutant $1 does not compile"
    fi
}
mutant depth-at-34 '& CAP_REV_SUBLET_FIELD_MASK)
#define CV_REC_PREV' '>> 1 & CAP_REV_SUBLET_FIELD_MASK)
#define CV_REC_PREV'
mutant generation-at-95 '((READ_ONCE((r)[1]) >> CAP_REV_SUBLET_GEN_SHIFT)' '((READ_ONCE((r)[1]) >> 31)'
mutant walk-4-9-9-8 '#define CV_TABLE_ROOT_SHIFT 26' '#define CV_TABLE_ROOT_SHIFT 27'
mutant capability-slot-at-66 '(((hi) >> CV_CAP_SLOT_SHIFT)' '(((hi) >> 2)'
mutant address-39-bits '#define CV_CAP_ADDR_MASK ((1UL << CV_CAP_ADDR_BITS) - 1)' '#define CV_CAP_ADDR_MASK ((1UL << 39) - 1)'
mutant growth-word-at-56 '#define CV_HDR_GROWTH 6' '#define CV_HDR_GROWTH 7'
exit $status
