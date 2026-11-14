#!/usr/bin/env bash
# The kernel's sublet-format.h and sublet-table.h against QEMU's table and capability code, on
# the host (checks.c), then mutants of sublet-format.h that must each fail.
#   tests/sublet-format/run.sh QEMU_TARGET_RISCV [KERNEL_SOURCES]
# QEMU_TARGET_RISCV is capstone-qemu's target/riscv at the revision the kernel runs on (it supplies
# cap_rev_table_abi.h too). KERNEL_SOURCES is a directory that holds sublet-format.h and
# sublet-table.h, such as arch/riscv/mm of a tree with the series applied. Without it the two
# files come from the series itself (patches/0011-riscv-mm-sublet-table.patch). Exit 0 only if
# the checks pass and every mutant fails.
set -euo pipefail
qemu=$(cd -- "${1:?capstone-qemu target/riscv}" && pwd)
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
status=0
if (($# >= 2)); then
    kernel=$(cd -- "$2" && pwd)
else
    # The new files of the series patch, written out: every line of a "--- /dev/null" file.
    kernel=$work/series
    python3 - "$here/../../patches/0011-riscv-mm-sublet-table.patch" "$kernel" <<'PYTHON'
import os, sys
patch, out = sys.argv[1], sys.argv[2]
os.makedirs(out)
lines, name = open(patch).read().split('\n'), None
for i, line in enumerate(lines):
    if line.startswith('+++ b/') and lines[i - 1] == '--- /dev/null':
        name = os.path.basename(line[6:])
        body = []
    elif name and line.startswith('@@'):
        continue
    elif name and line.startswith('+'):
        body.append(line[1:])
    elif name:
        open(os.path.join(out, name), 'w').write('\n'.join(body) + '\n')
        name = None
if name:
    open(os.path.join(out, name), 'w').write('\n'.join(body) + '\n')
PYTHON
fi

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

# mutant <name> <file> <old> <new>: the kernel's <file> (sublet-format.h or sublet-table.h) with
# <old> (found exactly once) replaced must fail, either at compile time or in a check.
mutant() {
    local dir="$work/$1"
    mkdir "$dir"
    cp "$kernel/sublet-format.h" "$kernel/sublet-table.h" "$dir/"
    python3 - "$kernel/$2" "$dir/$2" "$3" "$4" <<'PYTHON'
import sys
text = open(sys.argv[1]).read()
assert text.count(sys.argv[3]) == 1, sys.argv[3]
open(sys.argv[2], "w").write(text.replace(sys.argv[3], sys.argv[4]))
PYTHON
    verdict "$dir" "mutant $1"
}
verdict() {   # verdict <header dir> <name>: the build or a check must fail
    if build "$1" "$1/checks" > "$1/build.log" 2>&1; then
        if "$1/checks" > "$1/checks.log"; then
            echo "FAIL $2 passes the checks"; status=1
        else
            echo "PASS $2 fails $(grep -c '^FAIL' "$1/checks.log") checks"
        fi
    else
        echo "PASS $2 does not compile"
    fi
}
mutant depth-at-34 sublet-format.h '#define CV_REC_DEPTH(r) ((READ_ONCE((r)[0]) >> CAP_REV_SUBLET_DEPTH_SHIFT)' \
    '#define CV_REC_DEPTH(r) ((READ_ONCE((r)[0]) >> 34)'
mutant generation-at-95 sublet-format.h '((READ_ONCE((r)[1]) >> CAP_REV_SUBLET_GEN_SHIFT)' '((READ_ONCE((r)[1]) >> 31)'
mutant walk-4-9-9-8 sublet-format.h '#define CV_TABLE_ROOT_SHIFT 26' '#define CV_TABLE_ROOT_SHIFT 27'
mutant capability-slot-at-66 sublet-format.h '(((hi) >> CV_CAP_SLOT_SHIFT)' '(((hi) >> 2)'
mutant address-39-bits sublet-format.h '#define CV_CAP_ADDR_MASK ((1UL << CV_CAP_ADDR_BITS) - 1)' '#define CV_CAP_ADDR_MASK ((1UL << 39) - 1)'
mutant growth-word-at-56 sublet-format.h '#define CV_HDR_GROWTH 6' '#define CV_HDR_GROWTH 7'
# The walk inside the code path: the record lookup reads the middle index from slot[24:16].
mutant record-walk-middle-at-16 sublet-table.h \
    'entry = (const unsigned long *)SUBLET_VA(*entry) +
                ((slot >> (level == 0 ? CV_TABLE_MID_SHIFT : CV_TABLE_LEAF_SHIFT))' \
    'entry = (const unsigned long *)SUBLET_VA(*entry) +
                ((slot >> (level == 0 ? 16 : CV_TABLE_LEAF_SHIFT))'
# The growth count of the review finding: pages from the available count, without next.
mutant growth-ignores-reserved sublet-table.h \
    'need = (sublet_table_next(t) + required - sublet_table_free_count(t) + CV_TABLE_PAGE_RECORDS - 1) /
           CV_TABLE_PAGE_RECORDS;' \
    'need = (required - sublet_table_available(t) + CV_TABLE_PAGE_RECORDS - 1) / CV_TABLE_PAGE_RECORDS;'
# Control: QEMU's capability code at another id width must not pass.
mkdir "$work/qemu30" "$work/width-30"
cp "$qemu"/*.c "$qemu"/*.h "$work/qemu30/"
sed -i.bak 's/^#define CAP_ID_BITS 63$/#define CAP_ID_BITS 30/' "$work/qemu30/cap.h"
if grep -q '^#define CAP_ID_BITS 30$' "$work/qemu30/cap.h"; then
    cp "$kernel/sublet-format.h" "$kernel/sublet-table.h" "$work/width-30/"
    qemu_saved=$qemu; qemu=$work/qemu30
    verdict "$work/width-30" "control qemu-id-width-30"
    qemu=$qemu_saved
else
    echo "FAIL control qemu-id-width-30: cap.h has no CAP_ID_BITS 63 line"; status=1
fi
exit $status
