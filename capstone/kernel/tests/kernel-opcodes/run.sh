#!/usr/bin/env bash
# The Sublet instructions a built kernel contains, by encoding. The kernel may execute LDC,
# STC, LCC, CINCOFFSET, CSMINT, CDERIVE and CREVOKE, and no instruction of a deleted route:
# CSRUNV (funct7 0x24), the checked copy (0x25, 0x26) or CSRETIRE (0x53), which the 63-bit
# format makes illegal.
#   tests/kernel-opcodes/run.sh CROSS_PREFIX OBJECT...
# CROSS_PREFIX is the RISC-V tool prefix (riscv64-...-). Exit 0 only if the objects hold no
# forbidden encoding and the control object, assembled here with one CSRUNV, is caught.
set -euo pipefail
cross=${1:?RISC-V tool prefix}; shift
(($#)) || { echo "no objects" >&2; exit 2; }
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT

census() {   # census OBJECT: one line for each encoding of a Sublet instruction, "funct3 funct7 count"
    "${cross}objdump" -d "$1" | awk '$2 ~ /^[0-9a-f]{8}$/ {print $2}' | python3 -c '
import sys, collections
seen = collections.Counter()
for word in sys.stdin:
    v = int(word, 16)
    if v & 0x7f == 0x5b:
        f3 = (v >> 12) & 7
        seen[(f3, (v >> 25) if f3 == 1 else -1)] += 1
for (f3, f7), n in sorted(seen.items()):
    print(f3, f7, n)'
}
forbidden() {   # forbidden OBJECT: the forbidden encodings it holds
    census "$1" | awk '$1 == 1 && ($2 == 36 || $2 == 37 || $2 == 38 || $2 == 83)'
}

printf '.text\n.insn r 0x5b, 1, 0x24, x0, x10, x11\n' > "$work/control.S"
"${cross}gcc" -c "$work/control.S" -o "$work/control.o"
if [[ -n $(forbidden "$work/control.o") ]]; then
    echo "PASS control: a CSRUNV encoding is caught"
else
    echo "FAIL control: the census misses CSRUNV"; exit 1
fi
status=0
for object in "$@"; do
    bad=$(forbidden "$object")
    if [[ -n $bad ]]; then
        echo "FAIL $object holds a deleted route's encoding:"; echo "$bad"; status=1
    else
        echo "PASS $object: $(census "$object" | awk '{printf "%s%s x%s", sep, ($1 == 1 ? sprintf("funct7 0x%x", $2) : "funct3 " $1), $3; sep=", "}')"
    fi
done
exit $status
