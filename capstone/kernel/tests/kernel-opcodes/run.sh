#!/usr/bin/env bash
# The Sublet instructions a built kernel contains, by encoding, against an allowlist. The kernel
# may execute LDC (funct3 3), STC (funct3 4) and, under funct3 1, LCC (funct7 0x04), CINCOFFSET
# (0x0c), CSMINT (0x50), CDERIVE (0x51) and CREVOKE (0x52). Every other encoding of the custom
# opcode 0x5b fails, among them CSSUPERVISE (0x22), CSSUPERVISOR_GC (0x23), CSRUNV (0x24), the
# checked copy (0x25, 0x26) and CSRETIRE (0x53), which the 63-bit format makes illegal.
#   tests/kernel-opcodes/run.sh CROSS_PREFIX OBJECT...
# CROSS_PREFIX is the RISC-V tool prefix (riscv64-...-). Exit 0 only if every object holds only
# allowed encodings, each control object with one deleted encoding is caught, and the control
# object with every allowed encoding passes.
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
outside() {   # outside OBJECT: the encodings it holds that the allowlist does not name
    census "$1" | awk '!(($1 == 3 || $1 == 4) || ($1 == 1 && ($2 == 4 || $2 == 12 || $2 == 80 || $2 == 81 || $2 == 82)))'
}
assemble() {   # assemble NAME LINE...: an object from the given instruction lines
    local name=$1; shift
    printf '.text\n' > "$work/$name.S"
    printf '%s\n' "$@" >> "$work/$name.S"
    "${cross}gcc" -c "$work/$name.S" -o "$work/$name.o"
}

status=0
for f7 in 0x22 0x23 0x24 0x25 0x26 0x53; do
    assemble "deleted-$f7" ".insn r 0x5b, 1, $f7, x10, x11, x12"
    if [[ -n $(outside "$work/deleted-$f7.o") ]]; then
        echo "PASS control: funct7 $f7 is caught"
    else
        echo "FAIL control: the census misses funct7 $f7"; status=1
    fi
done
assemble allowed '.insn i 0x5b, 3, x10, x11, 0' '.insn s 0x5b, 4, x10, 0(x11)' \
    '.insn r 0x5b, 1, 4, x10, x11, x1' '.insn r 0x5b, 1, 0x0c, x10, x11, x12' \
    '.insn r 0x5b, 1, 0x50, x10, x11, x12' '.insn r 0x5b, 1, 0x51, x10, x11, x12' \
    '.insn r 0x5b, 1, 0x52, zero, x11, zero'
if [[ -z $(outside "$work/allowed.o") && $(census "$work/allowed.o" | wc -l) -eq 7 ]]; then
    echo "PASS control: the seven allowed encodings pass"
else
    echo "FAIL control: the allowed encodings are refused or not seen"; status=1
fi
((status == 0)) || exit $status

for object in "$@"; do
    bad=$(outside "$object")
    if [[ -n $bad ]]; then
        echo "FAIL $object holds encodings outside the allowlist (funct3 funct7 count):"; echo "$bad"; status=1
    else
        echo "PASS $object: $(census "$object" | awk '{printf "%s%s x%s", sep, ($1 == 1 ? sprintf("funct7 0x%x", $2) : "funct3 " $1), $3; sep=", "}')"
    fi
done
exit $status
