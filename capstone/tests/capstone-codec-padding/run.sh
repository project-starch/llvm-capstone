#!/usr/bin/env bash
# The compiler's representable-size helper against both codecs:
#   capstone/tests/capstone-codec-padding/run.sh QEMU_SRC MODEL [N]
# QEMU_SRC = a capstone-qemu tree (its target/riscv/cap_compress.c), MODEL = the RTL's bounds-model.py.
set -euo pipefail
qemu=${1:?capstone-qemu tree}; model=${2:?bounds-model.py}; n=${3:-20000}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); repo=$(cd -- "$here/../../.." && pwd)
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
src=$qemu/target/riscv
"${CC:-cc}" -O2 -c -I"$src" "$src/cap_compress.c" -o "$work/cap_compress.o"
"${CXX:-c++}" -O2 -Wall -std=c++17 -I"$src" -I"$repo/llvm/lib/Target/Capstone/MCTargetDesc" "$here/rep-size-sweep.cpp" "$work/cap_compress.o" -o "$work/sweep"
rc=0; "$work/sweep" "$n" > "$work/cases.txt" || rc=$?
python3 "$here/rep-size-sweep.py" "$model" < "$work/cases.txt" || rc=$?
exit $rc
