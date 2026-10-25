#!/usr/bin/env bash
# R-60 window slack, the claim performCIncOffsetCombine's comment rests on: for representable bounds [b,e), every
# cursor from b-2048 to e+2048 keeps its tag under QEMU's R-60 predicate (capstone_cursor_representable).
#   run.sh <capstone-qemu tree carrying R-60, e.g. 02f03540d7> [regions]
# Builds win.c against that tree's target/riscv/cap_compress.c on the host. Then the POSITIVE CONTROL: the same check
# demanding 64 KiB of slack must report short windows, or the check cannot fire. Exit 0 = claim holds and control
# fired; 1 = a region falls short; 2 = could not check (build failed, or the control did not fire).
set -u
Q=${1:?capstone-qemu tree}; N=${2:-300000}; H=$(cd "$(dirname "$0")" && pwd); T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
R=$Q/target/riscv
grep -q 'capstone_cursor_representable' "$R/cap_compress.c" || { echo "no R-60 predicate in $R/cap_compress.c"; exit 2; }
gcc -O2 -I"$R" -o "$T/win" "$H/win.c" "$R/cap_compress.c" || exit 2
sed 's/2048/65536/g; s/2049/65537/g' "$H/win.c" > "$T/winpc.c"
gcc -O2 -I"$R" -o "$T/winpc" "$T/winpc.c" "$R/cap_compress.c" || exit 2
"$T/win" "$N"; rc=$?
"$T/winpc" 20000 > "$T/pc.out"; pc=$?
tail -1 "$T/pc.out"
[ "$pc" = 1 ] || { echo "CONTROL DID NOT FIRE: a 64 KiB demand found no short window"; exit 2; }
exit $rc
