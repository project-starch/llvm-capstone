#!/bin/bash
# Build case.c against a native mruby build of the pin and run every case in a plain
# and an ASan arm. -O0 deliberately: at -O1 GCC eliminates a store nothing reads and
# the row goes silent for a reason that has nothing to do with the defect.
#
#   ./build-and-run.sh <an mruby build tree> [depth]
#
# The tree is a checkout built with probe/probe_config.rb, so it has build/host and
# build/asan with libmruby.a, and include/ plus the generated presym header under
# build/<arm>/include.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TREE=${1:?usage: build-and-run.sh <mruby build tree> [depth]}
DEPTH=${2:-30}
for arm in host asan; do
  L=$TREE/build/$arm/lib/libmruby.a
  [[ -f $L ]] || { echo "no $L -- build the tree with probe_config.rb first" >&2; exit 2; }
  GEN=$TREE/build/$arm/include; [[ -d $GEN ]] || GEN=$TREE/build/stress/include
  F=(); [[ $arm == asan ]] && F=(-fsanitize=address -fno-omit-frame-pointer)
  gcc -g -O0 "${F[@]}" -I"$TREE/include" -I"$GEN" -o "$HERE/case-$arm" "$HERE/case.c" "$L" -lm || exit 2
done
printf '\n%-5s %-30s %s\n' case plain "ASan"
printf '%-5s %-30s %s\n' ---- ----- ----
for w in 0 1 2 3 4 5; do
  p=$(cd "$HERE" && timeout 400 ./case-host $w grow.rb "$DEPTH" 2>&1 | tail -1)
  m=$(grep -oE 'stack moved|STACK DID NOT MOVE' <<<"$p")
  a=$(cd "$HERE" && timeout 500 ./case-asan $w grow.rb "$DEPTH" 2>&1 \
      | grep -oE 'heap-use-after-free|(WRITE|READ) of size [0-9]+' | head -2 | tr '\n' ' ')
  printf '%-5s %-30s %s\n' "$w" "${m:-?}" "${a:-<silent>}"
done
cat <<'EOF'

STACK DID NOT MOVE means the run is INVALID, not a MISS: nothing was stale, so the
case measured nothing. Raise the depth. A WRITE attribution is the defect's own store
landing in the freed block; a READ attribution is the case's read-back, which proves
the held pointer is dangling but leaves the store's fate to the compiler's evaluation
order -- which is c52faebb7's second stated reason and not something a case can fix.
EOF
