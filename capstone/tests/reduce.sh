#!/usr/bin/env bash
# Shrink a failing .ll to the smallest input that still fails the same way.
#
#   reduce.sh <input.ll> "<substring of the message>" [extra llc args...]
#
# The point is the tier drop. A backend failure found by a suite costs minutes to
# over an hour per look; the same failure as a reduced .ll costs one llc run --
# measured at 0.09 s on this tree. Reduce once, iterate hundreds of times, and the
# result is a lit test that keeps the bug from coming back.
#
# The message substring is what makes this honest: llvm-reduce will happily shrink
# to some OTHER failure if you only ask "does it still fail". Name the assert.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/capstone-test-env.sh"

[ $# -ge 2 ] || { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
IN=$1; MSG=$2; shift 2
[ -f "$IN" ] || { echo "reduce.sh: no such file: $IN" >&2; exit 2; }

BUILD=${CAPSTONE_LLVM_BUILD_DIR:-$CAPSTONE_REPO_ROOT/llvm/cmake-build-debug}
LLC=$BUILD/bin/llc
RED=$BUILD/bin/llvm-reduce
for t in "$LLC" "$RED"; do
  [ -x "$t" ] || { echo "reduce.sh: missing $t -- build it first" >&2; exit 2; }
done
LLC_ARGS=("-mtriple=capstone64" "$@")

WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
TEST=$WORK/interesting.sh
cat > "$TEST" <<INNER
#!/bin/bash
# Interesting == fails with THIS message. No pipe: grep's status must not be
# mistaken for llc's, and a closed pipe must not decide the answer.
out=\$("$LLC" ${LLC_ARGS[*]} -o /dev/null "\$1" 2>&1) || true
case "\$out" in *"$MSG"*) exit 0 ;; esac
exit 1
INNER
chmod +x "$TEST"

# Before reducing anything: does the ORIGINAL actually fail this way? Reducing an
# input that never reproduced produces a small file that proves nothing.
if ! "$TEST" "$IN"; then
  echo "reduce.sh: the input does NOT fail with '$MSG'" >&2
  echo "  ran: $LLC ${LLC_ARGS[*]} -o /dev/null $IN" >&2
  exit 1
fi
before=$(wc -l < "$IN")
echo "reduce.sh: input reproduces ($before lines); reducing..."

OUT=${REDUCE_OUT:-${IN%.ll}-reduced.ll}
"$RED" --test "$TEST" -o "$OUT" "$IN" >/dev/null 2>&1 || true
[ -s "$OUT" ] || { echo "reduce.sh: llvm-reduce produced nothing" >&2; exit 1; }

# And does the RESULT still fail the same way? llvm-reduce is not asked to be
# trustworthy here; it is asked to be checked.
if ! "$TEST" "$OUT"; then
  echo "reduce.sh: the reduced file no longer reproduces -- discarding it" >&2
  rm -f "$OUT"; exit 1
fi
after=$(wc -l < "$OUT")
echo "reduce.sh: $before -> $after lines, still fails with '$MSG'"
echo "  $OUT"
echo "  reproduce: $LLC ${LLC_ARGS[*]} -o /dev/null $OUT"
