#!/usr/bin/env bash
# Is any generated core/*.anvil.sv OLDER than the core/anvil_build/*.anvil it came from?
#
# WHY. A worktree made by scripts/mk-wt.sh generates the anvil output AT CREATION TIME. Apply a
# source patch afterwards and the generated .sv is never refreshed, so the source is right, the
# commit is right, and every local gate -- lint, the directed sweep, a cost measurement -- silently
# tests the PRE-EDIT design. It cost an afternoon on 2026-09-16: a 95-test sweep reported "zero
# changes", which read as "the change is inert" and actually meant "the change is absent".
#
# The signature is trivial and needs no knowledge of what changed: a generated file older than its
# source. It catches the case whether the edit came from a patch, an editor or a cherry-pick.
#
# run-synthesis.sh does `make clean && make` before Vivado, so staleness is harmless THERE. It is
# local gates that are fooled, which is why this belongs in front of them.
#
#   usage: bash capstone/tests/anvil-staleness-check.sh <worktree>   # exit 1 = STALE
set -uo pipefail
WT="${1:-.}"
GEN="$WT/core"; SRC="$WT/core/anvil_build"
[ -d "$SRC" ] || { echo "anvil-staleness: no $SRC -- wrong path?" >&2; exit 2; }

n=0; stale=0
for s in "$SRC"/*.anvil; do
  [ -e "$s" ] || continue
  g="$GEN/$(basename "$s").sv"
  n=$((n+1))
  if [ ! -f "$g" ]; then
    echo "  MISSING  $(basename "$g") -- never generated"; stale=$((stale+1))
  elif [ ! -s "$g" ]; then
    # A FAILED anvil leaves a ZERO-BYTE target with a fresh mtime: the Makefile redirects anvil's
    # stdout into the .sv before anvil runs, so a compile error produces an empty file that is
    # "newer than its source" and that make then considers up to date. Found 2026-09-16 when a
    # borrow-check failure hid behind an OK from this very gate. Empty is not generated.
    echo "  EMPTY    $(basename "$g") is ZERO BYTES -- anvil FAILED; make will not retry until clean"; stale=$((stale+1))
  elif [ "$s" -nt "$g" ]; then
    echo "  STALE    $(basename "$g") is OLDER than $(basename "$s")"; stale=$((stale+1))
  fi
done
[ "$n" -gt 0 ] || { echo "anvil-staleness: found no .anvil sources -- checked nothing, which is not a pass" >&2; exit 2; }

if [ "$stale" -gt 0 ]; then
  echo "anvil-staleness: STALE ($stale of $n). Run: make clean && make -C core/anvil_build (in the container)." >&2
  exit 1
fi
echo "anvil-staleness: OK ($n generated files all newer than their sources)"
