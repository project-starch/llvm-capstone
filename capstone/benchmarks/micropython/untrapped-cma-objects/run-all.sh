#!/usr/bin/env bash
# Re-run every row in README.md and report one line each.
#
# Ordering is deliberate: the C-glue rows first because they are the cheapest and
# each is a single boot, then the shared script image, then the rows that need a
# 2024 worktree. A row that fails does not stop the others -- the point is a table,
# not a bisection.
#
#   ./run-all.sh            all of them, roughly 20 boots
#   ./run-all.sh glue       only the C reconstructions (T07, T16, T29)
#   ./run-all.sh scripts    only the script-driven rows (T09, T10, T13)
#   ./run-all.sh spatial    only MPY-S05
#   ./run-all.sh backport   only the 2024 worktree rows (T02, T05)
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
M=capstone/benchmarks/micropython
MODE=${1:-all}

declare -a NAME PATHS
add() { NAME+=("$1"); PATHS+=("$2"); }
[[ $MODE == all || $MODE == glue ]] && {
  add MPY-T07 "$M/temporal-corpus/cases/MPY-T07_lexer-source-name-uaf/run.sh"
  add MPY-T16 "$M/temporal-corpus/cases/MPY-T16_deinit-after-gc-sweep-all/run.sh"
  add MPY-T29 "$M/temporal-corpus/cases/MPY-T29_gc-free-assert-invalid-block/run.sh"; }
[[ $MODE == all || $MODE == scripts ]] && {
  add MPY-T09 "$M/temporal-corpus/cases/MPY-T09_bytearray-resize-stale-view/run.sh"
  add MPY-T10 "$M/temporal-corpus/cases/MPY-T10_array-resize-stale-view/run.sh"
  add MPY-T13 "$M/temporal-corpus/cases/MPY-T13_write-callback-grows-buffer/run.sh"; }
[[ $MODE == all || $MODE == spatial ]] && {
  add MPY-S05 "$M/spatial-corpus/cases/MPY-S05_array-append-after-failed-grow/run.sh"; }
[[ $MODE == all || $MODE == backport ]] && {
  add MPY-T02 "$M/temporal-corpus/cases/MPY-T02_objarray-bytes-self-copy-uaf/run.sh"
  add MPY-T05 "$M/temporal-corpus/cases/MPY-T05_objarray-line509-uaf/run.sh"; }

declare -a VERD
worst=0
for i in "${!NAME[@]}"; do
  echo "== ${NAME[$i]}"
  out=/tmp/capstone/untrapped-${NAME[$i]}.out
  bash "${PATHS[$i]}" >"$out" 2>&1
  rc=$?
  line=$(grep -a "VERDICT" "$out" | tail -1 | sed 's/^ *//')
  case $rc in
    0)  VERD+=("as recorded    ${line#VERDICT: }") ;;
    75) VERD+=("VOID (infra)   no verdict; see $out"); [[ $worst -lt 1 ]] && worst=1 ;;
    *)  VERD+=("DIFFERS        ${line:-see $out}"); worst=2 ;;
  esac
  echo "   ${VERD[$i]}"
done

echo
echo "== summary =="
for i in "${!NAME[@]}"; do printf "  %-9s %s\n" "${NAME[$i]}" "${VERD[$i]}"; done
echo
case $worst in
  0) echo "All rows reproduce as recorded." ;;
  1) echo "Some rows produced NO verdict (infra). Re-run those; do not read them as passes." ;;
  2) echo "At least one row DIFFERS from its record. That is a finding, not a flake." ;;
esac
exit $worst
