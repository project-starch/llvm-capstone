#!/usr/bin/env bash
# Verify every corpus row still reproduces its defect on stock toolchains.
#
#   ./reproduce.sh              # run all 15 rows (assumes already built)
#   ./reproduce.sh --build      # build first, then run
#   ./reproduce.sh --rows 7 10  # only these
#
# This is the GROUND TRUTH column. The shims in ../cheri/shims/ are validated
# against these rows by check_shim_fidelity.py, and both measurement columns
# run those shims -- so if a row here stops reproducing, every number
# downstream is measuring something that no longer matches a real defect.
# It was the only column without a driver until 2026-08-02; the other two have
# had one since they existed.
#
# Each row's run.sh already exits non-zero when its expected violation does not
# appear, so this script's job is to run them all and refuse to summarise a
# failure away.
#
# Row 7 is deliberately NOT ASan-based: its oracle is a plain-build SIGSEGV,
# because ASan and valgrind both mask that defect by replacing realloc with an
# always-moving implementation. Its run.sh knows this; do not "fix" it by
# adding a sanitizer.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

BUILD=0
ROWS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --build) BUILD=1; shift ;;
    --rows)  shift; while [ $# -gt 0 ] && [[ $1 != --* ]]; do ROWS+=("$1"); shift; done ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ ${#ROWS[@]} -eq 0 ]; then
  for d in "$HERE"/[0-9] "$HERE"/[0-9][0-9]; do
    [ -d "$d" ] && ROWS+=("$(basename "$d")")
  done
fi
# numeric order, so the output reads like the paper table
IFS=$'\n' ROWS=($(printf '%s\n' "${ROWS[@]}" | sort -n)); unset IFS

pass=0; fail=0; skip=0
printf '%-5s %-10s %s\n' ROW RESULT DETAIL
printf '%.0s-' {1..64}; echo

for r in "${ROWS[@]}"; do
  d="$HERE/$r"
  [ -x "$d/run.sh" ] || { printf '%-5s %-10s %s\n' "$r" SKIP "no run.sh"; skip=$((skip+1)); continue; }

  if [ "$BUILD" -eq 1 ]; then
    if ! timeout 1800 "$d/build.sh" >"$d/build.log" 2>&1; then
      printf '%-5s %-10s %s\n' "$r" BUILD-FAIL "see $r/build.log"; fail=$((fail+1)); continue
    fi
  fi

  out=$(timeout 600 "$d/run.sh" 2>&1); rc=$?
  if [ "$rc" -eq 0 ]; then
    printf '%-5s %-10s %s\n' "$r" PASS "defect reproduced"
    pass=$((pass+1))
  else
    # Not summarised away: print enough to act on without opening the log.
    printf '%-5s %-10s %s\n' "$r" FAIL "rc=$rc  $(printf '%s' "$out" | tail -1 | cut -c1-40)"
    fail=$((fail+1))
  fi
done

printf '%.0s-' {1..64}; echo
echo "pass=$pass fail=$fail skipped=$skip  (of ${#ROWS[@]})"
[ "$fail" -eq 0 ] && [ "$skip" -eq 0 ]
