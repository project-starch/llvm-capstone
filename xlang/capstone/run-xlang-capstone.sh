#!/usr/bin/env bash
# Run the xlang Capstone column: every row, both variants, one boot each.
#
#   ./run-xlang-capstone.sh                 # all 14 rows
#   ./run-xlang-capstone.sh mruby_range_uaf # one row
#
# ONE BOOT PER RUN, deliberately. Batching several rows into a single guest
# command does not work: a row whose domain faults halts QEMU, and every row
# after it in that boot silently never runs. Observed 2026-08-01 -- the batch
# reported one result and stopped, which reads as "13 rows produced nothing"
# rather than as an error.
#
# Predictions live in rows.tsv and were committed BEFORE any of this ran.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${CAPSTONE_REPO_ROOT:=$(cd "$HERE/../.." && pwd)}"
export CAPSTONE_REPO_ROOT
# shellcheck source=/dev/null
source "$CAPSTONE_REPO_ROOT/capstone/tests/capstone-test-env.sh"

SHARE=${SHARE:-$CAPSTONE_TMP_ROOT/xlang-capstone}
SMOKE="$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py"
RESULTS="$SHARE/results.tsv"

rows=("$@")
if [ ${#rows[@]} -eq 0 ]; then
  rows=()
  for s in "$HERE"/../cheri/shims/*.c; do rows+=("$(basename "$s" .c)"); done
fi

printf 'row\tvariant\toutcome\tdetail\n' > "$RESULTS"

classify() { # $1 = log file
  local log="$1"
  if grep -qa 'XLANG-INVALID' "$log"; then echo "INVALID	harness could not measure"; return; fi
  if grep -qa 'use-after-free-survived\|overflow-survived\|survived' "$log"; then
    echo "MISS	stale access completed"; return
  fi
  if grep -qa 'domain halted by capability fault' "$log"; then
    echo "FAULT	$(grep -oa 'cause = [0-9]*' "$log" | tail -1)"; return
  fi
  if grep -qa "Assertion \`rs1_v->tag' failed" "$log"; then
    echo "FAULT	qemu assert on untagged operand (emulator gap, see 01-08-2026 note)"; return
  fi
  if grep -qa 'Print = ' "$log"; then echo "UNEXPECTED	debug-print reached; investigate"; return; fi
  echo "NORESULT	no marker (boot flake or timeout)"
}

for row in "${rows[@]}"; do
  for variant in revoke control; do
    if [ "$variant" = control ]; then dom="xlang_${row}_norevoke"; else dom="xlang_${row}"; fi
    log="$SHARE/run_${row}_${variant}.log"
    timeout 520 python3 "$SMOKE" \
      --share-dir "$SHARE" --log-file "$log" --timeout-multiplier 8 \
      --guest-command "cp /mnt/host/xlang_host.user /tmp/h && chmod 0755 /tmp/h && /tmp/h /mnt/host/$dom.dom" \
      --success-marker '__never__' >/dev/null 2>&1
    printf '%s\t%s\t%s\n' "$row" "$variant" "$(classify "$log")" >> "$RESULTS"
    printf '%-28s %-8s %s\n' "$row" "$variant" "$(classify "$log")"
  done
done

echo
echo "results: $RESULTS"
