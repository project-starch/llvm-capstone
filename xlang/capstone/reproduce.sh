#!/usr/bin/env bash
# Reproduce the xlang Capstone column from scratch and CHECK the result.
#
#   ./reproduce.sh              # clean rebuild of all 14 rows, run all 28, verify
#   ./reproduce.sh --keep       # reuse existing binaries (faster, weaker)
#   ./reproduce.sh --rows a b   # only these rows
#
# This is a CHECK, not a runner: it compares a fresh measurement against
# expected-results.tsv and EXITS NON-ZERO on any difference. A column nobody can
# re-measure is a claim, not a result -- the CHERI column was rebuilt from an
# empty CHERI_ROOT and re-measured byte-identical, and this exists so this
# column can be held to the same bar.
#
# Runtime: ~28 QEMU boots, roughly an hour. One boot per run, never batched --
# a faulting domain halts QEMU and every row after it in that boot silently
# never runs, which reads as "no result" rather than as an error.
#
# Boot flakes are real: 7 of 28 on the first pass, and on the clean-rebuild pass
# one row hung in the guest bootloader three times running, before any domain
# was loaded.
#
# WHAT MAY BE RETRIED, AND WHAT MAY NOT. A run is retried ONLY when it produced
# NO EVIDENCE at all (NORESULT) -- the guest never reached the shell, so no
# domain ran and nothing was measured. A run that produces a result is NEVER
# retried, even when that result disagrees with expected-results.tsv. Retrying
# absent evidence is legitimate; retrying until the answer comes out right is
# not, and the difference is the only thing keeping this a check.
#
# A run that still has no evidence after $ATTEMPTS tries is reported NORESULT
# and FAILS the check. It is not a MISS and it is not a FAULT.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${CAPSTONE_REPO_ROOT:=$(cd "$HERE/../.." && pwd)}"
export CAPSTONE_REPO_ROOT
# shellcheck source=/dev/null
source "$CAPSTONE_REPO_ROOT/capstone/tests/capstone-test-env.sh"

ATTEMPTS=${ATTEMPTS:-5}
SHARE=${SHARE:-$CAPSTONE_TMP_ROOT/xlang-capstone-repro}
SMOKE="$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py"
EXPECTED="$HERE/expected-results.tsv"
ACTUAL="$SHARE/actual-results.tsv"
KEEP=0; SEL=()

while [ $# -gt 0 ]; do
  case "$1" in
    --keep) KEEP=1; shift ;;
    --rows) shift; while [ $# -gt 0 ] && [[ $1 != --* ]]; do SEL+=("$1"); shift; done ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -f "$EXPECTED" ] || { echo "missing $EXPECTED" >&2; exit 2; }

rows=("${SEL[@]:-}")
if [ -z "${rows[0]:-}" ]; then
  rows=()
  while IFS=$'\t' read -r _ row _; do rows+=("$row"); done \
    < <(grep -v '^#' "$EXPECTED" | tail -n +2)
fi

if [ "$KEEP" -eq 0 ]; then
  echo "== clean build =="
  rm -rf "$SHARE"
fi
mkdir -p "$SHARE"

# --- build --------------------------------------------------------------------
echo "== building ${#rows[@]} rows x 2 variants =="
for row in "${rows[@]}"; do
  OUT_DIR="$SHARE" ROW="$row" bash "$HERE/build-xlang-capstone.sh" >/dev/null 2>&1 \
    || { echo "BUILD FAILED: $row (revoke)" >&2; exit 1; }
  OUT_DIR="$SHARE" ROW="$row" NO_REVOKE=1 bash "$HERE/build-xlang-capstone.sh" >/dev/null 2>&1 \
    || { echo "BUILD FAILED: $row (control)" >&2; exit 1; }
done

echo "== building host =="
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
"$GUEST_CC" -O2 -I"$HERE" -I"$CAPSTONE_REPO_ROOT/capstone/benchmarks/sqlite" \
  -o "$SHARE/xlang_host.user" "$HERE/xlang_shim_host.c" "$LIBCAPSTONE" \
  || { echo "HOST BUILD FAILED" >&2; exit 1; }

# --- run ----------------------------------------------------------------------
classify() { # $1 = log
  local log="$1"
  grep -qa 'XLANG-INVALID'                        "$log" && { echo INVALID;  return; }
  grep -qa 'survived'                             "$log" && { echo MISS;     return; }
  grep -qa 'domain halted by capability fault'    "$log" && { echo FAULT;    return; }
  grep -qa "Assertion \`rs1_v->tag' failed"       "$log" && { echo FAULT;    return; }
  echo NORESULT
}

printf 'row\tvariant\toutcome\n' > "$ACTUAL"
echo "== running $(( ${#rows[@]} * 2 )) boots =="
for row in "${rows[@]}"; do
  for variant in revoke control; do
    if [ "$variant" = control ]; then dom="xlang_${row}_norevoke"; else dom="xlang_${row}"; fi
    log="$SHARE/repro_${row}_${variant}.log"
    outcome=NORESULT
    for _ in $(seq 1 "$ATTEMPTS"); do
      timeout 520 python3 "$SMOKE" --share-dir "$SHARE" --log-file "$log" \
        --timeout-multiplier 8 \
        --guest-command "cp /mnt/host/xlang_host.user /tmp/h && chmod 0755 /tmp/h && /tmp/h /mnt/host/$dom.dom" \
        --success-marker '__never__' >/dev/null 2>&1
      outcome=$(classify "$log")
      [ "$outcome" != NORESULT ] && break
    done
    printf '%s\t%s\t%s\n' "$row" "$variant" "$outcome" >> "$ACTUAL"
    printf '  %-28s %-8s %s\n' "$row" "$variant" "$outcome"
  done
done

# --- verify -------------------------------------------------------------------
echo
echo "== verifying against expected-results.tsv =="
python3 - "$EXPECTED" "$ACTUAL" <<'PY'
import sys, collections
exp_path, act_path = sys.argv[1], sys.argv[2]

expected = {}
for line in open(exp_path):
    if line.startswith('#') or line.startswith('key\t'): continue
    key, row, rev, ctl, mech = line.rstrip('\n').split('\t')
    expected[row] = (rev, ctl, mech)

actual = collections.defaultdict(dict)
for i, line in enumerate(open(act_path)):
    if i == 0: continue
    row, variant, outcome = line.rstrip('\n').split('\t')
    actual[row][variant] = outcome

bad = []
for row, (rev, ctl, mech) in expected.items():
    if row not in actual:
        continue                                  # not selected this run
    a_rev = actual[row].get('revoke', 'NORESULT')
    a_ctl = actual[row].get('control', 'NORESULT')
    if (a_rev, a_ctl) != (rev, ctl):
        bad.append(f"  {row}: expected {rev}/{ctl}, got {a_rev}/{a_ctl}")
    # A catch attributed to revocation REQUIRES its control to miss. Checked
    # separately from equality so a both-FAULT result is never read as a
    # temporal catch just because the fault column happens to match.
    elif mech == 'revocation' and a_ctl != 'MISS':
        bad.append(f"  {row}: control did not MISS, so the catch is not "
                   f"attributable to the revoke")

checked = len([r for r in expected if r in actual])
if bad:
    print(f"REPRODUCTION FAILED  ({len(bad)} of {checked} rows differ)")
    print('\n'.join(bad))
    sys.exit(1)
print(f"REPRODUCED  {checked}/{checked} rows identical to expected-results.tsv")
PY
status=$?
echo
echo "logs: $SHARE"
exit $status
