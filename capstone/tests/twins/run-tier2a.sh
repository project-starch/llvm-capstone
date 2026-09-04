#!/usr/bin/env bash
# Tier 2a of the compiler-validation plan in one sitting: every execution suite at
# -O0 and -O2 (and the SLT value oracle at -O1 too), then the agreement gates.
#
#   run-tier2a.sh [suite ...]        default: slt rv8 coremark beebs
#
# Cheap and informative first, BEEBS (82 domains per level) last.  Every boot takes the
# nightly QEMU lock, so this can run alongside nothing else on QEMU.  Continues past a
# failing suite -- a failure IS the result -- and ends with a summary.  Result lines
# accumulate in results/<date>.tsv.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"
SQL="$CAPSTONE_REPO_ROOT/capstone/benchmarks/sqlite"
TSV=${TWINS_RESULTS:-$SCRIPT_DIR/results/$(date +%Y-%m-%d).tsv}
export SLT_TWIN_RESULTS="$TSV"
QID="qemu=$(sha256sum "$CAPSTONE_QEMU_BINARY" | cut -c1-12)@$(date -r "$CAPSTONE_QEMU_BINARY" +%Y-%m-%dT%H:%M)"
SUITES=${*:-slt rv8 coremark beebs}
declare -A RC

for s in $SUITES; do
  case "$s" in
    slt)
      CORPUS=$(bash "$SQL/fetch-sqllogictest.sh")
      for lvl in -O0 -O1 -O2; do
        bash "$SCRIPT_DIR/run-slt-twin.sh" "$lvl" "$CORPUS/select1.test" "$SQL/slt/q_two.test" "$SQL/slt/dd2_join.test"
        RC["slt$lvl"]=$?
      done ;;
    rv8|coremark|beebs)
      for lvl in -O0 -O2; do
        bash "$SCRIPT_DIR/run-twin-suite.sh" "$s" "$lvl"
      done
      python3 "$SCRIPT_DIR/compare-twins.py" --suite "$s" \
        --a "$CAPSTONE_TMP_ROOT/twins/$s-O0/summary.txt" --b "$CAPSTONE_TMP_ROOT/twins/$s-O2/summary.txt" \
        --label-a O0 --label-b O2 --meta "$QID" --tsv "$TSV"
      RC["$s"]=$? ;;
    *) echo "unknown suite $s" >&2; RC["$s"]=2 ;;
  esac
done

echo "== tier 2a summary ($QID)"
for k in "${!RC[@]}"; do printf '   %-10s exit %s\n' "$k" "${RC[$k]}"; done | sort
echo "   results: $TSV"
