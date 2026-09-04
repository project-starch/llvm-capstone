#!/usr/bin/env bash
# Run SQLLogicTest files in the silicon-config SQLite domain at ONE optimisation level
# and compare each against the native run of the same file (slt-compare.sh).
#
#   run-slt-twin.sh <-O0|-O1|-O2|-Os> [file.test ...]     default: select1.test
#
# One verdict line per file lands in results/<date>.tsv, naming the level, the domain
# image and the QEMU binary it ran on -- never the capture.  Exit 0 only if every file
# AGREEs.  Takes the nightly QEMU lock for each boot (the suites share one rootfs image
# and must never run two at once), and holds it across the domain build so a concurrent
# suite cannot see a half-written image directory.
#
# The domain is built with SLT_MAX_REPORTED raised to $SLT_TWIN_CAP through
# DOMAIN_EXTRA_DEFS so that every failing record is listed; the comparator refuses a run
# whose failures reach that cap.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"
SQL="$CAPSTONE_REPO_ROOT/capstone/benchmarks/sqlite"

LEVEL=${1:?usage: run-slt-twin.sh <-O0|-O1|-O2|-Os> [file.test ...]}
shift
case "$LEVEL" in -O0|-O1|-O2|-O3|-Os|-Oz) ;; *) echo "ERROR: level must be -O0/-O1/-O2/-Os, got $LEVEL" >&2; exit 2 ;; esac
CAP=${SLT_TWIN_CAP:-256}
LOCK="$CAPSTONE_TMP_ROOT/nightly-qemu.lock"

if [[ $# -eq 0 ]]; then
  CORPUS=$(bash "$SQL/fetch-sqllogictest.sh")
  set -- "$CORPUS/select1.test"
fi

QEMU=$CAPSTONE_QEMU_BINARY
[[ -x "$QEMU" ]] || { echo "ERROR: QEMU binary $QEMU missing; export CAPSTONE_QEMU_BINARY" >&2; exit 2; }
[[ -f "$CAPSTONE_BUILDROOT_DIR/build/images/rootfs.ext2" ]] || { echo "ERROR: no rootfs under $CAPSTONE_BUILDROOT_DIR/build/images; export CAPSTONE_BUILDROOT_DIR" >&2; exit 2; }
QID="qemu=$(sha256sum "$QEMU" | cut -c1-12)@$(date -r "$QEMU" +%Y-%m-%dT%H:%M)"

OUT="$CAPSTONE_TMP_ROOT/twins/slt$LEVEL"
mkdir -p "$OUT"
rc_all=0
for T in "$@"; do
  [[ -f "$T" ]] || { echo "ERROR: no such test file: $T" >&2; exit 2; }
  name=$(basename "$T" .test)
  LOG="$OUT/$name.log"
  rm -f "$LOG"
  echo "== slt twin $name $LEVEL"
  (
    flock 9
    SQLITE_OPT_LEVEL="$LEVEL" DOMAIN_EXTRA_DEFS="-DSLT_MAX_REPORTED=${CAP}u" \
      SLT_TEST="$T" OUT_DIR="$OUT" SLT_LOG_FILE="$LOG" \
      bash "$SQL/run-sqlite-slt.sh" > "$OUT/$name.run.log" 2>&1 \
      || echo "   run-sqlite-slt.sh exited $? (see $OUT/$name.run.log)"
  ) 9>"$LOCK"
  DOM="$OUT/sqlite_silicon.dom"
  if [[ -f "$DOM" ]]; then DID="dom=$(sha256sum "$DOM" | cut -c1-12)"; else DID="dom=none"; fi
  # No log at all is the same as no summary: the comparator must report ERROR, never skip.
  [[ -f "$LOG" ]] || : > "$LOG"
  bash "$SCRIPT_DIR/slt-compare.sh" "$T" "$LOG" "slt $name $LEVEL $DID $QID" || rc_all=1
done
exit $rc_all
