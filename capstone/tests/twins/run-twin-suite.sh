#!/usr/bin/env bash
# Run one runtime suite at ONE optimisation level, keeping its summary for the twins gate.
#
#   run-twin-suite.sh <rv8|beebs|coremark> <-O0|-O1|-O2|-Os>
#
# Output: $CAPSTONE_TMP_ROOT/twins/<suite><level>/summary.txt (the runner's stdout),
# logs/ (the runner's per-benchmark logs) and meta.txt (level, QEMU binary id, date).
# Takes the nightly QEMU lock for the whole suite: the suites share one rootfs image.
# The suite's own exit status is recorded, not propagated -- the verdict comes from
# compare-twins.py, which treats an empty summary as an ERROR.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"
BENCH="$CAPSTONE_REPO_ROOT/capstone/benchmarks"
RUNTIME="$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu"

SUITE=${1:?usage: run-twin-suite.sh <rv8|beebs|coremark> <level>}
LEVEL=${2:?usage: run-twin-suite.sh <rv8|beebs|coremark> <level>}
case "$LEVEL" in -O0|-O1|-O2|-O3|-Os|-Oz) ;; *) echo "ERROR: level must be -O0/-O1/-O2/-Os, got $LEVEL" >&2; exit 2 ;; esac
QEMU=$CAPSTONE_QEMU_BINARY
[[ -x "$QEMU" ]] || { echo "ERROR: QEMU binary $QEMU missing; export CAPSTONE_QEMU_BINARY" >&2; exit 2; }
[[ -f "$CAPSTONE_BUILDROOT_DIR/build/images/rootfs.ext2" ]] || { echo "ERROR: no rootfs under $CAPSTONE_BUILDROOT_DIR/build/images; export CAPSTONE_BUILDROOT_DIR" >&2; exit 2; }

OUT="$CAPSTONE_TMP_ROOT/twins/$SUITE$LEVEL"
rm -rf "$OUT"; mkdir -p "$OUT/logs"
QID="qemu=$(sha256sum "$QEMU" | cut -c1-12)@$(date -r "$QEMU" +%Y-%m-%dT%H:%M)"
printf 'suite=%s\nlevel=%s\n%s\ncompiler=%s\nstarted=%s\n' "$SUITE" "$LEVEL" "$QID" "$CAPSTONE_LLVM_BIN" "$(date -Is)" > "$OUT/meta.txt"
LOCK="$CAPSTONE_TMP_ROOT/nightly-qemu.lock"

case "$SUITE" in
  rv8)      CMD=(env DOMAIN_OPT_LEVEL="$LEVEL" LOG_DIR="$OUT/logs" bash "$BENCH/rv8/run-all-rv8.sh") ;;
  beebs)    CMD=(env DOMAIN_OPT_LEVEL="$LEVEL" RUN_ALL_BEEBS_LOG_DIR="$OUT/logs" bash "$BENCH/beebs/run-all-beebs.sh") ;;
  coremark) CMD=(env DOMAIN_OPT_LEVEL="$LEVEL" LOG_FILE="$OUT/logs/coremark.log" bash "$RUNTIME/run-coremark.sh") ;;
  *) echo "ERROR: unknown suite $SUITE" >&2; exit 2 ;;
esac

echo "== twin $SUITE $LEVEL ($QID)"
(
  flock 9
  "${CMD[@]}" > "$OUT/summary.txt" 2> "$OUT/stderr.txt"
  echo "exit=$?" >> "$OUT/meta.txt"
) 9>"$LOCK"
printf 'finished=%s\n' "$(date -Is)" >> "$OUT/meta.txt"
grep -c . "$OUT/summary.txt" > /dev/null || echo "WARNING: empty summary for $SUITE $LEVEL (see $OUT/stderr.txt)"
tail -3 "$OUT/summary.txt"
