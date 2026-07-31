#!/usr/bin/env bash
# Build a batch of staged SQLite diagnostic domains, one per CAPSTONE_SQLITE_STAGE value,
# into distinct .dom names ready for staging into the initramfs.
#
# Usage:  bash build-stage-probes.sh 57 58 59
#         PROBE_PREFIX=wd bash build-stage-probes.sh 55 56
#
# Exists because the batch-variants rule (CLAUDE.md, "Debugging a blocker: BATCH VARIANTS")
# means several of these get built per board load, and reconstructing the invocation by hand
# each time has already produced two wasted loads: a build whose artifact was never actually
# rebuilt, and a probe staged under the wrong name. This writes to OUT_DIR only -- it does
# NOT touch the overlay, build/target/ or the firmware, so it is safe to run while a board
# session is in flight.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

PREFIX=${PROBE_PREFIX:-wd}
DEST=${PROBE_DEST:-$CAPSTONE_TMP_ROOT/stage-probes}
mkdir -p "$DEST"

[[ $# -gt 0 ]] || { echo "usage: $0 <stage> [stage ...]" >&2; exit 2; }

for stage in "$@"; do
  out="$CAPSTONE_TMP_ROOT/sqlite-silicon-s$stage"
  echo "=== building stage $stage -> ${PREFIX}${stage}.dom ==="
  OUT_DIR="$out" DOMAIN_EXTRA_DEFS="-DCAPSTONE_SQLITE_STAGE=$stage" \
    bash "$SCRIPT_DIR/build-sqlite-silicon.sh" > "$out.log" 2>&1 || {
      echo "BUILD FAILED for stage $stage -- tail of $out.log:" >&2
      tail -20 "$out.log" >&2
      exit 1
    }
  src="$out/sqlite_silicon.dom"
  [[ -f "$src" ]] || { echo "no artifact at $src" >&2; exit 1; }
  cp -f "$src" "$DEST/${PREFIX}${stage}.dom"
  # Verify the stage actually differs -- a build that silently reused a cached object is the
  # failure mode that makes a probe "test nothing", so report the hash rather than trust it.
  printf '    %s  %s\n' "$(sha256sum "$DEST/${PREFIX}${stage}.dom" | cut -c1-16)" \
                        "$(stat -c%s "$DEST/${PREFIX}${stage}.dom") bytes"
done

echo
echo "built into $DEST:"
ls -l "$DEST" | tail -n +2
echo
echo "distinct hashes (must equal the number of stages built):"
sha256sum "$DEST"/*.dom | awk '{print $1}' | sort -u | wc -l
