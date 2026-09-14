#!/usr/bin/env bash
# Does an ancestor's withdrawal reach through PostgreSQL's own manager, and stop where it should?
#
#   run-pg-hierarchy.sh
#
# run-subpool-test.sh beside this asks the level below the same family of questions, and the level
# below has no children: its sub-pools are regions, and a region does not create regions. A context
# does. So this runs tools/hierarchy_sublet.c over unmodified mcxt.c and an aset.c with one patch,
# and the delegating is done by the context TREE rather than by a carve. Nothing in the driver tells
# mcxt.c to walk firstchild and nextchild. That is the part that is upstream's.
#
# It needs no trace and no recording. The host shares five regions because the image declares them,
# and the driver takes the first three; the trace file is an argument whose contents do not matter,
# the way run-subpool-test.sh already passes one.
#
# The node pool stays at the board's 65 536. A run this small spends a few hundred, so raising it
# would hide nothing and prove nothing, and leaving it says the scenarios fit on the bitstream.
#
# Exit 0 if every claim held, non-zero otherwise, as the nightly expects.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
source "$SCRIPT_DIR/domain-run.sh"

OUT=${OUT:-$CAPSTONE_TMP_ROOT/pg-mmgr-host}
SHARE_DIR=${SHARE_DIR:-$OUT/share-hierarchy}
LOG_FILE=${LOG_FILE:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-pg-hierarchy.log}
PG_ARENA=${PG_ARENA:-$((16 * 1024 * 1024))}
PG_SCRATCH=${PG_SCRATCH:-$((1 * 1024 * 1024))}
PG_TRACE=${PG_TRACE:-$((1 * 1024 * 1024))}
PG_PAYLOAD=${PG_PAYLOAD:-65536}
PYTHON=${PYTHON:-python3}

pgdom_cma "$PG_TRACE" "$PG_ARENA" "$PG_SCRATCH"
pgdom_clean pg_hierarchy.dom

echo "== regions: payload $PG_PAYLOAD, arena $PG_ARENA (linear), trace $PG_TRACE (ignored), scratch $PG_SCRATCH, cma=$PG_CMA"

PG_PAYLOAD=$PG_PAYLOAD PG_ARENA=$PG_ARENA PG_TRACE=$PG_TRACE PG_SCRATCH=$PG_SCRATCH \
  PG_DRIVER=tools/hierarchy_sublet.c PG_DOM_IMAGE=pg_hierarchy.dom \
  OUT="$OUT" DOM_OUT="$OUT/sublet" \
  bash "$SCRIPT_DIR/build-mmgr-sublet.sh"

cp "$OUT/sublet/pg_hierarchy.dom" "$SHARE_DIR/"
pgdom_host
printf '\0' > "$SHARE_DIR/nothing.bin"

PG_TIMEOUT_MULTIPLIER=${PG_TIMEOUT_MULTIPLIER:-16} \
pgdom_boot \
  "cp /mnt/host/pg_host.user /tmp/pg_host.user && chmod 0755 /tmp/pg_host.user && /tmp/pg_host.user /mnt/host/pg_hierarchy.dom /mnt/host/nothing.bin --tail --linear-arena --scratch $PG_SCRATCH" \
  "${PG_MARKER:-__CAPSTONE_PG_HIER_DONE__}"

echo
echo "== what the domain reported"
sed -n '/__CAPSTONE_PG_HIER_ENTRY__/,/__CAPSTONE_PG_HIER_DONE__/p' "$LOG_FILE" || true
grep -q '__CAPSTONE_PG_HIER_GOOD__' "$LOG_FILE" \
  && echo "every claim held" \
  || { echo "a claim did not hold; the table above says which" >&2; exit 1; }
echo "full serial log: $LOG_FILE"
