#!/usr/bin/env bash
# Replay a recorded PostgreSQL workload against its memory manager under
# Sublet, in a Capstone domain, under QEMU.
#
#   run-pg-sublet.sh <flattened trace>
#
# The protected arm of the A11 measurement. run-pg-replay.sh beside it is the
# unprotected one, over the same trace, through the same loop, against the same
# manager without the discipline. Everything the two report side by side
# therefore differs only by what the port does.
#
# Five regions, and two of them need saying. The arena is shared linear, with
# --linear-arena, because a handle senior to a non-linear region has nothing to
# revoke. And the driver's identity tables get a region of their own, because
# they are walked with ordinary pointer arithmetic and a linear capability
# refuses that.
#
# The node budget is the thing to watch on the board rather than here. A
# revocation node is spent by every split and every mrev, from a bump head with
# no reclamation on silicon, and this rung spends about two an object. The
# emulator has more and a free list, so it finishes; the board's resident
# bitstream has 65 536, which covers a slice.
# experiments/a11/postgres/nodes.py in the paper's repository prices it.
#
# What this shares with run-pg-replay.sh is in domain-run.sh beside this.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
source "$SCRIPT_DIR/domain-run.sh"

TRACE=${1:?usage: run-pg-sublet.sh <flattened trace>}
[ -f "$TRACE" ] || { echo "no trace at $TRACE" >&2; exit 1; }

OUT=${OUT:-$CAPSTONE_TMP_ROOT/pg-mmgr-host}
SHARE_DIR=${SHARE_DIR:-$OUT/share-sublet}
LOG_FILE=${LOG_FILE:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-pg-sublet.log}
PG_ARENA=${PG_ARENA:-$((64 * 1024 * 1024))}
PG_SCRATCH=${PG_SCRATCH:-$((48 * 1024 * 1024))}
PG_PAYLOAD=${PG_PAYLOAD:-65536}
PYTHON=${PYTHON:-python3}

pgdom_trace_region "$TRACE"
pgdom_cma "$PG_TRACE" "$PG_ARENA" "$PG_SCRATCH"
pgdom_clean pg_mmgr_sublet.dom

# The emulator's revocation-node pool, and it has to be raised for a run this
# long. A node comes from every split and every mrev, and this emulator reuses
# none of them: a freed chunk's capability copies persist in heap memory that
# has not been written over, so the node's refcount never reaches zero.
# Consumption is therefore about one node per allocation and not one per live
# object, which is also how the silicon behaves, so the number below is the
# honest one rather than a way around a limitation.
#
# The default, 65 536, matches the deployed bitstream. It is what the BOARD
# can do and it is the real constraint on a board measurement: this rung
# spends about two nodes an object, so 65 536 covers a slice of it. Raising it
# here measures the whole rung under the emulator and says so;
# experiments/a11/postgres/nodes.py prices what a board run would reach.
#
# Four nodes per record of the trace, which is above what the port spends per
# record and cheap: a node is twenty-four bytes.
PG_REV_NODES=${PG_REV_NODES:-$(( (TRACE_BYTES / 40) * 4 + 65536 ))}
export CAPSTONE_REV_NODES=$PG_REV_NODES

echo "== regions: payload $PG_PAYLOAD, arena $PG_ARENA (linear), trace $PG_TRACE (file $TRACE_BYTES), scratch $PG_SCRATCH, cma=$PG_CMA"
echo "== the emulator's revocation-node pool: $PG_REV_NODES ($(( PG_REV_NODES * 24 / 1048576 )) MiB); the board's bitstream has 65536"

PG_PAYLOAD=$PG_PAYLOAD PG_ARENA=$PG_ARENA PG_TRACE=$PG_TRACE PG_SCRATCH=$PG_SCRATCH \
  OUT="$OUT" DOM_OUT="$OUT/sublet" \
  bash "$SCRIPT_DIR/build-mmgr-sublet.sh"

cp "$OUT/sublet/pg_mmgr_sublet.dom" "$SHARE_DIR/"
pgdom_host
cp "$TRACE" "$SHARE_DIR/trace.a11"

PG_TIMEOUT_MULTIPLIER=${PG_TIMEOUT_MULTIPLIER:-60} \
pgdom_boot \
  "cp /mnt/host/pg_host.user /tmp/pg_host.user && chmod 0755 /tmp/pg_host.user && /tmp/pg_host.user /mnt/host/pg_mmgr_sublet.dom /mnt/host/trace.a11 --tail --linear-arena --scratch $PG_SCRATCH" \
  "${PG_MARKER:-__CAPSTONE_PG_REPLAY_DONE__}"

pgdom_report
if grep -q '__CAPSTONE_PG_SUBLET_ONE_EACH__' "$LOG_FILE"; then
  echo "one revocation per context that held an object"
else
  echo "the revocations do not match one per context; the table above says what they were" >&2
fi
