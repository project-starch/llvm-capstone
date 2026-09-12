#!/usr/bin/env bash
# Replay a recorded PostgreSQL workload in a Capstone domain, under QEMU.
#
#   run-pg-replay.sh <flattened trace>
#
# Builds both halves, delivers them into the shared directory with the trace,
# boots the guest once and enters the domain.
#
# The trace is a file the recorder in the paper's experiments/a11/postgres
# wrote and verify.py flattened. It is copied rather than referenced, because
# the guest sees only the shared directory.
#
# Region sizes are set once here and passed to both builds and the host, which
# is the mechanism, and the domain checking what the host published is the
# backstop. What this shares with run-pg-sublet.sh is in domain-run.sh beside
# this: sizing the trace region, sizing cma, staging and booting.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
source "$SCRIPT_DIR/domain-run.sh"

TRACE=${1:?usage: run-pg-replay.sh <flattened trace>}
[ -f "$TRACE" ] || { echo "no trace at $TRACE" >&2; exit 1; }

# OUT is the port's one output directory: build-mmgr-host.sh leaves the
# configured PostgreSQL tree in it, and the other scripts read that.
OUT=${OUT:-$CAPSTONE_TMP_ROOT/pg-mmgr-host}
SHARE_DIR=${SHARE_DIR:-$OUT/share}
LOG_FILE=${LOG_FILE:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-pg-replay.log}
PG_ARENA=${PG_ARENA:-$((64 * 1024 * 1024))}
PG_PAYLOAD=${PG_PAYLOAD:-65536}
PYTHON=${PYTHON:-python3}

pgdom_trace_region "$TRACE"
pgdom_cma "$PG_TRACE" "$PG_ARENA"
pgdom_clean pg_mmgr_capstone.dom

echo "== regions: payload $PG_PAYLOAD, arena $PG_ARENA, trace $PG_TRACE (file $TRACE_BYTES), cma=$PG_CMA"

PG_PAYLOAD=$PG_PAYLOAD PG_ARENA=$PG_ARENA PG_TRACE=$PG_TRACE \
  OUT="$OUT" DOM_OUT="$OUT/domain" \
  bash "$SCRIPT_DIR/build-mmgr-domain.sh"

cp "$OUT/domain/pg_mmgr_capstone.dom" "$SHARE_DIR/"
pgdom_host
cp "$TRACE" "$SHARE_DIR/trace.a11"

pgdom_boot \
  "cp /mnt/host/pg_host.user /tmp/pg_host.user && chmod 0755 /tmp/pg_host.user && /tmp/pg_host.user /mnt/host/pg_mmgr_capstone.dom /mnt/host/trace.a11 --tail" \
  "${PG_MARKER:-__CAPSTONE_PG_REPLAY_DONE__}"

pgdom_report
