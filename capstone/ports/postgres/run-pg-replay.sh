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
# is the mechanism; the domain checking what the host published is the
# backstop. cma= has to cover the arena and the trace together, and a region
# above 4 MiB comes from that area rather than the buddy allocator.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

TRACE=${1:?usage: run-pg-replay.sh <flattened trace>}
[ -f "$TRACE" ] || { echo "no trace at $TRACE" >&2; exit 1; }

# OUT is the port's one output directory: build-mmgr-host.sh leaves the
# configured PostgreSQL tree in it, and the other two scripts read that.
OUT=${OUT:-$CAPSTONE_TMP_ROOT/pg-mmgr-host}
SHARE_DIR=${SHARE_DIR:-$OUT/share}
LOG_FILE=${LOG_FILE:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-pg-replay.log}

# The trace region is rounded up to a megabyte above the file, so one build
# serves traces of different lengths without a rebuild.
TRACE_BYTES=$(stat -c %s "$TRACE")
PG_TRACE=${PG_TRACE:-$(( (TRACE_BYTES + 1048575) / 1048576 * 1048576 + 1048576 ))}
PG_ARENA=${PG_ARENA:-$((64 * 1024 * 1024))}
PG_PAYLOAD=${PG_PAYLOAD:-65536}
# Room for both regions and the kernel's own use of the area.
PG_CMA=${PG_CMA:-$(( (PG_TRACE + PG_ARENA) / 1048576 + 64 ))M}

MARKER=${PG_MARKER:-'__CAPSTONE_PG_REPLAY_DONE__'}
PYTHON=${PYTHON:-python3}

mkdir -p "$OUT" "$SHARE_DIR"
rm -f "$SHARE_DIR/pg_mmgr_capstone.dom" "$SHARE_DIR/pg_host.user" "$SHARE_DIR/trace.a11"

echo "== regions: payload $PG_PAYLOAD, arena $PG_ARENA, trace $PG_TRACE (file $TRACE_BYTES), cma=$PG_CMA"

PG_PAYLOAD=$PG_PAYLOAD PG_ARENA=$PG_ARENA PG_TRACE=$PG_TRACE \
  OUT="$OUT" DOM_OUT="$OUT/domain" \
  bash "$SCRIPT_DIR/build-mmgr-domain.sh"
cp "$OUT/domain/pg_mmgr_capstone.dom" "$SHARE_DIR/"

PG_EXTRA_DEFS="-DPG_REPLAY_PAYLOAD_SIZE=${PG_PAYLOAD}UL -DPG_REPLAY_ARENA_SIZE=${PG_ARENA}UL -DPG_REPLAY_TRACE_SIZE=${PG_TRACE}UL" \
  OUT="$OUT" OUT_HOST="$SHARE_DIR/pg_host.user" \
  bash "$SCRIPT_DIR/build-pg-host.sh"

cp "$TRACE" "$SHARE_DIR/trace.a11"

"$PYTHON" "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier "${PG_TIMEOUT_MULTIPLIER:-40}" \
  --kernel-arg "cma=$PG_CMA" \
  --guest-command \
    "cp /mnt/host/pg_host.user /tmp/pg_host.user && chmod 0755 /tmp/pg_host.user && /tmp/pg_host.user /mnt/host/pg_mmgr_capstone.dom /mnt/host/trace.a11 --tail" \
  --success-marker "$MARKER"

echo
echo "== what the domain reported"
sed -n '/__CAPSTONE_PG_REPLAY_ENTRY__/,/__CAPSTONE_PG_REPLAY_DONE__/p' "$LOG_FILE" || true
echo "full serial log: $LOG_FILE"
