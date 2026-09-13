#!/usr/bin/env bash
# Ask the hardware whether the level below keeps its promise.
#
#   run-subpool-test.sh
#
# Seven scenarios inside a Capstone domain, each one a claim the Sublet design
# makes about port/freestanding/pg_subpool.c. The one that only the hardware
# can answer is the second: a revoke that killed a linear child hands the
# region back uninitialised, and init is refused until it has been written
# through, so a sub-pool that works again after a reset is the proof that the
# one-revocation design is implementable and not just arguable.
#
# The guest host is the replay's, with one argument added: --linear-arena, which
# shares the arena REV_BORROWED rather than REV_DEFAULT. The difference is the
# whole discipline. A region shared the default way arrives non-linear, and a
# handle senior to a non-linear region has nothing to revoke, so the first run
# of this test stopped on it and said type 1 where it wanted 0.
#
# The host shares four regions and the test ignores the fourth, so a trace file
# is still an argument and its contents do not matter.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

OUT=${OUT:-$CAPSTONE_TMP_ROOT/pg-subpool-test}
SHARE_DIR=${SHARE_DIR:-$OUT/share}
LOG_FILE=${LOG_FILE:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-pg-subpool.log}

PG_PAYLOAD=${PG_PAYLOAD:-65536}
PG_ARENA=${PG_ARENA:-$((32 * 1024 * 1024))}
PG_TRACE=${PG_TRACE:-$((1 * 1024 * 1024))}
PG_CMA=${PG_CMA:-$(( (PG_TRACE + PG_ARENA) / 1048576 + 64 ))M}

MARKER=${PG_MARKER:-'__CAPSTONE_PG_SUBPOOL_DONE__'}
PYTHON=${PYTHON:-python3}

mkdir -p "$OUT" "$SHARE_DIR"
rm -f "$SHARE_DIR/pg_subpool_test.dom" "$SHARE_DIR/pg_host.user" \
      "$SHARE_DIR/nothing.bin"

echo "== regions: payload $PG_PAYLOAD, arena $PG_ARENA, trace $PG_TRACE (ignored), cma=$PG_CMA"

PG_PAYLOAD=$PG_PAYLOAD PG_ARENA=$PG_ARENA PG_TRACE=$PG_TRACE OUT="$OUT" \
  bash "$SCRIPT_DIR/build-subpool-test.sh"
cp "$OUT/pg_subpool_test.dom" "$SHARE_DIR/"

PG_EXTRA_DEFS="-DPG_REPLAY_PAYLOAD_SIZE=${PG_PAYLOAD}UL -DPG_REPLAY_ARENA_SIZE=${PG_ARENA}UL -DPG_REPLAY_TRACE_SIZE=${PG_TRACE}UL" \
  OUT="$OUT" OUT_HOST="$SHARE_DIR/pg_host.user" \
  bash "$SCRIPT_DIR/build-pg-host.sh"

# The fourth region is shared and never read, so its file is one byte.
printf '\0' > "$SHARE_DIR/nothing.bin"

"$PYTHON" "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier "${PG_TIMEOUT_MULTIPLIER:-8}" \
  --kernel-arg "cma=$PG_CMA" \
  --guest-command \
    "cp /mnt/host/pg_host.user /tmp/pg_host.user && chmod 0755 /tmp/pg_host.user && /tmp/pg_host.user /mnt/host/pg_subpool_test.dom /mnt/host/nothing.bin --tail --linear-arena" \
  --success-marker "$MARKER"

echo
echo "== what the domain reported"
sed -n '/__CAPSTONE_PG_SUBPOOL_ENTRY__/,/__CAPSTONE_PG_SUBPOOL_DONE__/p' "$LOG_FILE" || true
grep -q '__CAPSTONE_PG_SUBPOOL_GOOD__' "$LOG_FILE" \
  && echo "every claim held" \
  || { echo "a claim did not hold; the table above says which" >&2; exit 1; }
echo "full serial log: $LOG_FILE"
