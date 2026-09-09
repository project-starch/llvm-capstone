#!/usr/bin/env bash
# speedtest1 in a Capstone domain on QEMU: SQLite's own benchmark with fixed arguments and the
# lookaside pool on, the allocator chain the paper measures. Output arrives on the hostcall
# payload the host prints when the domain returns; the domain ends with
# __CAPSTONE_SPEEDTEST1_DONE__ rc=<n>.
#
#   run-sqlite-speedtest1.sh            --memdb --size 1 --testset main --verify --stats
#   SPEEDTEST1_ARGS='"--memdb","--size","2"' run-sqlite-speedtest1.sh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

SQLITE_VERSION=${SQLITE_VERSION:-3530300}
SQLITE_SRC_YEAR=${SQLITE_SRC_YEAR:-2026}
SRC_ROOT=$CAPSTONE_TMP_ROOT/sqlite-src
SPEEDTEST1_SRC=${SPEEDTEST1_SRC:-$SRC_ROOT/sqlite-src-$SQLITE_VERSION/test/speedtest1.c}
if [ ! -f "$SPEEDTEST1_SRC" ]; then
  # the amalgamation zip fetch-sqlite.sh takes does not ship speedtest1.c; the source tree does
  mkdir -p "$SRC_ROOT"
  curl -sfL -o "$SRC_ROOT/sqlite-src-$SQLITE_VERSION.zip" \
    "https://www.sqlite.org/$SQLITE_SRC_YEAR/sqlite-src-$SQLITE_VERSION.zip"
  unzip -qo "$SRC_ROOT/sqlite-src-$SQLITE_VERSION.zip" "sqlite-src-$SQLITE_VERSION/test/speedtest1.c" -d "$SRC_ROOT"
fi

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/speedtest1-build}
SHARE_DIR=${SHARE_DIR:-$OUT_DIR}
LOG_FILE=${LOG_FILE:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-speedtest1.log}
REGION=${SPEEDTEST1_REGION_SIZE:-1048576}          # the payload, as the SLT runner sizes it
export SQLITE_OPT_LEVEL=${SQLITE_OPT_LEVEL:--O1}
export SQLITE_LOOKASIDE=${SQLITE_LOOKASIDE:-1200,40}
# SPEEDTEST1_STACK_ARENA=<bytes> carves memsys5's arena from the domain's stack region (2 MiB
# fits --size 1 with room for the stack); the in-image heap then only needs to exist.
STACK_ARENA=${SPEEDTEST1_STACK_ARENA:-2097152}
DOMAIN_ARGS="${SPEEDTEST1_ARGS:+-DSPEEDTEST1_ARGS=$SPEEDTEST1_ARGS} ${SPEEDTEST1_STOP_AT:+-DSPEEDTEST1_STOP_AT=$SPEEDTEST1_STOP_AT} -DSPEEDTEST1_STACK_ARENA=$STACK_ARENA"
export DOMAIN_EXTRA_FLAGS="-DSQLITE_HEAP_SIZE=${SPEEDTEST1_HEAP:-16384} -DSQLITE_HC_REGION_SIZE=$REGION -DSPEEDTEST1_SRC=\"$SPEEDTEST1_SRC\" $DOMAIN_ARGS ${DOMAIN_EXTRA_FLAGS:-}"
export HOST_EXTRA_DEFS="-DSQLITE_HC_REGION_SIZE=$REGION ${HOST_EXTRA_DEFS:-}"

mkdir -p "$OUT_DIR" "$SHARE_DIR"
rm -f "$SHARE_DIR/speedtest1_capstone.dom" "$SHARE_DIR/sqlite_host.user"
DOMAIN_SRC="$SCRIPT_DIR/speedtest1_domain.c" OUT_DIR="$OUT_DIR" OUT_DOM="$SHARE_DIR/speedtest1_capstone.dom" \
  bash "$SCRIPT_DIR/build-sqlite-capstone.sh"
OUT_DIR="$OUT_DIR" OUT_HOST="$SHARE_DIR/sqlite_host.user" \
  bash "$SCRIPT_DIR/build-sqlite-host.sh"

python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier "${SPEEDTEST1_TIMEOUT_MULTIPLIER:-40}" \
  --guest-command \
    'cp /mnt/host/sqlite_host.user /tmp/sqlite_host.user && chmod 0755 /tmp/sqlite_host.user && /tmp/sqlite_host.user /mnt/host/speedtest1_capstone.dom --tail' \
  --success-marker '__CAPSTONE_SPEEDTEST1_DONE__ rc=0'

echo "run-sqlite-speedtest1.sh completed. Full serial log: $LOG_FILE"
