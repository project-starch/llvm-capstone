#!/usr/bin/env bash
# Run the SILICON-config SQLite domain under QEMU (stage S5 gate).
#
# Same five success markers as run-sqlite-memory.sh, but against the domain built by
# build-sqlite-silicon.sh: -capstone-gp-captable + gp-free + shrink off, one module,
# descriptor-driven glue, globals offset sized to .text.
#
# TWO THINGS THAT MUST BE REBUILT, or the run silently tests the wrong thing:
#  * sqlite_host.user -- it links libcapstone.c statically, and libcapstone is what
#    packs the globals offset into entry_offset. A stale host makes the monitor fall
#    back to 0x1000, which trips the blob-does-not-fit error (capstone_error 0xB10B).
#    That is how this was caught; the loud error did its job.
#  * the monitor, if sbi_capstone.c changed (see the C-11 recipe).
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

bash "$SCRIPT_DIR/build-sqlite-silicon.sh"
bash "$SCRIPT_DIR/build-sqlite-host.sh"

SHARE=${SHARE_DIR:-$CAPSTONE_TMP_ROOT/sqlite-silicon-share}
mkdir -p "$SHARE"
cp -f "$CAPSTONE_TMP_ROOT/sqlite-silicon/sqlite_silicon.dom" "$SHARE/"
cp -f "$CAPSTONE_TMP_ROOT/sqlite-build/sqlite_host.user"     "$SHARE/"

python3 "$ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE" \
  --log-file "$CAPSTONE_TMP_ROOT/sqlite-silicon.log" \
  --timeout-multiplier 6 \
  --guest-command \
    'cp /mnt/host/sqlite_host.user /tmp/h.user && chmod 0755 /tmp/h.user && /tmp/h.user /mnt/host/sqlite_silicon.dom' \
  --success-marker 'row name=alpha value=11' \
  --success-marker 'row name=beta value=22' \
  --success-marker 'row name=gamma value=33' \
  --success-marker '__CAPSTONE_SQLITE_EXTENDED_PASSED__' \
  --success-marker '__CAPSTONE_SQLITE_MEMORY_PASSED__'

echo "__CAPSTONE_SQLITE_SILICON_PASSED__"
