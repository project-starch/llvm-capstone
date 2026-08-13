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

# THE SAME OUT_DIR THE BUILD JUST USED, not a hardcoded one.
#
# This line read `$CAPSTONE_TMP_ROOT/sqlite-silicon/sqlite_silicon.dom` unconditionally while
# build-sqlite-silicon.sh has always honoured OUT_DIR. Setting OUT_DIR to keep a diagnostic build
# out of the default tree therefore built the new domain and RAN THE OLD ONE, with no error
# anywhere: the build log showed the probe being injected, the run log showed a clean pass, and
# the two described different binaries. Three consecutive QEMU runs were read as results before
# the missing probe output gave it away -- including a "the probe does not fire under QEMU"
# reading that was really "the probe was not in the domain that ran".
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/sqlite-silicon}
SHARE=${SHARE_DIR:-$CAPSTONE_TMP_ROOT/sqlite-silicon-share}
mkdir -p "$SHARE"
DOM="$OUT_DIR/sqlite_silicon.dom"
[[ -f "$DOM" ]] || { echo "ERROR: $DOM does not exist -- the build produced nothing" >&2; exit 1; }
cp -f "$DOM" "$SHARE/"
echo "== running $DOM  ($(sha256sum "$DOM" | cut -c1-16))"
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
