#!/usr/bin/env bash
# The guest-side host program for the PostgreSQL memory-manager domain.
#
#   build-pg-host.sh
#
# Cross-compiled for the guest with the buildroot toolchain, with libcapstone.c
# linked in directly, as the SQLite port's host is.
#
# The region sizes are the one thing both halves must agree on, and they are
# separate compilations: PG_EXTRA_DEFS must carry the same -D values the domain
# build used, and the host publishes what it used in the metadata so a drift
# shows up as a refusal rather than as a host mapping N bytes while the domain
# bounds its writes by M.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/pg-mmgr-host}
OUT_HOST=${OUT_HOST:-$OUT_DIR/pg_host.user}
HOST_SRC=${HOST_SRC:-$SCRIPT_DIR/pg_host.c}
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
# The caplifive-BUILDROOT copy, not the caplifive-system one: it is the only
# copy carrying the globals-offset packing, and without that the monitor copies
# .text as though it were globals and call_dom returns -1. Its debug counters
# emit an instruction only QEMU has, and they stay compiled out unless
# CAPSTONE_DEBUG_ENABLE is defined, so this binary runs on the board too.
# Resolved from the env var and not from the repo root, because in a worktree
# the submodules are empty and a symlink there makes the scanner block commits.
LIBCAPSTONE_C="$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib/libcapstone.c"

[ -x "$GUEST_CC" ] || { echo "no guest compiler at $GUEST_CC" >&2; exit 1; }
[ -f "$LIBCAPSTONE_C" ] || { echo "no libcapstone.c at $LIBCAPSTONE_C" >&2; exit 1; }

mkdir -p "$OUT_DIR"
read -r -a _defs <<< "${PG_EXTRA_DEFS:-}"

"$GUEST_CC" -O2 -pthread -Wall -I"$SCRIPT_DIR" \
  -I"$(dirname "$LIBCAPSTONE_C")" \
  -I"$CAPSTONE_BUILDROOT_DIR/package/modcapstone/include" \
  "${_defs[@]}" \
  -o "$OUT_HOST" \
  "$HOST_SRC" \
  "$LIBCAPSTONE_C"

{
  echo "date            $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "source          $(sha256sum "$HOST_SRC" | cut -c1-64)  $(basename "$HOST_SRC")"
  echo "libcapstone     $(sha256sum "$LIBCAPSTONE_C" | cut -c1-64)"
  echo "compiler        $("$GUEST_CC" --version | head -1)"
  echo "defines         ${PG_EXTRA_DEFS:-(none)}"
} > "$OUT_HOST.provenance"

echo "built $OUT_HOST"
