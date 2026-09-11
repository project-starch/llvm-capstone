#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/sqlite-build}
OUT_HOST=${OUT_HOST:-$OUT_DIR/sqlite_host.user}
HOST_SRC=${HOST_SRC:-$SCRIPT_DIR/sqlite_host.c}
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
# The caplifive-BUILDROOT copy is correct here and must NOT be swapped for the
# caplifive-system one. It is the only copy carrying the globals-offset packing
# (entry_off | (globals_off << 32)); the caplifive-system copy is 5 KB older with zero
# globals_off references, so building against it silently delivers gpoff = 0x1000
# instead of 0x140000 -- dom_gp then covers only the first 4 KiB of a 1.39 MB image and
# the monitor copies .text as though it were globals. Measured: that build printed no
# "Globals offset" line at all and call_dom returned -1.
#
# What this copy DOES need is the guard the caplifive-system copy already has. Its
# debug_counter_inc is unconditional and emits `.insn r 0x5b, 0x1, 0x45` = QEMU's
# csdebugcount, a QEMU-private debug op (funct7 >= 0x40 on opcode 0x5b is the whole
# QEMU-only block; CVA6 puts its debug ops on opcode 0x7b and has no counter op).
# QEMU bumps an emulator-side array; the FPGA raises ILLEGAL_INSTRUCTION. Board-proven
# at mepc 0x1e84, 52 bytes into create_region, word 8ae7905b.
#
# So: keep this file, and compile the counters out via the macro below.
# RESOLVED FROM $CAPSTONE_BUILDROOT_DIR, not from the repo root. capstone-test-env.sh:54 already
# exports it, defaulting to the same path, so this is identical in a normal checkout. It matters
# in a WORKTREE, where submodules are empty: the workaround was to symlink caplifive-buildroot
# into the worktree, and git then sees a type change whose content is the user's home path, so
# precommit-scan blocks every commit in that tree while a build is in flight -- including another
# lane's commit, on a hit that has nothing to do with their diff. Using the env var removes the
# need for the symlink instead of teaching the scanner to ignore it.
LIBCAPSTONE_C="$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib/libcapstone.c"

mkdir -p "$OUT_DIR"

# HOST_EXTRA_DEFS MUST CARRY THE SAME -DSQLITE_HC_REGION_SIZE AS THE DOMAIN BUILD.
# The two halves are separate compilations sharing one #define; a drift makes the host map
# N bytes while the domain bounds its writes by M. run-sqlite-slt.sh sets it once for both,
# and the domain refuses to run on a mismatch (SQLITE_HC_ERR_REGION_MISMATCH) -- the gate
# is the backstop, the single assignment is the mechanism.
read -r -a _host_defs <<< "${HOST_EXTRA_DEFS:-}"

"$GUEST_CC" -O2 -pthread -I"$SCRIPT_DIR" "${_host_defs[@]}" \
  -o "$OUT_HOST" \
  "$HOST_SRC" \
  "$LIBCAPSTONE_C"

# WHICH BASE THIS BINARY WAS BUILT AGAINST, written beside it so a delivered set carries its own
# provenance. The host links libcapstone.c out of the caplifive-buildroot submodule by absolute
# path (see LIBCAPSTONE_C above), so the HOST half of an artifact set moves whenever that submodule
# does -- while the domain half and the toolchain sit still, and no gate says a word about it.
#
# On 2026-09-11 two sets built 78 minutes apart had host binaries differing by 3,332 bytes: a
# submodule commit raising MAX_REGION_N from 64 to 96 landed between them, adding exactly
# 32*8 + 32*4 = 384 bytes of .bss. Source, defines and toolchain were identical and all older than
# both builds, so nothing else could have explained it -- and it was found only by chasing an
# unexpected hash rather than shrugging at it. The library sha is the load-bearing field: it does
# not depend on the submodule's git state being clean or even on it being a checkout.
_br_dir="$CAPSTONE_BUILDROOT_DIR"
{
  echo "host source:     $HOST_SRC"
  echo "libcapstone.c:   $LIBCAPSTONE_C"
  echo "libcapstone sha: $(sha256sum "$LIBCAPSTONE_C" | cut -d' ' -f1)"
  echo "buildroot HEAD:  $(git -C "$_br_dir" rev-parse HEAD 2>/dev/null || echo NOT-A-CHECKOUT)"
  echo "buildroot dirty: $(git -C "$_br_dir" status --porcelain 2>/dev/null | wc -l) modified path(s)"
  echo "built:           $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT_HOST.provenance"

echo "Built $OUT_HOST"
sed 's/^/  /' "$OUT_HOST.provenance"
