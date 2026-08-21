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
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"

mkdir -p "$OUT_DIR"

# HOST_EXTRA_DEFS MUST CARRY THE SAME -DSQLITE_HC_REGION_SIZE AS THE DOMAIN BUILD.
# The two halves are separate compilations sharing one #define; a drift makes the host map
# N bytes while the domain bounds its writes by M. run-sqlite-slt.sh sets it once for both,
# and the domain refuses to run on a mismatch (SQLITE_HC_ERR_REGION_MISMATCH) -- the gate
# is the backstop, the single assignment is the mechanism.
read -r -a _host_defs <<< "${HOST_EXTRA_DEFS:-}"

"$GUEST_CC" -O2 -I"$SCRIPT_DIR" "${_host_defs[@]}" \
  -o "$OUT_HOST" \
  "$HOST_SRC" \
  "$LIBCAPSTONE_C"

echo "Built $OUT_HOST"
