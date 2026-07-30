#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/sqlite-build}
OUT_HOST=${OUT_HOST:-$OUT_DIR/sqlite_host.user}
HOST_SRC=${HOST_SRC:-$SCRIPT_DIR/sqlite_host.c}
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
# MUST be the caplifive-SYSTEM copy, not caplifive-buildroot. The two trees have diverged
# and the difference is fatal on hardware: buildroot's copy defines
#   debug_counter_inc -> __asm__ volatile(".insn r 0x5b, 0x1, 0x45, x0, ...")
# UNCONDITIONALLY, while caplifive-system's guards it behind #ifdef CAPSTONE_DEBUG_ENABLE
# (never defined, so it compiles to nothing). funct7 0x45 is not in capstone-spec
# (highest defined is 0100001 = RETURN) and not in our LLVM backend, so it is a
# QEMU-only debug hook. QEMU tolerates it; the FPGA raises ILLEGAL INSTRUCTION.
#
# Board-proven 2026-07-30: SQLite died at mepc 0x1e84, 52 bytes into create_region,
# on word 8ae7905b -- exactly this encoding. The monitor diagnostics named the site
# (ILLX) and the load-base markers placed it inside the host image. Ladder rungs never
# hit it because ladder_perf_ctl.c is -nostdlib and hand-rolls the ioctl path, so it
# never links libcapstone at all -- which is why bigblob reproduced every
# create_domain input and still passed.
LIBCAPSTONE_C="$CAPSTONE_REPO_ROOT/capstone/caplifive-system/sw/buildroot/package/modcapstone/userspace/lib/libcapstone.c"

mkdir -p "$OUT_DIR"

"$GUEST_CC" -O2 -I"$SCRIPT_DIR" \
  -o "$OUT_HOST" \
  "$HOST_SRC" \
  "$LIBCAPSTONE_C"

echo "Built $OUT_HOST"
