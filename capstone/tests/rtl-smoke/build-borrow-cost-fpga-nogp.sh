#!/usr/bin/env bash
set -euo pipefail

# Build the GLOBAL-FREE / SINGLE-REGION RTL/FPGA borrow-cost variant (2026-07-20).
#
# This is the variant that actually runs the borrow/revoke measurement on real
# CapliFive silicon: unlike borrow_cost_fpga.{c,dom}, the domain here has NO
# module statics, so our LLVM Capstone backend emits NO `cincoffset gp` (the
# gp = PCC(cursor 0) form the RTL never fabricates and that stalls `delin gp` on
# the board). It pairs the gp-free entry glue (start-fpga-nogp.S) with a
# global-free domain (borrow_cost_fpga_nogp.c) and a single-region controller
# (borrow_cost_fpga_nogp_ctl.c): one REV_SHARED region that is both scratch and
# results, measured entirely inside the one REGION_SHARE entry. See
# agent-handoff/history 20-07-2026 + the NDSS plan gp section for the full trail.
#
# Output: $OUT_DIR/borrow_cost_fpga_nogp_ctl (soft-float .user controller) and
# $OUT_DIR/borrow_cost_fpga_nogp.dom (Capstone .dom). These two ship in the FPGA
# rootfs overlay (see README.md).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR}
TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
OUT_DIR=${1:-$TMP_ROOT/capstone-rtl-smoke}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O2}

GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
QEMU_PROBE_DIR="$SCRIPT_DIR/../runtime-qemu/borrow-cost-probe"

mkdir -p "$TMP_ROOT" "$OUT_DIR"

# Freestanding soft-float controller (no glibc -> zero FP -> never hits the `fsd`
# the board rejects). Own _start, raw Linux syscalls, integer-only output. Same
# ioctl protocol as libcapstone; single REV_SHARED-region flow (no CALL).
"$GUEST_CC" \
  -Os -static -no-pie -fno-pie -nostdlib -ffreestanding -fno-stack-protector \
  -march=rv64imac -mabi=lp64 \
  -I"$SCRIPT_DIR" \
  -I"$QEMU_PROBE_DIR" \
  -o "$OUT_DIR/borrow_cost_fpga_nogp_ctl" \
  "$SCRIPT_DIR/borrow_cost_fpga_nogp_ctl.c"
printf 'Built %s\n' "$OUT_DIR/borrow_cost_fpga_nogp_ctl"

# Domain payload (Capstone clang) with the gp-free entry glue. We build via .s so
# we can retarget domain_main's return.
#
# WHY THE .s PATCH: our clang lowers a C function's return to `cjalr zero, 0(ra)`
# (a capability return), which requires ra to hold a code CAPABILITY. The entry
# glue calls domain_main with a plain `call` (scalar ra) -- because on a cscall
# domain entry there is no gp-free way to form a code capability (PCC is not
# readable into a GPR, and CTVEC arrives as an untagged scalar). So we rewrite the
# single return to a plain `ret` (jalr zero, 0(ra)); plain call + plain ret both
# stay inside PCC, exactly the reference monitor's ABI (sbi_capstone.S: plain
# `call handle_exception`). domain_main is a leaf here (all measure_* inline), so
# `cjalr zero, 0(ra)` is its ONLY cjalr -- we assert exactly one rewrite.
ASM_DIR="$OUT_DIR/asm"
OBJ_DIR="$OUT_DIR/obj"
mkdir -p "$ASM_DIR" "$OBJ_DIR"
RAW_S="$ASM_DIR/borrow_cost_fpga_nogp.raw.s"
PATCHED_S="$ASM_DIR/borrow_cost_fpga_nogp.s"

"$CAPSTONE_CLANG" -target capstone64-unknown-elf -ffreestanding \
  "$DOMAIN_OPT_LEVEL" -I"$SCRIPT_DIR" -I"$QEMU_PROBE_DIR" \
  -S "$SCRIPT_DIR/borrow_cost_fpga_nogp.c" -o "$RAW_S"

# Guard 1: no gp anywhere in the compiler output (a new static / memcpy libcall
# would reintroduce it). Checked on the RAW asm so it can't match our own patch
# comment below.
if grep -qE '\bgp\b' "$RAW_S"; then
  echo "ERROR: gp reference in domain asm -- the domain is not gp-free:" >&2
  grep -nE '\bgp\b' "$RAW_S" >&2
  exit 1
fi

# Rewrite the capability return -> plain ret. Assert exactly one occurrence.
n_ret=$(grep -cE '^[[:space:]]*cjalr[[:space:]]+zero,[[:space:]]*0\(ra\)' "$RAW_S" || true)
if [[ "$n_ret" != "1" ]]; then
  echo "ERROR: expected exactly 1 'cjalr zero, 0(ra)' return to rewrite, found $n_ret" >&2
  grep -nE 'cjalr' "$RAW_S" >&2 || true
  exit 1
fi
sed -E 's/^([[:space:]]*)cjalr[[:space:]]+zero,[[:space:]]*0\(ra\)/\1jalr zero, 0(ra)  # retargeted capability-return to plain ret (see build script)/' \
  "$RAW_S" > "$PATCHED_S"

# Guard 2: no cjalr survives (neither call-form nor return-form).
if grep -qE '\bcjalr\b' "$PATCHED_S"; then
  echo "ERROR: a cjalr survived in the domain asm -- not silicon-safe gp-free:" >&2
  grep -nE '\bcjalr\b' "$PATCHED_S" >&2
  exit 1
fi

# Assemble the patched .s + entry glue + gct tail, then link (mirrors
# runtime-qemu/build-domain.sh, but feeds the patched domain object).
GCT_TAIL_SRC="$SCRIPT_DIR/../runtime-qemu/gct-section-end.S"
LINKER_SCRIPT="$CAPSTONE_REPO_ROOT/capstone/my_first_domain/link.ld"
"$CAPSTONE_CLANG" -target capstone64-unknown-elf -ffreestanding \
  -c "$SCRIPT_DIR/start-fpga-nogp.S" -o "$OBJ_DIR/start.o"
"$CAPSTONE_CLANG" -target capstone64-unknown-elf -ffreestanding \
  -c "$GCT_TAIL_SRC" -o "$OBJ_DIR/gct-tail.o"
"$CAPSTONE_CLANG" -target capstone64-unknown-elf -ffreestanding \
  -c "$PATCHED_S" -o "$OBJ_DIR/main.o"
"$CAPSTONE_LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DIR/borrow_cost_fpga_nogp.dom" \
  "$OBJ_DIR/start.o" "$OBJ_DIR/main.o" "$OBJ_DIR/gct-tail.o"

printf 'Built %s (%s)\n' "$OUT_DIR/borrow_cost_fpga_nogp.dom" "$DOMAIN_OPT_LEVEL"
printf 'asm: %s (return retargeted to plain ret)\n' "$PATCHED_S"
printf 'gp-free + cjalr-free check: OK\n'
