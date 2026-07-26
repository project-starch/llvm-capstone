#!/usr/bin/env bash
# Build the FPGA perf artifacts for one (or all) silicon-ladder rung(s): the
# generic soft-float controller (once) + each rung's perf .dom, and print the
# native oracle checksum for the correctness gate.
#
#   usage: build-ladder-fpga.sh [rung ...]      (default: all 7 ready rungs)
#   out:   $OUT_DIR/ladder_perf_ctl              (soft-float .user controller)
#          $OUT_DIR/<rung>.dom                    (Capstone gp-captable domain)
#          $OUT_DIR/<rung>.oracle                 (native cc -O0 checksum, 1 line)
#
# The domain is the same silicon-config gp-captable build as the QEMU ladder
# (build-ladder-domain.sh) but with the perf domain_main (<rung>_fpga_app.c) that
# also records mcycle. The controller (buildroot cross-gcc, not the Capstone
# clang) creates the domain, shares one region = the entry, and reads back
# retval + cycles. See ladder_perf_domain.h + ladder_perf_ctl.c.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

LAD="$SCRIPT_DIR/../runtime-qemu/silicon-ladder"
BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR}
GUEST_CC=${GUEST_CC:-$BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/ladder-fpga}
mkdir -p "$OUT_DIR"

RUNGS=("$@")
[[ ${#RUNGS[@]} -eq 0 ]] && RUNGS=(matmult_int coremark_matrix rv8_primes \
  beebs_crc32 beebs_insertsort beebs_prime beebs_recursion)

# 1. Controller (once) -- freestanding soft-float, no glibc (board rejects fsd).
"$GUEST_CC" -Os -static -no-pie -fno-pie -nostdlib -ffreestanding \
  -fno-stack-protector -march=rv64imac -mabi=lp64 \
  -o "$OUT_DIR/ladder_perf_ctl" "$SCRIPT_DIR/ladder_perf_ctl.c"
echo "Built $OUT_DIR/ladder_perf_ctl"

# 2. Per-rung perf domain + native oracle.
for R in "${RUNGS[@]}"; do
  APP="$LAD/${R}_fpga_app.c"
  HOST="$LAD/${R}_host.c"
  [[ -f "$APP" && -f "$HOST" ]] || { echo "missing ${R}_fpga_app.c / ${R}_host.c" >&2; exit 1; }
  # coremark_matrix overflows the 4 KiB PCC window at -O0; it is built -Os (see
  # silicon-ladder/run-coremark-matrix-qemu.sh). Others use the ladder default -O0.
  # LADDER_OPT overrides the per-rung default for the whole set. -O0 emits an
  # `ldc` reload of the region capability before EVERY region store; -O1+ keeps it
  # in a register. Since an extra capability store is the confirmed miscompute
  # trigger (26-07 bisect), optimisation level is a candidate workaround.
  OPT=${LADDER_OPT:--O0}
  [[ -z "${LADDER_OPT:-}" && "$R" == coremark_matrix ]] && OPT=-Os
  DOMAIN_OPT_LEVEL=$OPT bash "$LAD/build-ladder-domain.sh" "$APP" "$OUT_DIR/${R}.dom"
  # The oracle must see the same kernel-level -D knobs as the domain, or the
  # correctness gate silently compares against a different computation (e.g.
  # -DLADDER_CM_INIT_ONLY makes the rung return N, not the crc). Only -D/-U are
  # forwarded: the rest of DOMAIN_EXTRA_CFLAGS is Capstone-target-specific.
  HOST_DEFS=()
  for tok in ${DOMAIN_EXTRA_CFLAGS:-}; do
    [[ $tok == -D* || $tok == -U* ]] && HOST_DEFS+=("$tok")
  done
  cc -O0 "${HOST_DEFS[@]+"${HOST_DEFS[@]}"}" -I"$LAD" -o "$OUT_DIR/${R}_host" "$HOST"
  "$OUT_DIR/${R}_host" > "$OUT_DIR/${R}.oracle"
  echo "  oracle: $R = $(cat "$OUT_DIR/${R}.oracle")"
done
echo "Done. Artifacts in $OUT_DIR"
