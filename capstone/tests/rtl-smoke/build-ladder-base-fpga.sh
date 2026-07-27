#!/usr/bin/env bash
# Build the BASELINE half of the silicon-ladder overhead measurement: every rung's
# kernel as plain RISC-V, linked into ONE freestanding Linux binary.
#
#   usage: build-ladder-base-fpga.sh
#   out:   $OUT_DIR/ladder_base_ctl        (one static binary, all 7 rungs)
#
# The point of this build is that it differs from build-ladder-fpga.sh in exactly
# ONE dimension. The domain half compiles <rung>_kernel.h with $CAPSTONE_CLANG at
# $OPT for -target capstone64-unknown-elf with the gp-captable silicon flags; this
# builds the SAME header with the SAME clang at the SAME $OPT for -target
# riscv64-unknown-elf with no capability flags. Using buildroot gcc here instead
# would have made the ratio a measurement of two compilers rather than of the
# capability ABI, so the kernels are deliberately NOT built with GUEST_CC.
#
# The harness (syscalls, printing, _start) IS built with buildroot gcc, matching
# ladder_perf_ctl: it is measurement scaffolding outside the counter brackets, so
# its codegen cannot affect a rung's cycle count. Both sides are rv64imac/lp64, so
# the objects link together.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

LAD="$SCRIPT_DIR/../runtime-qemu/silicon-ladder"
CLANG=${CLANG:-$CAPSTONE_CLANG}
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/ladder-base}
OBJ_DIR=$OUT_DIR/obj
mkdir -p "$OBJ_DIR"

# Rung table comes from the SHARED spec so the capability half cannot drift apart
# from this one -- see ladder-rungs.spec for why that mattered (two silent -O
# mismatches, one of which produced five bogus "silicon failures").
SPEC_FILE=${SPEC_FILE:-"$SCRIPT_DIR/ladder-rungs.spec"}
mapfile -t RUNGS < <(grep -vE '^\s*(#|$)' "$SPEC_FILE")

OBJS=()
: > "$OUT_DIR/optlevels.txt"
for SPEC in "${RUNGS[@]}"; do
  IFS=: read -r R HDR FN OPT <<<"$SPEC"
  # LADDER_OPT overrides the per-rung default. The capability half and the
  # baseline MUST be built at the same level or the ratio measures optimisation
  # rather than capabilities -- beebs_recursion is measured at -O1 on the
  # capability side (the level that clears the -O0 miscompute), so its baseline
  # has to be -O1 as well.
  OPT=${LADDER_OPT:-$OPT}
  # -fno-jump-tables mirrors SILICON_FLAGS: there it is load-bearing (a .rodata
  # jump table is unreachable in a gp-captable domain), here it is not, but the
  # baseline keeps it so the two builds differ only in the target.
  "$CLANG" -target riscv64-unknown-elf -march=rv64imac -mabi=lp64 \
    -ffreestanding -fno-stack-protector -fno-jump-tables "$OPT" \
    -I"$LAD" -I"$SCRIPT_DIR" \
    -DLADDER_KERNEL_HDR="\"$HDR\"" -DLADDER_COMPUTE="$FN" -DLADDER_EXPORT="base_$R" \
    -c "$SCRIPT_DIR/ladder_base_kern.c" -o "$OBJ_DIR/base_$R.o"
  OBJS+=("$OBJ_DIR/base_$R.o")
  # See issue I-1: the runner cross-checks this against the capability half.
  echo "$R $OPT" >> "$OUT_DIR/optlevels.txt"
  # Native oracle beside the object. run_ladder_base_fpga.py hard-fails on a
  # missing <rung>_host, and building them only in the QEMU script meant a
  # board run from a fresh OUT_DIR aborted at the artifact check.
  [[ "$R" == null ]] || cc -O0 -o "$OUT_DIR/${R}_host" "$LAD/${R}_host.c"
  echo "  kernel: $R ($OPT, $FN)"
done

# _zicsr: the harness reads the counter CSRs directly (csrr cycle/time/instret).
# Newer binutils split Zicsr out of the base ISA, so it must be named explicitly;
# it adds no instructions to the kernels, which are already linked as objects.
"$GUEST_CC" -Os -static -no-pie -fno-pie -nostdlib -ffreestanding \
  -fno-stack-protector -march=rv64imac_zicsr -mabi=lp64 \
  -o "$OUT_DIR/ladder_base_ctl" "$SCRIPT_DIR/ladder_base_ctl.c" "${OBJS[@]}"

# Static gate: the baseline must contain NO capability instructions. If any leaked
# in, the "plain RISC-V" denominator would be measuring capabilities too and the
# whole ratio would be meaningless -- so fail the build rather than the analysis.
DIS=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$OUT_DIR/ladder_base_ctl")
NCAP=$(grep -cEw 'cjalr|ldc|stc|scc|cincoffset|mrev|csdrop|shrink|delin|revoke' <<<"$DIS" || true)
echo "static: capability-instructions=$NCAP (must be 0)"
[[ "$NCAP" == "0" ]] || { echo "FAIL: capability instructions in the baseline" >&2; exit 1; }

echo "Built $OUT_DIR/ladder_base_ctl ($(stat -c%s "$OUT_DIR/ladder_base_ctl") bytes)"
