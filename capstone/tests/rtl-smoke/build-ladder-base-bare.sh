#!/usr/bin/env bash
# Build the BARE-METAL baseline: the plain-RISC-V half of the overhead measurement
# as an S-mode OpenSBI payload, with no Linux underneath (issue I-2).
#
#   usage: build-ladder-base-bare.sh
#   out:   $OUT_DIR/fw_payload_base_bare.bin   (flashable firmware)
#          $OUT_DIR/ladder_base_bare.elf       (for disassembly)
#
# The firmware is assembled by REUSING the known-good prefix of the existing
# fw_payload -- OpenSBI itself plus the device tree, everything below the payload
# offset -- and substituting our program for the Linux Image that normally sits at
# 0x200000. That avoids rebuilding OpenSBI entirely, which the project notes warn
# is easy to get silently wrong (a fw_payload missing its embedded FDT+kernel fails
# to boot with no diagnostic).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

LAD="$SCRIPT_DIR/../runtime-qemu/silicon-ladder"
CLANG=${CLANG:-$CAPSTONE_CLANG}
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/ladder-base-bare}
OBJ_DIR=$OUT_DIR/obj
mkdir -p "$OBJ_DIR"

# Reference firmware to take the OpenSBI+FDT prefix from.
REF_FW=${REF_FW:-$HOME/capstone-b-artifacts/fw_payload_fpga_up_gpfree.bin}
PAYLOAD_OFF=${PAYLOAD_OFF:-$((0x200000))}
[[ -f "$REF_FW" ]] || { echo "reference firmware missing: $REF_FW" >&2; exit 1; }

SPEC_FILE=${SPEC_FILE:-"$SCRIPT_DIR/ladder-rungs.spec"}
mapfile -t RUNGS < <(grep -vE '^\s*(#|$)' "$SPEC_FILE")

# Generate the rung table so it cannot drift from the spec (the hand-maintained
# table in ladder_base_ctl.c cost a board boot on 2026-07-27 by silently omitting
# two rungs that the build script knew about).
TBL="$OBJ_DIR/ladder_rungs_table.h"
{
  echo "/* GENERATED from ladder-rungs.spec by build-ladder-base-bare.sh -- do not edit. */"
  echo "struct rung { const char *name; unsigned (*fn)(void); };"
  for SPEC in "${RUNGS[@]}"; do
    IFS=: read -r R _ _ _ _ <<<"$SPEC"
    echo "unsigned base_$R(void);"
  done
  echo "static const struct rung RUNGS[] = {"
  for SPEC in "${RUNGS[@]}"; do
    IFS=: read -r R _ _ _ _ <<<"$SPEC"
    echo "  { \"$R\", base_$R },"
  done
  echo "};"
  echo "#define NRUNGS ((int)(sizeof RUNGS / sizeof RUNGS[0]))"
} > "$TBL"

OBJS=()
: > "$OUT_DIR/optlevels.txt"
for SPEC in "${RUNGS[@]}"; do
  # Field 5 (per-rung domain knobs) is deliberately DISCARDED here: DOMAIN_WINDOW
  # and LADDER_NO_RO_COPY are properties of the Capstone gp-captable glue, and the
  # baseline is plain riscv64 with no glue at all. Reading it into a variable the
  # baseline never uses would invite someone to "helpfully" apply it and destroy
  # the comparison.
  IFS=: read -r R HDR FN OPT _ <<<"$SPEC"
  OPT=${LADDER_OPT:-$OPT}
  # Identical flags to the Linux baseline build: same clang, same -O, same target,
  # same -fno-jump-tables. Only the harness around the kernel differs, so a
  # bare-vs-Linux difference is attributable to the environment and nothing else.
  # -mcmodel=medany: the payload lives at 0x80200000, above the low 2 GB that the
  # default medlow model can reach with lui/addi (R_RISCV_HI20 truncation). medany
  # uses auipc/addi instead -- the SAME instruction count, a different opcode pair,
  # so it should not perturb an instruction-count comparison against the medlow
  # Linux baseline. That is an expectation, not an assumption: the two builds are
  # compared rung-by-rung below and any instret difference is reported.
  "$CLANG" -target riscv64-unknown-elf -march=rv64imac -mabi=lp64 -mcmodel=medany \
    -ffreestanding -fno-stack-protector -fno-jump-tables "$OPT" \
    -I"$LAD" -I"$SCRIPT_DIR" \
    -DLADDER_KERNEL_HDR="\"$HDR\"" -DLADDER_COMPUTE="$FN" -DLADDER_EXPORT="base_$R" \
    -c "$SCRIPT_DIR/ladder_base_kern.c" -o "$OBJ_DIR/base_$R.o"
  OBJS+=("$OBJ_DIR/base_$R.o")
  echo "$R $OPT" >> "$OUT_DIR/optlevels.txt"
  [[ "$R" == null ]] || cc -O0 -o "$OUT_DIR/${R}_host" "$LAD/${R}_host.c"
  echo "  kernel: $R ($OPT, $FN)"
done

"$GUEST_CC" -Os -static -nostdlib -ffreestanding -fno-stack-protector \
  -march=rv64imac_zicsr -mabi=lp64 -mcmodel=medany -I"$OBJ_DIR" \
  -DUART_SHIFT=${UART_SHIFT:-2} \
  -T "$SCRIPT_DIR/link-base-bare.ld" \
  -o "$OUT_DIR/ladder_base_bare.elf" \
  "$SCRIPT_DIR/ladder_base_bare_start.S" "$SCRIPT_DIR/ladder_base_bare.c" "${OBJS[@]}"

# Static gate, same as the Linux baseline: NO capability instructions may reach the
# denominator, or the "plain RISC-V" side would be measuring capabilities too.
DIS=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$OUT_DIR/ladder_base_bare.elf")
NCAP=$(grep -cEw 'cjalr|ldc|stc|scc|cincoffset|mrev|csdrop|shrink|delin|revoke' <<<"$DIS" || true)
echo "static: capability-instructions=$NCAP (must be 0)"
[[ "$NCAP" == "0" ]] || { echo "FAIL: capability instructions in the baseline" >&2; exit 1; }

"$CAPSTONE_LLVM_BIN/llvm-objcopy" -O binary "$OUT_DIR/ladder_base_bare.elf" \
  "$OBJ_DIR/payload.bin"

# firmware = [OpenSBI + FDT prefix from REF_FW] ++ [our payload at PAYLOAD_OFF]
head -c "$PAYLOAD_OFF" "$REF_FW" > "$OUT_DIR/fw_payload_base_bare.bin"
cat "$OBJ_DIR/payload.bin" >> "$OUT_DIR/fw_payload_base_bare.bin"

REF_SZ=$(stat -c%s "$REF_FW"); NEW_SZ=$(stat -c%s "$OUT_DIR/fw_payload_base_bare.bin")
echo "payload: $(stat -c%s "$OBJ_DIR/payload.bin") bytes"
echo "firmware: $NEW_SZ bytes (reference was $REF_SZ; JTAG reload time scales with this)"
echo "Built $OUT_DIR/fw_payload_base_bare.bin"
