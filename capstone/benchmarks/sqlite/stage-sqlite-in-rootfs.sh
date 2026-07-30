#!/usr/bin/env bash
# Stage the silicon SQLite domain into the buildroot rootfs overlay (stage S7 delivery).
#
# WHY BAKING, NOT TRANSFER. The domain is ~1.6 MB. The board's file transfer is
# base64 over the UART console, and at its pacing floor (16 chars per socket.io emit,
# 0.02 s) that is >= 63 minutes for this image -- before counting network round-trip,
# which dominates. Even gzipped it is >= 17 minutes. Unusable for iteration.
#
# The firmware already goes to the board over JTAG in ~2 min for 15 MB, so carrying
# SQLite inside it costs essentially nothing extra. It also needs NO reserved RAM
# address, so it never touches the "never guess a RAM address" rule that the
# load_image route would.
#
# Proven under QEMU 2026-07-28: with an EMPTY 9p share, `/test-domains/sqlite_host.user
# /test-domains/sqlite_silicon.dom` produces all five success markers.
#
# For the FPGA the same overlay ends up in the fw_payload's initramfs -- rebuild the
# firmware with the recipe in memory `project_fpga_fw_payload_build_recipe` (build from
# caplifive-SYSTEM fpga/ariane, and it MUST embed FDT+kernel or the board boots silently
# to nothing).
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

# BOTH overlays, on purpose. QEMU boots from caplifive-BUILDROOT; the FPGA firmware is
# built from caplifive-SYSTEM (BR2_ROOTFS_OVERLAY in configs/fpga_defconfig). Staging
# only the first is why the FPGA image carried a domain a DAY OLDER than the build on
# 2026-07-29: it predated the delin fix and the output_text fix, so a board session
# would have re-measured bugs that were already fixed, with nothing in the log to say so.
# A stale artifact frozen inside an image is worse than a stale file on disk, because it
# survives rebuilds invisibly.
OVERLAYS=(
  "$ROOT/capstone/caplifive-buildroot/overlay/test-domains"          # QEMU
  "$ROOT/capstone/caplifive-system/sw/buildroot/overlay/test-domains" # FPGA
)

bash "$SCRIPT_DIR/build-sqlite-silicon.sh"
bash "$SCRIPT_DIR/build-sqlite-host.sh"

SRC_DOM=$CAPSTONE_TMP_ROOT/sqlite-silicon/sqlite_silicon.dom
SRC_USR=$CAPSTONE_TMP_ROOT/sqlite-build/sqlite_host.user
for f in "$SRC_DOM" "$SRC_USR"; do
  [[ -f "$f" ]] || { echo "missing build output: $f" >&2; exit 1; }
done

# REFUSE to stage a diagnostic build as the canonical domain.
#
# This is not hypothetical tidiness. On 2026-07-30 an INTERP_BUILD_LIMIT=900 build was
# staged here and baked into firmware; it carves the cap table at full size but fills only
# the first 900 of 1059 slots, so the first `ldc` from an unfilled slot yields an untagged
# capability and the `stc` through it stalls the pipeline with no trap. That produced a
# reproducible "SQLite hangs on silicon" signature -- pc frozen, mcause=0 -- which was
# investigated as a hardware defect for hours. It was our own diagnostic knob.
#
# The clamp is invisible in the descriptor (`count` still reads 1059) because it is applied
# to the glue's loop counter at run time, so this checks the ENTRY CODE of the artifact
# about to be staged. Variants keep their own filenames and are unaffected.
if ! python3 "$SCRIPT_DIR/gp-carve-count.py" "$SRC_DOM" "${CARVE_BUDGET:-1000}"; then
  echo "" >&2
  echo "REFUSING TO STAGE $SRC_DOM as the canonical sqlite_silicon.dom." >&2
  echo "Rebuild with INTERP_EXTRA_CFLAGS unset, or stage a variant under its own name." >&2
  echo "Override deliberately with STAGE_ALLOW_DIAGNOSTIC=1 (do not use for a real run)." >&2
  [[ "${STAGE_ALLOW_DIAGNOSTIC:-0}" == "1" ]] || exit 1
  echo "STAGE_ALLOW_DIAGNOSTIC=1 set -- proceeding anyway." >&2
fi

for OVERLAY in "${OVERLAYS[@]}"; do
  mkdir -p "$OVERLAY"
  cp -f "$SRC_DOM" "$OVERLAY/"
  cp -f "$SRC_USR" "$OVERLAY/"
  echo "staged into $OVERLAY"
done

# Record what was staged so a later run can prove the image matches the build rather
# than assuming it. Compare against the live build before trusting any board result.
echo "  sqlite_silicon.dom sha256 = $(sha256sum "$SRC_DOM" | cut -c1-32)"
echo "  sqlite_host.user   sha256 = $(sha256sum "$SRC_USR" | cut -c1-32)"

echo
echo "Now REBUILD BOTH images so the overlays are picked up:"
echo "  QEMU: cd $ROOT/capstone/caplifive-buildroot && make build"
echo "  FPGA: cd $ROOT/capstone/caplifive-system/sw/buildroot && \\"
echo "        make build LINUX_PAYLOAD=1 A=opensbi-rebuild CAPSTONE_CC_PATH=\$(realpath ../../../capstone-c)"
echo "        (LINUX_PAYLOAD=1 is REQUIRED -- without it the payload is not embedded and"
echo "         you get a ~2 MB image that boots to nothing and carries no initramfs)"
echo "then run with an EMPTY share dir and:"
echo "  /test-domains/sqlite_host.user /test-domains/sqlite_silicon.dom"
