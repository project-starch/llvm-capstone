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
OVERLAY=$ROOT/capstone/caplifive-buildroot/overlay/test-domains

bash "$SCRIPT_DIR/build-sqlite-silicon.sh"
bash "$SCRIPT_DIR/build-sqlite-host.sh"

mkdir -p "$OVERLAY"
cp -f "$CAPSTONE_TMP_ROOT/sqlite-silicon/sqlite_silicon.dom" "$OVERLAY/"
cp -f "$CAPSTONE_TMP_ROOT/sqlite-build/sqlite_host.user"     "$OVERLAY/"
echo "staged into $OVERLAY:"; ls -la "$OVERLAY"

echo
echo "Now rebuild the image so the overlay is picked up:"
echo "  cd $ROOT/capstone/caplifive-buildroot && make build"
echo "then run with an EMPTY share dir and:"
echo "  /test-domains/sqlite_host.user /test-domains/sqlite_silicon.dom"
