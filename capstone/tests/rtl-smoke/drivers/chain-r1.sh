#!/bin/bash
# chain-r1.sh -- the working template for "one chain, N boots, one driver": runs board-r1e4.sh once per boot
# over an invocation list (12 list lines per boot), each boot a fresh bake, and writes CHAIN_<TAG>_DONE when
# the last boot has exited. This is how the R1 (E3), E4, F4 and M2 (F5) boots were run; the M2 chain
# (2026-09-15) was exactly this with TAG=F5, BOOTS=4, IMG=build9 fdd3029ff0f96680, LIST=lists/f5-chase.txt.
#
#   CHAIN_TAG=f5 CHAIN_BOOTS=4 R1_IMG=<dom at DOMAIN_BASE_VA=0x410000> R1_HASH=<sha256/16> R1_HOST=<sqlite_host_rr.user> \
#   R1_LIST=<list> R1_QEMU_LOG=<emulator log> \
#   R1_BOOT_TAG=sw8x-f5 R1_BOOT_DESC="..." R1_PREREG="..." [R1_QEMU_GATE='R1 lat'] [R1_QEMU_GATE_MIN=5] \
#   [R1_OUT_TAG=r1f5] [MEMLOCK=<lock file>] bash chain-r1.sh
#
# Launch it in the background with a monitor on its .out file; the chain ends itself (CLAUDE.md: background
# tasks end when their work ends). One board: never two chains, never a chain beside another lane's boot.
# The emulator run of the image must be on record first (R1_QEMU_LOG holds >= R1_QEMU_GATE_MIN lines
# matching R1_QEMU_GATE, and ${CAPSTONE_ARTIFACTS}/qemu-pass/<sha256> exists -- run-r1-qemu.sh writes it).
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
U=${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}/unify; mkdir -p "$U"
TAG=${CHAIN_TAG:?set CHAIN_TAG=<short name, e.g. f5>}; BOOTS=${CHAIN_BOOTS:?set CHAIN_BOOTS=<number of boots>}
: "${R1_IMG:?}" "${R1_HASH:?}" "${R1_LIST:?}" "${R1_QEMU_LOG:?}" "${R1_BOOT_TAG:?}" "${R1_BOOT_DESC:?}" "${R1_PREREG:?}" "${R1_HOST:?set R1_HOST=<path to sqlite_host_rr.user>}"
# every one of board-r1e4.sh's six mandatory variables is checked HERE too, so a chain fails at its own
# first line rather than four boots deep: R1_BOOT is set per iteration below, the other five above.
OUTTAG=${R1_OUT_TAG:-r1$TAG}
cd "$R"
for k in $(seq 1 "$BOOTS"); do
  R1_BOOT=$k R1_OUT_TAG=$OUTTAG R1_BOOT_DESC="$R1_BOOT_DESC (boot $k of $BOOTS)" \
    bash "$R/capstone/tests/rtl-smoke/drivers/board-r1e4.sh" > "$U/board-$OUTTAG-b$k.console" 2>&1
  echo "R1_${TAG^^}_B${k}_EXIT=$?" >> "$U/board-$OUTTAG-b$k.console"
  echo "$(date +%T) $TAG boot $k: $(tail -1 "$U/board-$OUTTAG-b$k.console")"
done
echo "CHAIN_${TAG^^}_DONE"
