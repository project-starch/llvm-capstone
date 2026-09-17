#!/bin/bash
# chain-leaf.sh -- the node-table capacity probe, FIVE boots over board-r1e4.sh, one M1_LEAF/M1_TOUCH
# geometry per boot, six repetitions of the drop pattern each.
#
# WHY. Re-reading the five captured M1_LIVE geometries per snapshot shows take_cyc/n is a function of
# cumulative ALLOCATIONS and not of the preceding revoke's walk length: at a walk of 254 dead nodes it
# reads 67 cycles at M1_LIVE=2 and 130 at 64, while at alloc 512 every geometry reads 67-68. The floor
# is 66.7-67.5 everywhere and the rise begins at alloc 1792-2048 = 28.0-32.0 KiB of 16-byte node table.
# The hardware side gives the arithmetic from the configured geometry: the D-cache is 32,768 B, 8-way,
# 16-byte lines, and the revocation node table lies inside the cacheable window, so ONE NODE IS ONE LINE
# and the cache holds exactly 2,048 nodes = 32.0 KiB of table. This chain asks whether that boundary is
# really capacity, by spending the same cache on something else and seeing the knee move.
#
#   LEAF_HOST=<sqlite_host_rr.user> LEAF_BASE=<dir holding p128/ p8k/ p16k/ p24k/ p24k-t64/> bash chain-leaf.sh
#
# Every image is linked at DOMAIN_BASE_VA=0x410000 (the k800 control owns the build default 0x10000 and
# preflight C15 refuses a collision) and carries its own qemu-pass record: five geometries are five
# images, and an image inherits nothing from its neighbour.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
U=${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}/unify; mkdir -p "$U"
L=$R/capstone/tests/rtl-smoke/drivers/lists
: "${LEAF_HOST:?set LEAF_HOST=<path to sqlite_host_rr.user, the readback host>}" "${LEAF_BASE:?}"

PREREG="PRIMARY: knee_alloc, the first snapshot whose take_cyc/n exceeds 1.25x the arm's own alloc<=512 floor. If the node table and the pool share the 32 KiB D-cache then knee_alloc = (K - resident)/16 with K fitted in 28672..32768, NOT assumed: predicted 1792 at resident 128, 1280 at 8192, 768 at 16384, 256 at 24576 -- a four-point LINE with a slope, not a step, and a touch-volume account has no reason to produce that constant. SEPARATOR (p24k-t64): carves 24,576 bytes and leaves 128 resident, so it has the address spread of the largest arm and the residency of the smallest; a knee tracking RESIDENCY puts it with p128 near 1792, a knee tracking the CARVE puts it with p24k near 256. It is the arm that decides whether the sweep measured footprint or merely address range. NULL, and it is a real outcome: the knee does not move across all five, the node path does not share capacity with data, and the 28-32 KiB agreement is coincidence. CONTROLS: the alloc<=512 floor must read 66.7-67.5 in EVERY arm -- a floor that moves means the arms differ in something besides the knee and the comparison is void, which is the reading that would stop this chain; leaf/touch/pool/resident on the start line must equal the built geometry, since five arms differ only in defines and a transcript that does not state its own geometry can be told apart only by image hash; k800 retval=4 first and last. CAPACITY: 6 x 4,096 = 24,576 nodes, 37 percent of the 65,536-node pool, and six region-bearing invocations against the cap of twelve. NOT PRIMARY TIMING: six repetitions inside ONE boot per arm, so METHODS' five-runs-across-three-boots is not met and this is calibration, labelled as such."

cd "$R"
k=0
for n in p128 p8k p16k p24k p24k-t64; do
  k=$((k+1))
  IMG=$LEAF_BASE/$n/r1_slots_pools.dom
  [ -f "$IMG" ] || { echo "missing image for $n: $IMG"; exit 1; }
  HASH=$(sha256sum "$IMG" | cut -c1-16)
  case $n in
    p128)     DESC="node-table capacity probe, resident 128 B: the reference arm, the knee where the existing captures put it";;
    p8k)      DESC="node-table capacity probe, resident 8 KiB: one quarter of the D-cache spent on pool data";;
    p16k)     DESC="node-table capacity probe, resident 16 KiB: half the D-cache spent on pool data";;
    p24k)     DESC="node-table capacity probe, resident 24 KiB: three quarters of the D-cache spent on pool data";;
    p24k-t64) DESC="node-table capacity probe, SEPARATOR: 24 KiB carved and 128 B resident -- the spread of the largest arm, the residency of the smallest";;
  esac
  R1_BOOT=1 R1_OUT_TAG=leaf-$n R1_BOOT_TAG=sw8x-lf R1_BOOT_DESC="$DESC" R1_PREREG="$PREREG" \
  R1_IMG=$IMG R1_HASH=$HASH R1_HOST=$LEAF_HOST R1_LIST=$L/m1-leaf-$n.txt \
  R1_QEMU_LOG=$LEAF_BASE/qemu-$n/qemu-drop/boot.log R1_QEMU_GATE='R1 m1 end' R1_QEMU_GATE_MIN=1 \
    bash "$R/capstone/tests/rtl-smoke/drivers/board-r1e4.sh" > "$U/board-leaf-$n.console" 2>&1
  echo "LEAF_${n}_EXIT=$?" >> "$U/board-leaf-$n.console"
  echo "$(date +%T) leaf boot $k/5 ($n, $HASH): $(tail -1 "$U/board-leaf-$n.console")"
done
echo CHAIN_LEAF_DONE
