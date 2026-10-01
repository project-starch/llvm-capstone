#!/bin/bash
# One boot of the 2026-10-01 M1 campaigns on R-43 v2, by tag. Pre-registrations: the paper bundles
# experiments/results/M1/2026-10-01-v2-series/ and .../2026-10-01-v2-turnover-witness/ (branch board/silicon-evidence).
# Runs board-r1e4.sh from the MAIN clone (it needs the buildroot submodules), with this lane's lists.
#   bash board-m1v2.sh <tag>      (m1v2s-1..5: the series; m1tw-1..3 LCC scan, m1tw-4..6 single probes)
set -u
LISTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lists
MAIN=${CAPSTONE_MAIN_CLONE:-$HOME/dev/llvm-capstone}; A=${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}
Q=${M1V2_QEMU_DIR:-$A/m1v2-2026-10-01/qemu}   # <image>-<shape>/boot.log, the emulator run each boot cites
tag=${1:?boot tag, e.g. m1v2s-1}
case $tag in
  m1v2s-1)         img=/tmp/capstone/r1-campaign/r1_slots_pools.dom; h=1b7a04fe237e1580; list=m1-order9-ctl.txt; ql=$Q/1b7a04fe-drop/boot.log
                   pre="BRIDGE: drop and ring stop=target alloc=655320, 159 snapshots, minted-revoked=31; on 054cea69b this image read take 72.029/72.006 give 103.058/101.749; v2 read take ~94 on 9b24aa31 -- a take shift > 0.1 on THIS image is the bitstream; k800 retval=4 twice" ;;
  m1v2s-5)         img=$A/m1-repro/byteid5/r1_slots_pools.dom; h=249cfda958f22f16; list=m1-maxret43296.txt; ql=$Q/249cfda9-pressure/boot.log
                   pre="BRIDGE: pressure stop=buffer alloc=43297 (054cea69b take 72.010 give 71.052); release released at 43290, end 51948 (phase 1 take 72.175 give 71.827; phase 2 take 86.943 give 93.085); k800 twice" ;;
  m1v2s-2|m1v2s-3|m1v2s-4)
                   img=$A/m1v2-2026-10-01/rt/r1_slots_pools.dom; h=eb3ed1e3f8ac4d82; ql=$Q/eb3ed1e3-pressure/boot.log
                   common="k800 retval=4; drop and ring stop=target alloc=655320, 159 snapshots; pressure stop=buffer alloc=43297 x4; release released 43290 end 51948; minted-revoked=31 every snapshot; take image-dependent (72 or 94); live controls: ring and pressure age0 read ok live=1 is_live_data=1, pressure age2 write ok live=1 via_live_alias=165, pressure age2 read ok live=1 is_live_data=1"
                   case $tag in
                     m1v2s-2) list=m1v2-rt-series-rstale0.txt; pre="$common; LAST stale READ age 0: no 'probe read ok live=0', cause 25 at image+0x4868, record LATCHED" ;;
                     m1v2s-3) list=m1v2-rt-series-wstale0.txt; pre="$common; LAST stale WRITE age 0: no 'probe write ok live=0', cause 25 at image+0x48cc, record LATCHED" ;;
                     m1v2s-4) list=m1v2-rt-series-rstale2.txt; pre="$common; LAST stale READ age 2: no 'probe read ok live=0', cause 25 at image+0x4868, record LATCHED" ;;
                   esac ;;
  m1tw-1|m1tw-2|m1tw-3)
                   img=$A/m1v2-2026-10-01/lcc/r1_slots_pools.dom; h=42bdc05826fe5f4a; ql=$Q/42bdc058-pressure/boot.log; list=m1v2-lcc-scan.txt
                   pre="LCC scan, returns: k800 twice; every arm: old_valid=0 and live_valid=queries; drop and ring: queries=655320 first_queries=40958 first_valid=0 (a 14-bit wrap would read 2, first at alloc 524257); pressure queries=43296 first_queries=2706; release queries=51948 first_queries=3247" ;;
  m1tw-4|m1tw-5|m1tw-6)
                   img=$A/m1v2-2026-10-01/rt/r1_slots_pools.dom; h=eb3ed1e3f8ac4d82; ql=$Q/eb3ed1e3-pressure/boot.log
                   case $tag in
                     m1tw-4) list=m1v2-rt-p2read.txt;  pre="alone after k800, stale READ age 2 refused: cause 25 at image+0x4868, tval[9:0]=0x3c0 (BLOCKING), record LATCHED with id generation 1352 and index head-1, serving_idx head-16" ;;
                     m1tw-6) list=m1v2-rt-ringread.txt; pre="alone after k800, one 10C ring arm then a stale READ of slot 0 previous alias: cause 25 at image+0x4868; refused generation 4094; RETIREMENT reads head=B+69 serving=B+44 index=B+53, a 14-bit WRAP reads head=B+37 serving=B+12 index=B+21, with B=head(m1tw-4)-37" ;;
                     m1tw-5) list=m1v2-rt-p2write.txt; pre="alone after k800, stale WRITE age 2 refused: cause 25 at image+0x48cc, tval[9:0]=0x3c0 (BLOCKING), id generation 1352 index head-1, serving head-16" ;;
                   esac ;;
  *) echo "unknown tag $tag" >&2; exit 2 ;;
esac
[ -f "$LISTS/$list" ] || { echo "no list $LISTS/$list" >&2; exit 2; }
[ -f "$ql" ] || { echo "no emulator log $ql" >&2; exit 2; }
cd "$MAIN" || exit 2
FPGA_BITSTREAM=caplifive_r43_8f6a0af.bit REFUSAL_RECORD=1 \
R1_IMG=$img R1_HASH=$h R1_HOST=$A/h1-r42/sqlite_host_rr.2c9e82d1.user R1_LIST=$LISTS/$list R1_BOOT=1 \
R1_QEMU_LOG=$ql R1_QEMU_GATE='R1 m1 end' R1_QEMU_GATE_MIN=1 \
R1_OUT_TAG=$tag R1_BOOT_TAG=$tag R1_BOOT_DESC="M1 on R-43 v2, $list on $h" R1_PREREG="$pre" \
  bash "$MAIN/capstone/tests/rtl-smoke/drivers/board-r1e4.sh"
