#!/bin/bash
# The pre-registered bare sessions: call-retpc (control) first, then CAPSTL_SET=hot (intloop-q64, capstl-q64/-q47/-q16)
# or CAPSTL_SET=evict (the cache-miss set, PREREG.md).
# One power cycle per image, so a wedge in one costs nothing in the next.
main() {
  set -u
  local F=$(cd "$(dirname "$0")" && pwd) OUT=${CAPSTL_OUT:-/tmp/capstone/capstl-run}
  local L=$F/../sup-bare-2026-10-03-ladder V1=$F/../sup-bare-2026-10-02/images
  export FPGA_URL="$(cat "${CAPSTONE_FPGA_URL_FILE:-$HOME/.claude-kisp/secrets/fpga-console-url}")"
  # the runner HARD-STOPS unless this equals the resident bitstream's name; set CAPSTL_BITSTREAM after a reflash
  export FPGA_BITSTREAM=${CAPSTL_BITSTREAM:-caplifive_supcall_36a641e0b.bit}
  mkdir -p $OUT
  local jobs=("control-call-retpc|$V1/call-retpc.bin|80003800")
  if [ "${CAPSTL_SET:-hot}" = hot ]; then
    for n in intloop-q64 capstl-q64 capstl-q47 capstl-q16; do jobs+=("$n|$F/images/$n.bin|80003c40"); done
  elif [ "$CAPSTL_SET" = s16stores ]; then   # the RTL lane's sup-s16-stores.S, bare
    for n in s16st-n4-r1 s16st-n4-r3 s16st-n4-r3-fence s16st-n32-r1 s16st-esc-n8; do jobs+=("$n|$F/images/$n.bin|$(cat $F/images/$n.rec)"); done
  elif [ "$CAPSTL_SET" = s16next ]; then   # the PLAIN twins, the escape discriminator, and an esc-n8 repeat
    for n in s16st-plain-n32-r1 s16st-plain-n4-r3 s16st-plain-n24-r3 s16st-plain-n32-r3 s16st-esc-n8-retfence s16st-esc-n8; do jobs+=("$n|$F/images/$n.bin|$(cat $F/images/$n.rec)"); done
  elif [ "$CAPSTL_SET" = s16fence ]; then   # a fence before the CALL and the domain's RETURN, on four twins that hung
    for n in s16st-plain-n32-r1-fence s16st-plain-n4-r3-fence2 s16st-n4-r3-fence2 s16st-plain-n32-r3-fence2; do jobs+=("$n|$F/images/$n.bin|$(cat $F/images/$n.rec)"); done
  elif [ "$CAPSTL_SET" = accept715 ]; then   # the S-16/S-17 fix bitstream (capstone-ariane 715bdd1fe): every S-16 arm without a fence, S-17 last
    for n in armdep-nt-d16-q64 s16sd24 s16st-n4-r1 s16st-n4-r3 s16st-n4-r3-fence s16st-n32-r1 s16st-esc-n8 s16st-esc-n8-retfence \
             s16st-plain-n32-r1 s16st-plain-n4-r3 s16st-plain-n24-r3 s16st-plain-n32-r3 s16st-esc-n8-r1 s16st-esc-n32-r1 s16st-esc-n64-r1 \
             arm12-ld-q64 arm12-ldc-q64; do
      jobs+=("$n|$F/images/$n.bin|$(cat $F/images/$n.rec)"); done
  elif [ "$CAPSTL_SET" = s16escr1 ]; then   # the RTL lane's three one-round escape arms exactly
    for n in s16st-esc-n8-r1 s16st-esc-n32-r1 s16st-esc-n64-r1; do jobs+=("$n|$F/images/$n.bin|$(cat $F/images/$n.rec)"); done
  elif [ "$CAPSTL_SET" = s10b ]; then   # S-10b's primed route on the R-29/S-10b fix (PREREG.md "s10b")
    jobs+=("s10b-primed|$F/images/s10b-primed.bin|$(cat $F/images/s10b-primed.rec)")
  elif [ "$CAPSTL_SET" = r51 ]; then   # R-51: the PC capability before and after a RETURN-based yield (PREREG.md "r51")
    for n in r51-before r51-after; do jobs+=("$n|$F/images/$n.bin|$(cat $F/images/$n.rec)"); done
  elif [ "$CAPSTL_SET" = s17rep ]; then   # S-17's matched pair again (repeats on 715bdd1fe)
    for n in arm12-ldc-q64 arm12-ld-q64; do jobs+=("$n|$F/images/$n.bin|$(cat $F/images/$n.rec)"); done
  elif [ "$CAPSTL_SET" = precall ]; then   # the S-16 workaround candidate, and the stc18 rerun
    for n in s16pre-nt s16pre-stc24 s16stc18; do jobs+=("$n|$F/images/$n.bin|$(cat $F/images/$n.rec)"); done
  elif [ "$CAPSTL_SET" = deep ]; then   # extended apertures on S-16/S-17, the slot-0 control, the store-count dose-response
    for n in s16stc12 s16stc18 s16stc24 s16sd24 armdep-nt-d16-q64 arm12-ldc-q64 s16nop-nt; do jobs+=("$n|$F/images/$n.bin|$(cat $F/images/$n.rec)"); done
  elif [ "$CAPSTL_SET" = s16parts3 ]; then   # S-16: three-part combinations with part 8
    for n in s16p11-nt s16p13-nt s16p14-nt; do jobs+=("$n|$F/images/$n.bin|$(cat $F/images/$n.rec)"); done
  elif [ "$CAPSTL_SET" = s16parts ]; then   # S-16 bisect on the fast repro
    for n in s16p8-nt s16p9-nt s16p10-nt s16p12-nt; do jobs+=("$n|$F/images/$n.bin|$(cat $F/images/$n.rec)"); done
  elif [ "$CAPSTL_SET" = notrace ]; then   # armdep without the per-resume trace prints (do they protect?)
    jobs+=("armdep-nt-d16-q64|$F/images/armdep-nt-d16-q64.bin|$(cat $F/images/armdep-nt-d16-q64.rec)")
  elif [ "$CAPSTL_SET" = wedge ]; then   # the three hang types again, read with the wedge apertures (run_sup_bare_wedge.py)
    for n in arm12-ldc-q64 armdep-d16-q64; do jobs+=("$n|$F/images/$n.bin|$(cat $F/images/$n.rec)"); done
    jobs+=("mswapfix-plain-noploop|$F/images/mswapfix-plain-noploop.bin|80015000")
  elif [ "$CAPSTL_SET" = dyn ]; then   # LDC vs ld after the post-CALL ccsrrw; the dependent arm twice with dots per 16
    for n in arm12-ld-q64 arm12-ldc-q64 armdep-d16-q64; do jobs+=("$n|$F/images/$n.bin|$(cat $F/images/$n.rec)"); done
    jobs+=("armdep-d16-q64-rep2|$F/images/armdep-d16-q64.bin|$(cat $F/images/armdep-d16-q64.rec)")
  elif [ "$CAPSTL_SET" = armed ]; then   # armed, many resumes: sp-dependent load +/- fence; print / csrr after the CALL
    for n in armdep-q64 armdep-fence-q64 arm-print-q64 arm-csrr-q64; do jobs+=("$n|$F/images/$n.bin|$(cat $F/images/$n.rec)"); done
  elif [ "$CAPSTL_SET" = postcall ]; then   # the hang repeated, a fence / 8 nops after the CALL, pairs without K, armed+fence
    jobs+=("mswapfix-plain-noploop|$F/images/mswapfix-plain-noploop.bin|80015000")
    for n in swap15-plain-fence swap15-plain-nop8 swappart9-plain swappart10-plain swappart12-plain swap15-armed-fence; do
      jobs+=("$n|$F/images/$n.bin|$(cat $F/images/$n.rec)"); done
  elif [ "$CAPSTL_SET" = swappairs ]; then   # pairs with part 8, and the full macro, each with 'K' after the CALL
    for p in 9k 10k 12k; do jobs+=("swappart$p-plain|$F/images/swappart$p-plain.bin|80014000"); done
    jobs+=("swappart15k-plain|$F/images/swappart15k-plain.bin|80015000")
  elif [ "$CAPSTL_SET" = swapparts3 ]; then   # the three-part combinations
    jobs+=("swappart7-plain|$F/images/swappart7-plain.bin|80015000" "swappart11-plain|$F/images/swappart11-plain.bin|80014000")
    jobs+=("swappart13-plain|$F/images/swappart13-plain.bin|80015000" "swappart14-plain|$F/images/swappart14-plain.bin|80014000")
  elif [ "$CAPSTL_SET" = swapparts ]; then   # bisect the swap on the plain control: none, then one part each
    for p in 0 1 2 4 8; do jobs+=("swappart$p-plain|$F/images/swappart$p-plain.bin|80014000"); done
  elif [ "$CAPSTL_SET" = mswapfix ]; then   # the clash-fixed swap: its plain control, then one armed arm
    for n in mswapfix-plain-noploop mswapfix-noploop-q64; do jobs+=("$n|$F/images/$n.bin|80015000"); done
  elif [ "$CAPSTL_SET" = dbg ]; then   # the latency probe, then the MSWAP debug set (its own control first)
    jobs+=("latprobe-capstl-q64|$F/images/latprobe-capstl-q64.bin|80003c40")
    for n in latprobe-mevict-capstl-q64 latprobe-evict-capstl-q100k; do jobs+=("$n|$F/images/$n.bin|80014000"); done
    for n in mswapdbg-plain-noploop mswapdbg-noploop-q64 mswapdbg-capstl-q64; do jobs+=("$n|$F/images/$n.bin|80015000"); done
  elif [ "$CAPSTL_SET" = latprobe ]; then   # the eviction positive control
    jobs+=("latprobe-capstl-q64|$F/images/latprobe-capstl-q64.bin|80003c40")
    for n in latprobe-mevict-capstl-q64 latprobe-evict-capstl-q100k; do jobs+=("$n|$F/images/$n.bin|80014000"); done
  elif [ "$CAPSTL_SET" = mswap ]; then   # the FPGA monitor's __domcallsaves sequence around every CALL
    for n in mswap-capstl-q64 mswap-evict-noploop-q100k mswap-evict-capstl-q100k mswap-mevict-capstl-q64; do
      jobs+=("$n|$F/images/$n.bin|80015000"); done
  else   # CAPSTL_SET=evict: the cache-miss set (board_rec moves with the 64 KiB buffer)
    for n in evict-noploop-q100k evict-intloop-q100k evict-capstl-q100k evict-capstl-q20k mevict-noploop-q64 mevict-capstl-q64; do
      jobs+=("$n|$F/images/$n.bin|80014000"); done
  fi
  for j in "${jobs[@]}"; do
    IFS='|' read -r tag img rec <<< "$j"
    echo "$tag START $(date +%H:%M:%S) $(sha256sum $img | cut -c1-16)" >> $OUT/SEQ
    SUP_IMG=$img SUP_OUT=$OUT/runs/$tag SUP_REC_ADDR=$rec SUP_TIMEOUT=${CAPSTL_TIMEOUT:-90} \
      taskset -c 0-7,32-39 nice -n 10 python3 ${SUP_RUNNER:-$L/run_sup_bare.py} > $OUT/runs-$tag.log 2>&1
    echo "$tag END rc=$? $(date +%H:%M:%S)" >> $OUT/SEQ
    sleep 8
  done
}
main "$@"; exit $?
