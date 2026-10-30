#!/usr/bin/env bash
# The heapfix + heap-last (hfl) images on the QEMU twin (kit-q34, the campaign twin), serially under the shared
# qemu.lock. Test size for all 7 benchmarks, then lbm and mcf train. Each run is compared by output hash with the
# campaign twin's run of the same stage (--twin): census-q34 for test, census-heapfix for perlbench test (the campaign
# images fail two perl sub-tests by the heap defect, the heapfix images do not), census-train for lbm/mcf train.
# DEADLINE GUARD: a run starts only if now + its estimate (from the campaign twin's duration, x1.25) ends before
# HFL_DEADLINE (default 05:00 on 2026-10-31), so no run holds the shared lock past it. Log census-hfl/chain.log;
# marker census-hfl/.chain-done.
main() {
  set -u; S=${CAPSTONE_WORK:?}/spec; C=$S/census-hfl; W=${CAPSTONE_WORK:?}/wt-spec-layout; ST=$S/stages-test-hfl
  local L=$C/chain.log deadline
  mkdir -p $C
  deadline=$(date -d "${HFL_DEADLINE:-2026-10-31 05:00}" +%s)
  export CAPSTONE_QEMU_LOCK=$HOME/.capstone-locks/qemu.lock
  unset CAPSTONE_MOVC_NULL_SCALAR CAPSTONE_QEMU_DOMAIN_LOCK CAPSTONE_R60
  # name | stage | twin result dir | estimate min | step s
  while IFS='|' read -r name st twin est step; do
    [ -n "$name" ] || continue
    o=$C/$name
    [ -e $o ] && { echo "$(date +%T) $name exists, skipped" >> $L; continue; }
    # Wait here while another lane holds the lock, so the deadline is judged when the run can actually start.
    until flock -n -x "$CAPSTONE_QEMU_LOCK" -c true; do [ $(date +%s) -gt $deadline ] && break; sleep 30; done
    now=$(date +%s)
    if [ $((now + est * 75)) -gt $deadline ]; then
      echo "$(date +%T) SKIP $name: estimate $est min x1.25 ends after the deadline" >> $L; continue
    fi
    echo "$(date +%T) BEGIN $name (estimate $est min)" >> $L
    taskset -c 16-31,48-63 nice -n 10 python3 $W/capstone/ports/spec-cpu2017/board/run-stage.py run --backend qemu --kit $S/kit-q34 \
      --stage $st --out $o --tree $S/tree-hfl/SPEC17 --step-s $step --twin $twin/result.json > $o.log 2>&1
    echo "$(date +%T) END $name rc=$?" >> $L
    grep -E "^(READ|RUN)" $o.log | sed 's/^/    /' >> $L
  done <<EOF
519.lbm_r-test|$ST/519.lbm_r-test|$S/census-q34/519.lbm_r-test|5|21600
538.imagick_r-test|$ST/538.imagick_r-test|$S/census-q34/538.imagick_r-test|2|21600
544.nab_r-test|$ST/544.nab_r-test|$S/census-q34/544.nab_r-test|5|21600
505.mcf_r-test|$ST/505.mcf_r-test|$S/census-q34/505.mcf_r-test|24|21600
557.xz_r-test|$ST/557.xz_r-test|$S/census-q34/557.xz_r-test|26|21600
500.perlbench_r-test-capstone-lifetimes|$ST/500.perlbench_r-test-capstone-lifetimes|$S/census-heapfix/500.perlbench_r-test-capstone-lifetimes|98|21600
525.x264_r-test-capstone-lifetimes|$ST/525.x264_r-test-capstone-lifetimes|$S/census-q34/525.x264_r-test|131|21600
500.perlbench_r-test-capstone-spatial|$ST/500.perlbench_r-test-capstone-spatial|$S/census-heapfix/500.perlbench_r-test-capstone-spatial|59|21600
519.lbm_r-train-capstone-lifetimes|$S/stages-train-hfl/519.lbm_r-train-capstone-lifetimes|$S/census-train/stages-train1-519.lbm_r-train|29|28800
525.x264_r-test-capstone-spatial|$ST/525.x264_r-test-capstone-spatial|$S/census-q34/525.x264_r-test|131|21600
505.mcf_r-train-capstone-lifetimes|$S/stages-train-hfl/505.mcf_r-train-capstone-lifetimes|$S/census-train/stages-train1-505.mcf_r-train|61|28800
EOF
  echo "$(date +%T) chain done" >> $L
  touch $C/.chain-done
}
main "$@"; exit
