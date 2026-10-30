#!/usr/bin/env bash
# Second part of the hfl twin check, after twin-hfl.sh (marker census-hfl/.chain-done): perlbench TRAIN ObjLife first
# (it decides the board line's step and the T2 prediction), then what time leaves. Same deadline guard as
# twin-hfl.sh, with the factor per item (perl train T2's estimate is the campaign twin's measured 80 min).
# Outputs census-hfl/<name>-2 (the -2 avoids the DEFERRED placeholders that made twin-hfl.sh skip these items).
# Log census-hfl/chain2.log; marker census-hfl/.chain2-done.
main() {
  set -u; S=${CAPSTONE_WORK:?}/spec; C=$S/census-hfl; W=${CAPSTONE_WORK:?}/wt-spec-layout; ST=$S/stages-test-hfl
  local L=$C/chain2.log deadline end
  deadline=$(date -d "${HFL_DEADLINE:-2026-10-31 05:00}" +%s)
  end=$(date -d "2026-10-31 06:00" +%s)
  echo "$(date +%T) waiting for twin-hfl.sh" >> $L
  until [ -e $C/.chain-done ]; do [ $(date +%s) -gt $end ] && { echo "$(date +%T) chain 1 never finished" >> $L; touch $C/.chain2-done; exit 1; }; sleep 30; done
  export CAPSTONE_QEMU_LOCK=$HOME/.capstone-locks/qemu.lock
  unset CAPSTONE_MOVC_NULL_SCALAR CAPSTONE_QEMU_DOMAIN_LOCK CAPSTONE_R60
  # name | stage | twin result dir | estimate min | factor percent | step s
  while IFS='|' read -r name st twin est fac step; do
    [ -n "$name" ] || continue
    o=$C/$name-2
    [ -e $o ] && { echo "$(date +%T) $name exists, skipped" >> $L; continue; }
    until flock -n -x "$CAPSTONE_QEMU_LOCK" -c true; do [ $(date +%s) -gt $deadline ] && break; sleep 30; done
    now=$(date +%s)
    if [ $((now + est * 60 * fac / 100)) -gt $deadline ]; then
      echo "$(date +%T) SKIP $name: estimate $est min x$fac% ends after the deadline" >> $L; continue
    fi
    echo "$(date +%T) BEGIN $name (estimate $est min)" >> $L
    taskset -c 16-31,48-63 nice -n 10 python3 $W/capstone/ports/spec-cpu2017/board/run-stage.py run --backend qemu --kit $S/kit-q34 \
      --stage $st --out $o --tree $S/tree-hfl/SPEC17 --step-s $step --twin $twin/result.json > $o.log 2>&1
    echo "$(date +%T) END $name rc=$?" >> $L
    grep -E "^(READ|RUN)" $o.log | sed 's/^/    /' >> $L
  done <<EOF
500.perlbench_r-train-capstone-lifetimes|$S/stages-train-hfl/500.perlbench_r-train-capstone-lifetimes|$S/census-train/500.perlbench_r-train-capstone-lifetimes|80|110|28800
500.perlbench_r-test-capstone-spatial|$ST/500.perlbench_r-test-capstone-spatial|$S/census-heapfix/500.perlbench_r-test-capstone-spatial|59|115|21600
519.lbm_r-train-capstone-lifetimes|$S/stages-train-hfl/519.lbm_r-train-capstone-lifetimes|$S/census-train/stages-train1-519.lbm_r-train|29|125|28800
EOF
  echo "$(date +%T) chain 2 done" >> $L
  touch $C/.chain2-done
}
main "$@"; exit
