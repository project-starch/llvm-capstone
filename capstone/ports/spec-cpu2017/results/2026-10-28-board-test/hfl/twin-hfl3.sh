#!/usr/bin/env bash
# Third part of the hfl twin check: the xz train command-0 stages (input.combined only) at 936M guest RAM, as
# census-train's link-mem936M runs, after twin-hfl2.sh (census-hfl/.chain2-done). Same deadline guard (05:00,
# estimate 31 min per arm from the T2 link run's 45 min for 163.6e9, x1.25). Log census-hfl/chain3.log; marker .chain3-done.
main() {
  set -u; S=$CAPSTONE_WORK/spec; C=$S/census-hfl; W=$CAPSTONE_WORK/wt-spec-layout
  local L=$C/chain3.log deadline end
  deadline=$(date -d "${HFL_DEADLINE:-2026-10-31 05:00}" +%s); end=$(date -d "2026-10-31 06:00" +%s)
  echo "$(date +%T) waiting for twin-hfl2.sh" >> $L
  until [ -e $C/.chain2-done ]; do [ $(date +%s) -gt $end ] && { echo "$(date +%T) chain 2 never finished" >> $L; touch $C/.chain3-done; exit 1; }; sleep 30; done
  export CAPSTONE_QEMU_LOCK=$HOME/.capstone-locks/qemu.lock
  unset CAPSTONE_MOVC_NULL_SCALAR CAPSTONE_QEMU_DOMAIN_LOCK CAPSTONE_R60
  for arm in capstone-spatial capstone-lifetimes; do
    name=557.xz_r-train-$arm-link-c0-mem936M; o=$C/$name
    [ -e $o ] && { echo "$(date +%T) $name exists, skipped" >> $L; continue; }
    until flock -n -x "$CAPSTONE_QEMU_LOCK" -c true; do [ $(date +%s) -gt $deadline ] && break; sleep 30; done
    if [ $(( $(date +%s) + 31 * 75 )) -gt $deadline ]; then echo "$(date +%T) SKIP $name: estimate 31 min x1.25 ends after the deadline" >> $L; continue; fi
    echo "$(date +%T) BEGIN $name" >> $L
    SPEC_CENSUS_MEM=936M taskset -c 16-31,48-63 nice -n 10 python3 $W/capstone/ports/spec-cpu2017/board/run-stage.py run --backend qemu \
      --kit $S/kit-q34 --stage $S/stages-train-hfl/557.xz_r-train-$arm-link-c0 --out $o --tree $S/tree-hfl/SPEC17 --step-s 28800 \
      --twin-inband-max 1048576 --twin $S/census-train/557.xz_r-train-$arm-link-mem936M-c0/result.json > $o.log 2>&1
    echo "$(date +%T) END $name rc=$?" >> $L
    grep -E "^(READ|RUN)" $o.log | sed 's/^/    /' >> $L
  done
  echo "$(date +%T) chain 3 done" >> $L; touch $C/.chain3-done
}
main "$@"; exit
