#!/usr/bin/env bash
# The Plain train stages on the QEMU twin (kit-q34, 936M guest RAM as the board's), one after another under the shared
# QEMU lock: rc 0 and specdiff for every command, and the output hashes, before any Plain stage reaches a board.
# Log census-plain/chain.log; marker .chain-done.
main() {
  set -u; local S=/mnt/hdd/$USER/capstone/spec C=/mnt/hdd/$USER/capstone/spec/census-plain W=/mnt/hdd/$USER/capstone/wt-spec
  local L=$C/chain.log name o
  export CAPSTONE_QEMU_LOCK=$HOME/.capstone-locks/qemu.lock
  unset CAPSTONE_MOVC_NULL_SCALAR CAPSTONE_QEMU_DOMAIN_LOCK CAPSTONE_R60
  for name in 519.lbm_r-train-plain-rv64 505.mcf_r-train-plain-rv64 557.xz_r-train-plain-rv64-link-c0 544.nab_r-train-plain-rv64 \
              538.imagick_r-train-plain-rv64 500.perlbench_r-train-plain-rv64 525.x264_r-train-plain-rv64; do
    o=$C/$name
    [ -e $o ] && { echo "$(date +%T) $name exists, skipped" >> $L; continue; }
    echo "$(date +%T) BEGIN $name" >> $L
    SPEC_CENSUS_MEM=936M taskset -c 16-31,48-63 nice -n 10 python3 $W/capstone/ports/spec-cpu2017/board/run-stage.py run --backend qemu \
      --kit $S/kit-q34 --stage $S/stages-train-plain/$name --out $o --tree $S/tree-plain/SPEC17 --step-s 28800 \
      --twin-inband-max 1048576 > $o.log 2>&1
    echo "$(date +%T) END $name rc=$?" >> $L
    grep -E "^(READ|RUN|   )" $o.log | sed 's/^/    /' >> $L
  done
  echo "$(date +%T) chain done" >> $L; touch $C/.chain-done
}
main "$@"; exit
