#!/bin/bash
# twin-renew.sh ARM (nested lane, 2026-11-26): perlbench's Line B CRENEW variant on the trap-route twin, as twin-ports.sh
# runs the "Full + ports" stages (qemu-crv, Image.twin = 749fed26, the keep-tags fw_jump.elf, sublet.dom_wide_grants=1,
# BOOT_MEM=962M, /init = init-twin), one perlbench stage per boot, with oneop's opcount plugin (renew/opcount.so) through
# renew/boot-plugin.py, which passes -plugin and lets QEMU exit by init's poweroff so that the plugin writes its file.
# ARM:
#   cal-stock  B5TL1's stock makerand.pl stage (SP-B5TL1-g4mk's)            } the plugin's calibration, before the PREREG
#   cal-full   the port stage (ports/stages/500.perlbench_r, perl-full)      }
#   full2      cal-full again, after the PREREG: the same-day control
#   ren1, ren2 the variant's stage (renew/stages/fullren, perl-fullren), twice
#   fx         the fixture stage (renew/stage-fx), no plugin
# Ends with a TWIN-DONE line in every case.
main() {
  set -euo pipefail
  local A=${1:?arm} X=<hdd>/capstone/nested/b5b2/ports W=<hdd>/capstone/trapk B=<hdd>/capstone/nested/b5b2
  local R=$X/renew; local D=$R/twin-$A Q=<hdd>/capstone/qemu-crv/build/qemu-system-riscv64 K=<hdd>/capstone/nested/rectrace-lb/Image.twin
  local SP=<hdd>/capstone/trapk/fpga/spec EXP PLUG=""
  [ ! -e "$D" ] || { echo "STOP $D exists"; return 2; }
  mkdir -p $D/ovl/stages
  cp -a $X/init-twin $D/ovl/init; cp -a $SP/SP-B5TL1-g3/ovl/smoke.dom $SP/SP-B5TL1-g3/ovl/probes.list $D/ovl/
  case $A in
    cal-stock) cp -a $SP/SP-B5TL1-g4mk/ovl/stages/B5TL1-500.perlbench_r $D/ovl/stages/; EXP=B5TL1-500.perlbench_r;;
    cal-full|full2) cp -a $X/stages/500.perlbench_r $D/ovl/stages/B5TL1P-500.perlbench_r; EXP=B5TL1P-500.perlbench_r;;
    ren1|ren2) cp -a $R/stages/fullren $D/ovl/stages/B5TL1P-500.perlbench_r; EXP=B5TL1P-500.perlbench_r;;
    fx) cp -a $R/stage-fx $D/ovl/stages/P1FX-perlren; EXP=P1FX-perlren;;
    *) echo "STOP arm $A"; return 2;;
  esac
  [ $A = fx ] || PLUG="$R/opcount.so,out=$D/opcount.txt"
  [ -z "$(find $D/ovl -type l)" ] || { echo "STOP symlinks in the overlay"; return 2; }
  [ -x $D/ovl/init ] && [ "$(stat -c %a $D/ovl/init)" = 755 ] || { echo "STOP /init is not executable"; return 2; }
  for s in $D/ovl/stages/*; do [ -s $s/stage/run.sh ] || { echo "STOP no run.sh in $s"; return 2; }; done
  (cd $D/ovl && find . -mindepth 1 | sort | cpio -o -H newc -R 0:0 2>/dev/null) > $D/overlay.cpio
  cat $W/fpga/base-rootfs.cpio $D/overlay.cpio | gzip -9 -n > $D/initramfs.cpio.gz
  (cd $D/ovl && find . -type f | sort | xargs sha256sum) > $D/overlay.sha256
  cp -a $R/boot-plugin.py $D/boot.py
  sha256sum $Q $K $D/initramfs.cpio.gz $D/boot.py $R/opcount.so > $D/inputs.sha256
  echo "$(date '+%F %T') asking for the QEMU lock" > $D/lock.log
  if [ -n "${TR_DRY:-}" ]; then echo "DRY: would boot $A, expect $EXP, plugin [${PLUG}]"; return 0; fi
  BOOT_PLUGIN="$PLUG" BOOT_MEM=962M BOOT_SILENCE_S=21600 CAPSTONE_QEMU_DOMAIN_LOCK=$HOME/.capstone-locks/ports-twin.lock \
    TRAPTEST_APPEND=sublet.dom_wide_grants=1 \
    python3 $B/corpus/locked.py taskset -c 28-29,60-61 nice -n 10 python3 $D/boot.py $Q \
    <hdd>/capstone/sbi-keep-run/fw-y/platform/generic/firmware/fw_jump.elf $K \
    $D/initramfs.cpio.gz $D/boot.log off > $D/verdict.txt 2> $D/lock.err || echo "boot rc=$?"
  echo "$(date '+%F %T') done" >> $D/lock.log
  if [ $A != fx ]; then
    SPEC_STAGE_ROOT=$X/twinroot python3 <hdd>/capstone/b5b2linux/grade-spec-chain-v2.py $D/boot.log $D/grade $EXP > $D/grade.txt 2>&1 || echo "grade rc=$?"
    [ -s $D/opcount.txt ] && command grep -q '^END$' $D/opcount.txt || echo "NO-OPCOUNT $D/opcount.txt"
  fi
  command grep -a -E 'TRAPTEST: stage .* (rc|mem)|FIXTURE_RC|FIXTURE_FAULT|control smoke|BOOT: ' $D/boot.log $D/verdict.txt > $D/lines.txt || true
}
main "$@"; rc=$?; echo "TWIN-DONE $1 rc=$rc"; exit $rc
