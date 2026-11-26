#!/bin/bash
# diag-boot.sh (nested lane, 2026-11-26): why spec-gcc's v3 trace boot hung after OpenSBI. Smoke-only boots, 240 s
# silence limit, one per variant: A v3 QEMU bare; B + dcplugin; C + CAPSTONE_RECTRACE_V2/DC (the failed form).
main() {
  set -u
  local R=<hdd>/capstone/nested/rectrace-dc L=<hdd>/capstone/nested/rectrace-lb B=<hdd>/capstone/nested/b5b2
  local W=<hdd>/capstone/trapk Q=$R/qemu/build/qemu-system-riscv64 G=$R/diag
  local O=$G/ovl; [ -e $O ] || { mkdir -p $O/stages; cp -a $L/ovl-small/init $L/ovl-small/smoke.dom $O/; : > $O/probes.list
    (cd $O && find . -mindepth 1 | sort | cpio -o -H newc -R 0:0 2>/dev/null) > $G/overlay.cpio
    cat $W/fpga/base-rootfs.cpio $G/overlay.cpio | gzip -9 -n > $G/initramfs.cpio.gz; }
  cp -a $B/ports/renew/boot-plugin.py $G/boot.py
  for v in ${VARIANTS:-A B C}; do
    local D=$G/$v; [ -e $D ] && { echo "$D exists"; return 2; }; mkdir -p $D
    local -a E=(BOOT_MEM=963M BOOT_SILENCE_S=240 CAPSTONE_QEMU_DOMAIN_LOCK=$HOME/.capstone-locks/rectrace-lb.lock TRAPTEST_APPEND=sublet.dom_wide_grants=1)
    case $v in
      A*) ;;
      B3*) E+=(BOOT_PLUGIN="$R/dcplugin.so,out=$D/dcplugin.txt,snap=$D/snap.txt");;   # 2026-11-27: the window snapshots
      B*) E+=(BOOT_PLUGIN="$R/dcplugin.so,out=$D/dcplugin.txt");;
      C*) E+=(BOOT_PLUGIN="$R/dcplugin.so,out=$D/dcplugin.txt" CAPSTONE_RECTRACE_V2=$D/trace CAPSTONE_RECTRACE_DC=1);;
    esac
    echo "$(date '+%T') $v begin"
    env "${E[@]}" python3 $B/corpus/locked.py taskset -c 30-31,62-63 nice -n 10 python3 $G/boot.py "$Q" \
      <hdd>/capstone/sbi-keep-run/fw-y/platform/generic/firmware/fw_jump.elf $L/Image.twin \
      $G/initramfs.cpio.gz $D/boot.log off > $D/verdict.txt 2> $D/lock.err
    echo "$(date '+%T') $v end: $(tail -n 1 $D/verdict.txt) linux=$(command grep -a -c 'Linux version' $D/boot.log) smoke=$(command grep -a -c 'control smoke.dom rc=7' $D/boot.log)"
  done
}
main "$@"; exit
