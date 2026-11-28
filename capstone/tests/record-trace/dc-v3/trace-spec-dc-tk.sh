#!/bin/bash
# trace-spec-dc-tk.sh NAME [STAGE_DIR] (nested lane, 2026-11-27): trace-spec-dc-ic.sh with QEMU built from the tracer
# with the TICKS side file (qemu/build-tk) and CAPSTONE_RECTRACE_TICKS=1, so the trace also gets <trace>.ticks: one line
# per machine timer interrupt with its event index, exact instruction counts and the D$ model's read totals. The trace
# format is unchanged. Otherwise as trace-spec-dc-ic.sh, whose header follows.
# trace-spec-dc-ic.sh NAME [STAGE_DIR] (nested lane, 2026-11-27): trace-spec-dc.sh under QEMU -icount ($TD_ICOUNT, e.g.
# shift=7), so guest time, and with it the timer's M round trips per instruction, is set by the instruction count and
# not by the host's speed; /init is ovl-ic/init (stage bound 30000 s) and the boot script boot-plugin-ic.py (-icount).
# Otherwise as trace-spec-dc.sh, whose header follows.
# trace-spec-dc.sh NAME [STAGE_DIR] (nested lane, 2026-11-26): one record-access trace in format v3 (D$ residency) of
# one B5TL1 SPEC test stage on the trap-route twin. It is rectrace-lb/trace-spec.sh v2 with four changes:
#   - QEMU is rectrace-dc/qemu (nested/record-trace-dc: v2 at 702f24732b plus the online 64x4 LB and the D$ model);
#   - CAPSTONE_RECTRACE_DC=1 and the plugin rectrace-dc/dcplugin.so (out=runs/spec-NAME/dcplugin.txt);
#   - the boot script is ports/renew/boot-plugin.py (runtime's boot-long.py plus -plugin and a wait for QEMU's own exit
#     after TRAPTEST: done, so the plugin's and the tracer's at-exit totals are written), with BOOT_MEM=963M and
#     BOOT_SILENCE_S=3600, the memory and limit of v2's mcf trace (boot-3600.py at 963M);
#   - traces go to rectrace-dc/traces;
#   - (2026-11-27) the plugin also writes snap.txt, the model's totals at each console byte, for dc-window.py.
# The overlay is built exactly as v2's: rectrace-lb/ovl-small's /init and smoke.dom, no probes, STAGE_DIR copied
# (symlinks resolved) to /stages/B5TL1-<bench>.
main() {
  set -euo pipefail
  local NM=${1:?name} SD=${2:-} R=<hdd>/capstone/nested/rectrace-dc L=<hdd>/capstone/nested/rectrace-lb
  local W=<hdd>/capstone/trapk B=<hdd>/capstone/nested/b5b2
  local D=$R/runs/spec-$NM O=$R/runs/spec-$NM/ovl Q=$R/qemu/build-tk/qemu-system-riscv64
  [ ! -e "$D" ] || { echo "STOP $D exists"; return 2; }
  mkdir -p $O/stages $R/traces
  cp -a $R/ovl-ic/init $L/ovl-small/smoke.dom $O/; : > $O/probes.list
  if [ -n "$SD" ]; then
    local SR; SR=$(readlink -f "$SD"); local SO=$O/stages/B5TL1-$(basename "$SD")
    mkdir -p $SO && cp -a "$SR/." $SO/
    [ -s $SO/stage/run.sh ] || { echo "STOP no stage/run.sh in the copy of $SR"; return 2; }
    [ -z "$(find $SO -type l -lname '/*')" ] || { echo "STOP absolute symlinks in the copy of $SR"; return 2; }
  fi
  [ -x $O/init ] || { echo "STOP /init is not executable"; return 2; }
  (cd $O && find . -mindepth 1 | sort | cpio -o -H newc -R 0:0 2>/dev/null) > $D/overlay.cpio
  cat $W/fpga/base-rootfs.cpio $D/overlay.cpio | gzip -9 -n > $D/initramfs.cpio.gz
  cp -a $R/boot-plugin-ic.py $D/boot.py
  local IC=${TD_ICOUNT:?TD_ICOUNT, e.g. shift=7}
  sha256sum $Q $L/Image.twin $D/initramfs.cpio.gz $D/boot.py $R/dcplugin.so > $D/inputs.sha256
  (cd $O && find . -type f | sort | xargs sha256sum) > $D/overlay.sha256
  echo "$(date '+%F %T') asking for the QEMU lock" > $D/lock.log
  if [ -n "${TD_DRY:-}" ]; then echo "DRY: would trace $NM with -icount $IC"; return 0; fi
  CAPSTONE_RECTRACE_V2=$R/traces/spec-$NM CAPSTONE_RECTRACE_DC=1 CAPSTONE_RECTRACE_TICKS=1 BOOT_ICOUNT=$IC BOOT_PLUGIN="$R/dcplugin.so,out=$D/dcplugin.txt,snap=$D/snap.txt" \
    BOOT_MEM=963M BOOT_SILENCE_S=3600 CAPSTONE_QEMU_DOMAIN_LOCK=$HOME/.capstone-locks/rectrace-lb.lock TRAPTEST_APPEND=sublet.dom_wide_grants=1 \
    python3 $B/corpus/locked.py taskset -c 30-31,62-63 nice -n 10 python3 $D/boot.py "$Q" \
    <hdd>/capstone/sbi-keep-run/fw-y/platform/generic/firmware/fw_jump.elf $L/Image.twin \
    $D/initramfs.cpio.gz $D/boot.log off > $D/verdict.txt 2> $D/lock.err
  echo "$(date '+%F %T') done" >> $D/lock.log
  echo TRACE-DONE $NM
}
main "$@"; exit
