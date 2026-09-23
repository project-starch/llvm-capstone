#!/usr/bin/env bash
# Run FFmpeg milestone images in Capstone QEMU.   usage: run-qemu.sh [stage 1..5 | all]
#
# `all` runs M1..M5 and the M5 flipped-input control in ONE boot, ascending, each image
# independent (`;`, not `&&`), so a boot answers every stage it can reach and the first
# stage that does not return is the bisection point (CLAUDE.md: batch variants).
#
# QEMU runs are serialized across lanes: this takes $CAPSTONE_QEMU_LOCK itself (the shared
# rootfs.ext2 must never see two guests), unless the caller already holds it.
#
# WHAT DECIDES A PASS. run-domain-smoke.py checks its markers as SUBSTRINGS of the guest's
# output, and that output includes the shell's echo of the whole command. So the boot
# control and every __FFAPP_BEGIN/END__ marker are satisfied by the echo alone, "STAGE M2"
# also matches "STAGE M2a", and the chain's exit status is always the last echo's. They are
# kept as a coarse smoke check; they are NOT the verdict (audit, 2026-09-23). The verdict is:
#   * per image: its OWN section of the log -- between the whole-line BEGIN/END markers, the
#     echoed command cannot match them -- must contain the whole line
#     "__CAPSTONE_FFAPP_STAGE_REACHED__ <n>", printed by the host only when capstone_main
#     returned exactly the milestone the image was built for (src/linux-guest/ffapp_host.c);
#   * for stage 5: the M5 section's frame lines must equal the native reference, and the
#     flipped-input control -- which must itself have REACHED 5 -- must decode as many
#     frames with at least one changed hash (host/compare-md5.py).
# The kernel console is raised to level 7, so the module's own allocation line
# ("code size = ..., tot_size = ...") lands in the log beside the loader's.
#
# Prerequisites: build-native.sh (the oracle) and build-domain.sh (the images).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
APP_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$APP_DIR/../../../tests/capstone-test-env.sh"

STAGE=${1:-1}
case $STAGE in
  1|2|3|4|5) STAGES=$STAGE ;;
  all) STAGES="1 2 3 4 5" ;;
  # m2diag: M2a (open_input only), then M2 with the host printing every request, so a
  # stage that does not return shows whether rounds keep flowing (slow) or stop (stuck).
  m2diag) STAGES="6 2v" ;;
  # probe: the FFAPP_DIAG M2a image (first bytes read, demuxers, probe verdict, open error)
  probe) STAGES="6diag9p 6diag" ;;   # matched pair: 9p path, then /tmp path
  *) echo "stage must be 1..5, all, m2diag or probe" >&2; exit 2;;
esac
WORK=${FFAPP_WORK:-$CAPSTONE_TMP_ROOT/ffmpeg-app}
# FFAPP_HEAP picks the heap arm's images (build-domain.sh); level0 is the run of record's.
case ${FFAPP_HEAP:-level0} in
  level0) DOM="$WORK/domain"; LOG=${LOG_FILE:-$WORK/qemu-$STAGE.log} ;;
  shrink) DOM="$WORK/domain-shrink"; LOG=${LOG_FILE:-$WORK/qemu-shrink-$STAGE.log} ;;
  sublet) DOM="$WORK/domain-sublet"; LOG=${LOG_FILE:-$WORK/qemu-sublet-$STAGE.log} ;;
  *) echo "FFAPP_HEAP must be level0, shrink or sublet" >&2; exit 2 ;;
esac
SHARE="$WORK/share"
for f in "$DOM/ffapp.user" "$WORK/input.mkv" "$WORK/stock.framemd5"; do
  [ -f "$f" ] || { echo "missing $f; run build-native.sh and build-domain.sh first" >&2; exit 2; }
done
rm -rf "$SHARE"; mkdir -p "$SHARE"
cp "$DOM/ffapp.user" "$WORK/input.mkv" "$WORK/input.flip.mkv" "$SHARE/"

RUN="dmesg -n 7; cp /mnt/host/ffapp.user /tmp/ffapp.user && chmod 0755 /tmp/ffapp.user && cp /mnt/host/input.mkv /mnt/host/input.flip.mkv /tmp/"
MARKERS=(--success-marker '__CAPSTONE_QEMU_BOOT_CONTROL_OK__')
SECTIONS=()          # "BEGIN END expected-stage" per image, verified after the boot
for spec in $STAGES; do
  st=${spec%v}; verbose=; [ "$spec" != "$st" ] && verbose=" verbose"
  img=ffapp_m$st.dom
  case $st in *diag|*diag9p) img=ffapp_m$st.dom; st=${st%%diag*} ;; esac
  [ -f "$DOM/$img" ] || { echo "missing $DOM/$img; run build-domain.sh" >&2; exit 2; }
  cp "$DOM/$img" "$SHARE/"
  RUN="$RUN; echo __FFAPP_BEGIN_M${st}__; /tmp/ffapp.user /mnt/host/$img $st$verbose; echo __FFAPP_END_M${st}__"
  SECTIONS+=("__FFAPP_BEGIN_M${st}__ __FFAPP_END_M${st}__ $st")
  label=M$st; [ "$st" = 6 ] && label=M2a
  MARKERS+=(--success-marker "STAGE $label" --success-marker "__CAPSTONE_FFAPP_STAGE_REACHED__ $st")
done
case " $STAGES " in *" 5 "*)
  cp "$DOM/ffapp_m5flip.dom" "$SHARE/"
  RUN="$RUN; echo __FFAPP_BEGIN_FLIP__; /tmp/ffapp.user /mnt/host/ffapp_m5flip.dom 5; echo __FFAPP_END_FLIP__"
  SECTIONS+=("__FFAPP_BEGIN_FLIP__ __FFAPP_END_FLIP__ 5")
  MARKERS+=(--success-marker '__FFAPP_END_FLIP__') ;;
esac

smoke=(python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py"
       --share-dir "$SHARE" --log-file "$LOG" --timeout-multiplier "${TIMEOUT_MULTIPLIER:-8}"
       --guest-command "echo __CAPSTONE_QEMU_BOOT_CONTROL_OK__; $RUN" "${MARKERS[@]}")
# FFAPP_BUILDROOT_DIR: boot from a directory whose build/images/ holds a PRIVATE rootfs copy
# (kernel and firmware may be symlinks to the shared ones). Used on 2026-09-23 while the
# shared rootfs.ext2 was corrupt: the copy was repaired, the shared image left untouched.
[ -n "${FFAPP_BUILDROOT_DIR:-}" ] && smoke+=(--buildroot-dir "$FFAPP_BUILDROOT_DIR")
# A git worktree has no QEMU build either; point at the main clone's.
[ -n "${FFAPP_QEMU_BINARY:-}" ] && smoke+=(--qemu-binary "$FFAPP_QEMU_BINARY")
set +e
if [ "${CAPSTONE_QEMU_LOCK_HELD:-0}" = 1 ]; then
  "${smoke[@]}"
else
  mkdir -p "$(dirname "$CAPSTONE_QEMU_LOCK")"
  CAPSTONE_QEMU_LOCK_HELD=1 flock -x -w "${FFAPP_LOCK_WAIT:-3600}" "$CAPSTONE_QEMU_LOCK" "${smoke[@]}"
fi
rc=$?
set -e
echo "run-domain-smoke exit status $rc; serial log $LOG"
[ "$rc" = 0 ] || exit "$rc"

# Only a line that IS the marker delimits a section: the shell echoes the whole guest
# command, markers and all, and a pattern match on that line would pull M1..M4's output
# (including M4's frame-0 hash) into M5's.
section() { python3 -c '
import sys
begin, end, cur, out = sys.argv[2], sys.argv[3], False, []
for line in open(sys.argv[1], errors="replace"):
    t = line.strip()
    if t == begin: cur = True; continue
    if t == end: break
    if cur: out.append(line.rstrip("\r\n"))
print("\n".join(out))' "$LOG" "$1" "$2"; }

verdict=0
for sec in "${SECTIONS[@]}"; do
  read -r b e want <<<"$sec"
  if section "$b" "$e" | grep -xF "__CAPSTONE_FFAPP_STAGE_REACHED__ $want" >/dev/null; then
    echo "section $b: REACHED $want"
  else
    echo "section $b: did NOT reach $want" >&2; verdict=1
  fi
done
[ "$verdict" = 0 ] || { echo "FAILED: at least one image did not return its own milestone" >&2; exit 1; }

case " $STAGES " in *" 5 "*)
  section __FFAPP_BEGIN_M5__ __FFAPP_END_M5__     > "$WORK/domain-m5.out"
  section __FFAPP_BEGIN_FLIP__ __FFAPP_END_FLIP__ > "$WORK/domain-m5flip.out"
  python3 "$SCRIPT_DIR/compare-md5.py" "$WORK/stock.framemd5" "$WORK/domain-m5.out" \
    --control "$WORK/domain-m5flip.out" ;;
esac
