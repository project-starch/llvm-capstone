#!/bin/bash
# S1 on silicon, the SQLite half: A1's six probes in the measurement harness, one image per arm
# (memsys5 = unprotected, sublet = the port), the probe chosen at run time (`--s1-probe <n>`).
# Pre-registration: the paper bundle experiments/results/S1/2026-10-01-sqlite-a1-silicon/work-order.md;
# cells: tests/rtl-smoke/s1-sqlite-silicon-2026-10-01/cells.tsv. One boot = k800, every returning cell
# (boot "all") in table order, then the boot's ONE fault cell LAST (M-1: a fault wedges the core).
# Adapted from board-r322.sh; the pins, bake, VA, membership and watchdog guards are its.
set -u; R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd); U=${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}/unify
# The runner and its tools come from THIS tree; the FPGA buildroot, monitor and capstone-c come from the
# clone that has the submodules checked out (a git worktree has none; their directories are empty).
SUB=${CAPSTONE_SUBMODULE_ROOT:-$R}
B=$SUB/capstone/caplifive-system/sw/buildroot; BM=$B/components/opensbi/lib/sbi/capstone-sbi
FW=${CAPSTONE_BR_FW:-$B/build-fpga/build/opensbi-custom/build/platform/fpga/ariane/firmware}
CC=$SUB/capstone/capstone-c
[ -f $B/Makefile ] || { echo "no FPGA buildroot at $B (a worktree? set CAPSTONE_SUBMODULE_ROOT)" >&2; exit 1; }
F=$(cd "$(dirname "${BASH_SOURCE[0]}")/../s1-sqlite-silicon-2026-10-01" && pwd)
E=${S1_DIR:-${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}/s1-sqlite-silicon}   # <arm>/sqlite_silicon.dom
MEMLOCK=${MEMLOCK:-$HOME/bin/logs/machine-memory.lock}
BOOT=${BOOT:?boot 1..3}; TAG=${RUNTAG:-a}
OUT=$U/board-s1sql-b${BOOT}${TAG}; mkdir -p $OUT; LOG=$OUT/log; : > $LOG; rm -f $OUT/marker
say(){ echo "$(date +%T) $*" | tee -a $LOG; }; fail(){ say "FAIL: $*"; echo failed > $OUT/marker; exit 1; }
bake(){ say "bake $1: waiting for the machine memory lock (another lane may be measuring)"
  flock -w 7200 "$MEMLOCK" bash -c '
    B=$1; CC=$2; OUT=$3; tag=$4
    for a in modcapstone-rebuild linux-rebuild opensbi-rebuild; do
      ( cd $B && make build LINUX_PAYLOAD=1 A=$a CAPSTONE_CC_PATH=$CC ) > $OUT/bake-$tag-$a.log 2>&1 || exit 1
    done' _ "$B" "$CC" "$OUT" "$1"; }
pgrep -f 'python3 -m fpga_driver' >/dev/null && fail "a board runner is live"
[ "$(git -C $BM rev-parse --short HEAD)" = 2dcd3a5 ] || fail "monitor is not 2dcd3a5"
cd $B || fail "no FPGA buildroot copy"
[ "$(git rev-parse --short HEAD)" = d04bd83 ] || fail "FPGA copy is at $(git rev-parse --short HEAD), not d04bd83 (the #3 module)"
HOSTF=$B/overlay/test-domains/sqlite_host.user
[ "$(sha256sum $HOSTF | cut -c1-16)" = 2c9e82d101b48160 ] || fail "sqlite_host.user is not 2c9e82d101b48160 (the host the QEMU readings used)"
[ "$(sha256sum $B/overlay/test-domains/k800.dom | cut -c1-16)" = b2d60e525f807ea4 ] || fail "k800.dom is not the stock b2d60e525f807ea4"
# The cell table: name arm probe va hash hostargs boot want. Every boot runs the "all" cells in table
# order, then its own cell (a FAULT) last.
CELLS=$(awk -F'\t' '!/^#/ && $7=="all"' $F/cells.tsv; awk -F'\t' -v b="$BOOT" '!/^#/ && $7==b' $F/cells.tsv)
[ -n "$CELLS" ] || fail "no cells for boot $BOOT"
nf=$(printf '%s\n' "$CELLS" | awk -F'\t' '$8 ~ /^FAULT/' | wc -l); [ "$nf" -le 1 ] || fail "$nf FAULT cells in one boot"
last=$(printf '%s\n' "$CELLS" | tail -1 | awk -F'\t' '{print $8}'); case "$last" in FAULT*) ;; *) [ "$nf" -eq 0 ] || fail "the FAULT cell is not last";; esac
T=${CAPSTONE_BR_OVERLAY:-$B/overlay/test-domains}; TT=${CAPSTONE_BR_TARGET:-$B/build-fpga/target/test-domains}
# DRYRUN=1: every check above and below up to the bake, staging into scratch directories; the shared
# overlay, the bake and the board are not touched.
[ "${DRYRUN:-0}" = 1 ] && { T=$OUT/dry/overlay; TT=$OUT/dry/target; }
mkdir -p "$T" "$TT"
QP=${CAPSTONE_QEMU_PASS_DIR:-$HOME/capstone-artifacts/qemu-pass}
STAGED=""; D="/test-domains/lpc|k800:/test-domains/k800.dom"; PRE="  k800 -> 4"
while IFS=$'\t' read -r name arm probe va hash hostargs boot want; do
  src=$E/$arm/sqlite_silicon.dom; [ -f "$src" ] || fail "$name: no image $src"
  full=$(sha256sum $src|cut -d' ' -f1)
  [ "${full:0:16}" = "$hash" ] || fail "$name: image hash ${full:0:16} is not cells.tsv's $hash"
  grep -q "^$full  s1-$arm.dom\$" $F/SHA256SUMS || fail "$name: not the image SHA256SUMS records for s1-$arm.dom"
  [ -f "$QP/$full" ] || fail "$name: no QEMU record $QP/$full"
  e=$(python3 -c "import struct,sys;print(hex(struct.unpack_from('<Q',open(sys.argv[1],'rb').read(),0x18)[0]))" $src)
  [ "$e" = "$va" ] || fail "$name: entry $e, cells.tsv says $va"
  [ -n "$want" ] || fail "$name: empty want column -- the row did not split into 8 fields (an empty field collapses under tab IFS; use '-')"
  case "$hostargs" in -) hostargs="" ;; *) hostargs=" $hostargs" ;; esac
  dst=s1-$arm.dom
  case " $STAGED " in *" $dst "*) ;; *) cp -f "$src" "$T/$dst" && cp -f "$src" "$TT/$dst" || fail "stage $dst"; STAGED="$STAGED $dst";; esac
  D="$D,/test-domains/$dst:--speedtest1$hostargs --testset main --size 1 --s1-probe $probe"
  PRE="$PRE
  $name ($arm, probe $probe) -> $want"
done <<< "$CELLS"
say "pieces: monitor 2dcd3a5, module d04bd83, host 2c9e82d1; staged for boot $BOOT:$STAGED"
say "pre-registered:
$PRE"
say "=== stage list: $D"
[ "${DRYRUN:-0}" = 1 ] && { echo dryrun > "$OUT/marker"; say "DRYRUN: all pre-bake checks passed; stopping before the overlay, the bake and the board"; exit 0; }
# retire the big images that are not this boot's
STASH=$OUT/retired; mkdir -p $STASH; RESTORED=0
RETIRE="rtpc bigregion.user trapctl.dom fillsd.dom fillwarm.dom fillcost.dom fillnop.dom speedtest1_seven.dom speedtest1_baseline sqlite_host_rr.user speedtest1.dom"
restore(){ [ "$RESTORED" = 1 ] && return 0; RESTORED=1
  for f in $RETIRE; do [ -f "$STASH/$f" ] && { cp -f "$STASH/$f" "$T/$f"; cp -f "$STASH/$f" "$TT/$f"; }; done
  for f in $STAGED; do rm -f "$T/$f" "$TT/$f"; done
  bake restore && say "rebaked with the retired set back" || say "WARN: restore rebake failed -- the next boot MUST rebake"; }
trap restore EXIT
for f in $RETIRE; do [ -f "$T/$f" ] && cp -f "$T/$f" "$STASH/$f"; done
for f in $RETIRE; do rm -f "$T/$f" "$TT/$f"; done
for f in lpc k800.dom sqlite_host.user; do cp -f "$T/$f" "$TT/$f"; done
say "overlay: $(ls $T | tr '\n' ' ')"
python3 - "$B" <<'PY' | tee -a $LOG
import sys,glob,struct
L=sys.argv[1]; seen={}
for p in sorted(glob.glob(L+'/overlay/test-domains/*')):
    d=open(p,'rb').read()
    if d[:4]!=b'\x7fELF' or not p.endswith('.dom'): continue
    e=struct.unpack_from('<Q',d,0x18)[0]; n=p.split('/')[-1]
    print(f"  entry {e:#012x}  {n}"); seen.setdefault(e,[]).append(n)
dup={e:v for e,v in seen.items() if len(v)>1}
print('entry-VA collisions:', 'none' if not dup else dup); sys.exit(1 if dup else 0)
PY
[ ${PIPESTATUS[0]} -eq 0 ] || fail "two staged images share an entry VA"
bake run || fail "bake after staging"
H=$(sha256sum $FW/fw_payload.bin|cut -d' ' -f1); echo $H > $OUT/fw.sha
say "fw_payload ${H:0:12} (monitor $(git -C $BM rev-parse --short HEAD), buildroot $(git -C $B rev-parse --short HEAD))"
python3 - "$B" "$STAGED" <<'PY' | tee -a $LOG
import sys
L=sys.argv[1]; files=sys.argv[2].split()
cpio=open(L+'/build/images/rootfs.cpio','rb').read()
miss=[f for f in files+['k800.dom','lpc','sqlite_host.user'] if cpio.find(open(L+'/overlay/test-domains/'+f,'rb').read()[:4096])<0]
print('initramfs membership:', 'all present' if not miss else 'MISSING '+','.join(miss))
sys.exit(1 if miss else 0)
PY
[ ${PIPESTATUS[0]} -eq 0 ] || fail "initramfs check"
export FPGA_URL="$(cat "${CAPSTONE_FPGA_URL_FILE:-$HOME/.claude-kisp/secrets/fpga-console-url}")"; export FPGA_FW=$FW/fw_payload.bin
export FPGA_BITSTREAM=${FPGA_BITSTREAM:-caplifive_r43_8f6a0af.bit} REFUSAL_RECORD=${REFUSAL_RECORD:-1}
export PREFLIGHT_ALLOW_SHORT=1 PREFLIGHT_ALLOW_SLOTS=1
export ENTRY_STALL_S=420 EARLY_HALT_CONTROL=0 WEDGE_TRACER=0 HALT_MUX_READS=0
export SQLITE_HOST=/test-domains/sqlite_host.user
# A returning probe ends with SQLITE_HC_RET_DONE, i.e. `SQ: obs=0`, which is not a staged-marker family:
# without this the runner HARD STOPs on the first probe. A probe's setup failure (0x5117E105) still stops it.
export PROBE_SENTINELS=0
BUDGET=${BUDGET:-600}; export SQLITE_STAGE_TIMEOUT=$BUDGET SQLITE_IDLE_S=$BUDGET
export SQLITE_STAGE_DOMS="$D" PROBE_SCOPED_OUT=$OUT/boot.txt PROBE_RAW_OUT=$OUT/boot-raw.txt
say "=== boot s1sql-b${BOOT}${TAG} = S1 SQLite probes on hardware, boot $BOOT ==="
say "=== stage list: $D"
for i in $(seq 1 30); do c=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' "$FPGA_URL" 2>/dev/null); [ "$c" != "000" ] && break; say "console down ($i)"; sleep 60; done
# The runner resolves its repo from its own file and the preflight from its cwd, so both must run in the
# clone that has the submodules -- and that clone's runner, preflight and watchdog must be this tree's.
for f in capstone/tests/rtl-smoke/fpga_driver capstone/tests/preflight-board-run.sh capstone/tests/rtl-smoke/board-watchdog.sh; do
  diff -rq -x __pycache__ "$R/$f" "$SUB/$f" > /dev/null || fail "$SUB/$f differs from this tree's; the runner would not be the one checked"
done
cd $SUB/capstone/tests/rtl-smoke
T0=$(date +%s)
timeout $((10*BUDGET + 2400)) python3 -m fpga_driver.run_sqlite_stages_fpga > $OUT/driver.log 2>&1 &
RUNNER=$!
ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=${ENTRY_STALL_S:-420} \
  bash board-watchdog.sh "$OUT/driver.log" "${WD_IDLE:-900}" "$RUNNER" > $OUT/watchdog.log 2>&1 &
WD=$!
wait $RUNNER; rc=$?
kill $WD 2>/dev/null; wait $WD 2>/dev/null
say "runner ended after $(( $(date +%s) - T0 )) s with rc=$rc"
python3 - $OUT/driver.log <<'PY' | tee -a $LOG
import re,sys
from fpga_driver.transcript import read, scope_to_run, uart_text, strip_markers, find_all, require, arm_segments, TranscriptError
whole=read(sys.argv[1]); framed=scope_to_run(whole)
try: joined=require(uart_text(framed), f"UART after this run's load_image in {sys.argv[1]}")
except TranscriptError as e: print(f"  ERROR: {e}"); sys.exit(2)
s=strip_markers(joined)
try: arms=require(arm_segments(framed), "'[stages] --> TEST' arms after this run's load_image")
except TranscriptError as e: print(f"  ERROR: {e}"); sys.exit(2)
print("  control:", find_all(r'RESULT k800 retval=[0-9-]+', s))
print("  boot banners after this run's load_image (must be 1):", max(len(find_all(r'OpenSBI v', s)), len(find_all(r'Linux version', s))))
KEEP=re.compile(r"(NOTRAP done|__CAPSTONE_SQLITE_[A-Z]+_PASSED__|row name=|insert rc=|cross_rows=|rows=|before=|max=|write rc=|domain halted|cause)")
for a in arms:
    tl=re.search(r'TRAP LOG \{seen,mcause\[6:0\]\}\s+(0x[0-9a-f]+)', a.framed)
    lines=[l.strip() for l in a.uart.splitlines() if KEEP.search(l)]
    v=("returned" if a.returned else "WEDGED trap-log "+(tl.group(1) if tl else '?') if a.returned is False else "no result")
    print(f"  {a.label[:60]:60} -> {v}")
    for l in lines[:12]: print(f"      {l}")
PY
srs=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ] && grep -Eq 'preflight:? BLOCKED' "$OUT/driver.log" 2>/dev/null; then
  echo refused > "$OUT/marker"; say "refused: preflight BLOCKED (runner rc=$rc)"; exit 1; fi
[ "$rc" -eq 0 ]  || { echo failed > "$OUT/marker"; say "failed: runner rc=$rc"; exit 1; }
[ "$srs" -eq 0 ] || { echo failed > "$OUT/marker"; say "failed: summary rc=$srs"; exit 1; }
echo done > "$OUT/marker"; say "done"
