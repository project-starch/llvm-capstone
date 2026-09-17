#!/bin/bash
# Boots sw83.. -- E3 / R1 on silicon, boot $R1_BOOT of 8 (12 invocations each, 90 in all = 5 fresh-domain repetitions x 18): the slots-and-pools harness (capstone/sublet/r1, image 6a569a7e5e34178b at
# entry 0x410000) through the readback host, every (arm, series, pattern) as TWO separate invocations of --reps 1
# (fresh domain and fresh arena grant each; METHODS: a repetition is a fresh domain), a tables region of 64 KiB above
# each pool so the release keeps the slot (rc=1; a popped pool breaks the next create_region in the same boot --
# chain18 run 2, 2026-09-15), arenas 4 MiB (nodes/bytes/depth), 8 MiB (heap), 2 MiB (object). Controls first and last.
# Node demand ~5,600 for the boot (emulator sweep: ~2,800 for each Sublet-arm repetition of everything), budget 50,000.
# Pre-registered (the Sublet arm): nd = 2n for a shared point of n leaves (2/8/32/128/512), 32 for bytes/heap points,
# 47 for depth points (32 + 15 delegations), 34 then 1 for object points; ty=3 (UNINIT) and fb=B for shared/combined,
# ty!=3 and fb=0 for individual; bad=0 and ok=1 everywhere; the spatial arm rv=fl=in=0 by construction.
# Predicted cycles from the July primitives and the fill probes: fl ~ 1.5 cycles/byte (24 per 16-byte stc) so
# ~390 k at B=256 KiB and ~1.57 M at 1 MiB; rv flat in n if the RTL's revoke is O(1) (~100-200) -- a rise with n is
# the node-linear finding R1 asks about; bk ~ 10-20 per slot cleared; re a few hundred.
# BUDGET 300 s per stage; ENTRY_STALL_S 420.
# ENTRY VA: every image this driver stages must be linked at DOMAIN_BASE_VA=0x410000. The k800 control
# rung occupies the build script's DEFAULT base 0x10000, so a harness built with the default collides
# with the control and preflight C15 refuses the boot (R-3 hangs a second domain at a reused VA). The
# hash named above is the E3/R1 vintage of the harness; the image itself arrives by R1_IMG/R1_HASH.
# An image validated ONLY on the emulator can carry the default base, because nothing is staged beside
# it there -- the emulator-validated build and the bootable build are then not the same file.
set -u; R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd); U=${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}/unify
B=$R/capstone/caplifive-system/sw/buildroot; BM=$B/components/opensbi/lib/sbi/capstone-sbi
FW=${CAPSTONE_BR_FW:-$B/build-fpga/build/opensbi-custom/build/platform/fpga/ariane/firmware}
CC=$R/capstone/capstone-c
MEMLOCK=${MEMLOCK:-$HOME/bin/logs/machine-memory.lock}   # the lead may point a small relink at a private lock (2026-09-15: the M2 bakes waited on another project's 56-way build)
R1_IMG=${R1_IMG:?}; R1_HASH=${R1_HASH:?}; R1_LIST=${R1_LIST:?}; R1_QEMU_LOG=${R1_QEMU_LOG:?}; export R1_HASH
OUT=$U/board-${R1_OUT_TAG:-r1e4}-b${R1_BOOT:?}; mkdir -p $OUT; LOG=$OUT/log; : > $LOG; rm -f $OUT/marker
say(){ echo "$(date +%T) $*" | tee -a $LOG; }; fail(){ say "FAIL: $*"; echo failed > $OUT/marker; exit 1; }
bake(){ say "bake $1: waiting for the machine memory lock (another lane may be measuring)"
  flock -w 7200 "$MEMLOCK" bash -c '
    B=$1; CC=$2; OUT=$3; tag=$4
    for a in modcapstone-rebuild linux-rebuild opensbi-rebuild; do
      ( cd $B && make build LINUX_PAYLOAD=1 A=$a CAPSTONE_CC_PATH=$CC ) > $OUT/bake-$tag-$a.log 2>&1 || exit 1
    done' _ "$B" "$CC" "$OUT" "$1"; }
pgrep -f 'python3 -m fpga_driver' >/dev/null && fail "a board runner is live"
[ "$(git -C $BM rev-parse --short HEAD)" = 2dcd3a5 ] || fail "monitor is not 2dcd3a5 (the D3 writeback fix; 4274268 is the OLD pin)"
cd $B || fail "no FPGA buildroot copy"
[ "$(git rev-parse --short HEAD)" = d04bd83 ] || fail "FPGA copy is at $(git rev-parse --short HEAD), not d04bd83 (the #3 module)"
IMG=$R1_IMG
[ "$(sha256sum $IMG|cut -c1-16)" = "$R1_HASH" ] || fail "the R1 harness image is not $R1_HASH"
[ -f ${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}/qemu-pass/$(sha256sum $IMG|cut -d' ' -f1) ] || fail "no QEMU pass record for the harness image"
R1_QEMU_GATE=${R1_QEMU_GATE:-R1 lat}; R1_QEMU_GATE_MIN=${R1_QEMU_GATE_MIN:-5}
[ "$(grep -ac "$R1_QEMU_GATE" "$R1_QEMU_LOG")" -ge "$R1_QEMU_GATE_MIN" ] || fail "the emulator run of this image is not on record in $R1_QEMU_LOG (fewer than $R1_QEMU_GATE_MIN '$R1_QEMU_GATE' lines)"
RRH=${R1_HOST:?set R1_HOST=<path to sqlite_host_rr.user, the readback host>}
[ "$(sha256sum $RRH|cut -c1-16)" = "${R1_HOST_HASH:-2c9e82d101b48160}" ] || fail "rr host is not ${R1_HOST_HASH:-2c9e82d101b48160}"
LPC=${CAPSTONE_BR_OVERLAY:-$B/overlay/test-domains}/lpc
[ "$(sha256sum $LPC|cut -c1-16)" = 3b93a2b6e2adfa36 ] || fail "lpc on the overlay is not 3b93a2b6e2adfa36"
[ -f "$B/overlay/test-domains/k800.dom" ] || fail "k800.dom missing from the overlay"
say "pieces: R1 harness $R1_HASH (E4 build: latency + calib; emulator run on record), rr host 2c9e82d101b48160, lpc 3b93a2b6e2adfa36"
T=${CAPSTONE_BR_OVERLAY:-$B/overlay/test-domains}; TT=${CAPSTONE_BR_TARGET:-$B/build-fpga/target/test-domains}; mkdir -p "$T" "$TT"
for s in sqlite_host_rr.user; do cp -f "$T/$s" $OUT/$s.before 2>/dev/null; done
cp -f $IMG "$T/r1_slots_pools.dom" || fail "stage the R1 harness"
cp -f $RRH "$T/sqlite_host_rr.user" || fail "stage rr host"
STASH=$OUT/retired; mkdir -p $STASH; RESTORED=0
RETIRE="rtpc bigregion.user trapctl.dom fillsd.dom fillwarm.dom fillcost.dom fillnop.dom speedtest1_seven.dom speedtest1_baseline speedtest1.dom sqlite_host.user"
restore(){ [ "$RESTORED" = 1 ] && return 0; RESTORED=1
  for s in sqlite_host_rr.user; do [ -f $OUT/$s.before ] && { cp -f $OUT/$s.before "$T/$s"; cp -f $OUT/$s.before "$TT/$s"; }; done; rm -f "$T/r1_slots_pools.dom" "$TT/r1_slots_pools.dom"
  for f in $RETIRE; do [ -f "$STASH/$f" ] && { cp -f "$STASH/$f" "$T/$f"; cp -f "$STASH/$f" "$TT/$f"; }; done
  bake restore && say "rebaked with the retired set back" || say "WARN: restore rebake failed -- the next boot MUST rebake"; }
trap restore EXIT
for f in $RETIRE; do [ -f "$T/$f" ] && cp -f "$T/$f" "$STASH/$f"; done
for f in $RETIRE; do rm -f "$T/$f" "$TT/$f"; done
for f in lpc sqlite_host_rr.user r1_slots_pools.dom k800.dom; do cp -f "$T/$f" "$TT/$f"; done
say "staged: $(ls $T | tr '\n' ' ')"
python3 - "$B" <<'PY' | tee -a $LOG
import sys,glob,struct
L=sys.argv[1]; seen={}
for p in sorted(glob.glob(L+'/overlay/test-domains/*')):
    d=open(p,'rb').read()
    if d[:4]!=b'\x7fELF': continue
    e=struct.unpack_from('<Q',d,0x18)[0]; n=p.split('/')[-1]
    print(f"  entry {e:#012x}  {n}"); seen.setdefault(e,[]).append(n)
dup={e:v for e,v in seen.items() if len(v)>1}
print('entry-VA collisions:', 'none' if not dup else dup); sys.exit(1 if dup else 0)
PY
[ ${PIPESTATUS[0]} -eq 0 ] || fail "two staged images share an entry VA"
bake run || fail "bake after staging"
KO=${CAPSTONE_KO:-$(ls -t ${CAPSTONE_BR_BUILD:-$B/build-fpga}/build/modcapstone*/module/capstone.ko 2>/dev/null|head -1)}
[ "$(strings $KO|grep -c 'domain declares dom_data')" -ge 1 ] || fail "the baked .ko lacks the #3 marker"
[ "$(sha256sum $B/build/target/capstone.ko|cut -c1-16)" = "$(sha256sum $KO|cut -c1-16)" ] || fail "target capstone.ko is not the built one"
H=$(sha256sum $FW/fw_payload.bin|cut -d' ' -f1); echo $H > $OUT/fw.sha
say "fw_payload ${H:0:12} (monitor $(git -C $BM rev-parse --short HEAD), buildroot $(git -C $B rev-parse --short HEAD))"
python3 - "$B" <<'PY' | tee -a $LOG
import sys,hashlib,os
L=sys.argv[1]
import os
want={'r1_slots_pools.dom':os.environ['R1_HASH'],'lpc':'3b93a2b6e2adfa36','sqlite_host_rr.user':'2c9e82d101b48160'}
bad=0
for d in ('overlay/test-domains','build/target/test-domains'):
    for f,exp in want.items():
        p=os.path.join(L,d,f)
        got=hashlib.sha256(open(p,'rb').read()).hexdigest()[:16] if os.path.exists(p) else 'ABSENT'
        ok=got==exp; bad+=0 if ok else 1
        print(f"  {d:26} {f:20} {got}  {'ok' if ok else 'MISMATCH exp '+exp}")
cpio=open(L+'/build/images/rootfs.cpio','rb').read()
miss=[f for f in ['k800.dom','lpc','r1_slots_pools.dom','sqlite_host_rr.user'] if cpio.find(open(L+'/overlay/test-domains/'+f,'rb').read()[:8192])<0]
print('initramfs membership:', 'all present' if not miss else 'MISSING '+','.join(miss))
ko=open(L+'/build/target/capstone.ko','rb').read()
print('initramfs carries the #3 module:', 'yes' if cpio.find(ko[:8192])>=0 and b'domain declares dom_data' in cpio else 'NO')
sys.exit(1 if (bad or miss or cpio.find(ko[:8192])<0) else 0)
PY
[ ${PIPESTATUS[0]} -eq 0 ] || fail "staged set / initramfs check"
export FPGA_URL="$(cat "${CAPSTONE_FPGA_URL_FILE:-$HOME/.claude-kisp/secrets/fpga-console-url}")"; export FPGA_FW=$FW/fw_payload.bin
# The resident name is a KNOB with the current silicon as its default, so a reflash costs one
# exported variable rather than an edit to each of the seven drivers. Set it to the string the
# console REPORTS as flash_state.nv_bitstream_name after the flash, never to the filename you
# happened to upload. A wrong value hard-stops the boot, which is the safe direction.
export FPGA_BITSTREAM=${FPGA_BITSTREAM:-caplifive_m1_054cea69b.bit} FPGA_BITSTREAM_UNVERIFIED=1
export PREFLIGHT_ALLOW_SHORT=1 PREFLIGHT_ALLOW_SLOTS=1
export ENTRY_STALL_S=420 EARLY_HALT_CONTROL=0 WEDGE_TRACER=0 HALT_MUX_READS=0
export SQLITE_HOST=/test-domains/sqlite_host_rr.user
BUDGET=300; export SQLITE_STAGE_TIMEOUT=$BUDGET SQLITE_IDLE_S=$BUDGET
D="/test-domains/lpc|k800:/test-domains/k800.dom"
# this boot's slice of the invocation list: lines [12*(k-1), 12*k) of e3/invocations.txt (rep arm series pattern arena)
: "${R1_BOOT:?set R1_BOOT=k (1-based)}"
LIST=$R1_LIST; FROM=$(( 12*(R1_BOOT-1)+1 )); TO=$(( 12*R1_BOOT ))
NINV=0
while read -r rep arm ser pat arena extra; do
  D="$D,/test-domains/sqlite_host_rr.user|/test-domains/r1_slots_pools.dom:--speedtest1 --arena $arena --tables 65536 --testset r1 --arm $arm --series $ser --pattern $pat ${extra:---reps 1}"
  NINV=$((NINV+1))
done < <(sed -n "${FROM},${TO}p" $LIST)
[ "$NINV" -ge 1 ] || fail "no invocations for boot $R1_BOOT"
D="$D,/test-domains/lpc|k800:/test-domains/k800.dom"
export SQLITE_STAGE_DOMS="$D" PROBE_SCOPED_OUT=$OUT/boot.txt PROBE_RAW_OUT=$OUT/boot-raw.txt
say "=== boot ${R1_BOOT_TAG:-sw8x-e4} = ${R1_BOOT_DESC:-E4/H1 calibration}: control, $NINV harness invocations (list lines $FROM-$TO of $R1_LIST), control ==="
# R1_PREREG REPLACES the series-specific default rather than appending to it: an m1 boot that advertised
# the E4 default ("4-5 point lines per invocation") reads as under-reporting when its terminal record is
# the end line instead (apollo, 2026-09-15). The two universal expectations stay in front of it.
say "=== pre-registered: retval=4 twice; $NINV x speedtest1-ran=0x4EB1xxxx; ${R1_PREREG:-R1 lat per-load: 4 and 16 KiB in the D$ (a few cycles), 64 KiB..1 MiB at DRAM latency (tens of cycles, flat from 256 KiB); R1 calib cyc_cyc = the mcycle read-to-read cost; nodes regression nd=2n}; $NINV x released pool rc=1; every R1 point line ok=1 bad=0 (5 lines per nodes/bytes/heap invocation, 4 per depth/object); nd = 2n on shared nodes points; fb = B on shared/combined, 0 on individual; no refused lines; no create_region failure (<= 12 region-bearing invocations per boot: the module ran out at the 15th on the emulator, chain18 run 4) ==="
for i in $(seq 1 30); do c=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' "$FPGA_URL" 2>/dev/null); [ "$c" != "000" ] && break; say "console down ($i)"; sleep 60; done
cd $R/capstone/tests/rtl-smoke
T0=$(date +%s)
timeout $((14*BUDGET + 2400)) python3 -m fpga_driver.run_sqlite_stages_fpga > $OUT/driver.log 2>&1 &
RUNNER=$!
ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=${ENTRY_STALL_S:-420} \
  bash board-watchdog.sh "$OUT/driver.log" "${WD_IDLE:-900}" "$RUNNER" > $OUT/watchdog.log 2>&1 &
WD=$!
wait $RUNNER; rc=$?
kill $WD 2>/dev/null; wait $WD 2>/dev/null
say "runner ended after $(( $(date +%s) - T0 )) s with rc=$rc"
python3 - $OUT/driver.log $OUT/watchdog.log $OUT <<'PY' | tee -a $LOG
import re,sys,os
from fpga_driver.transcript import read, scope_to_run, uart_text, strip_markers, find_all, require, arm_segments, TranscriptError
whole=read(sys.argv[1]); framed=scope_to_run(whole)
try: joined=require(uart_text(framed), f"UART after this run's load_image in {sys.argv[1]}")
except TranscriptError as e: print(f"  ERROR: {e}"); sys.exit(2)
s=strip_markers(joined)                      # the domain's lines (markers deleted); joined keeps the markers
wd=read(sys.argv[2]) if len(sys.argv)>2 and os.path.exists(sys.argv[2]) else ''
out=sys.argv[3]; t=joined; flat=s                 # flat: the domain's lines, seam-joined and marker-free
lines=[]; buf=None
for l in flat.split('\n'):
    if buf is not None:
        buf+=l
        if re.search(r' ok=[01]$', buf): lines.append(buf); buf=None
        continue
    if l.startswith('R1 s=') and not re.search(r' ok=[01]$', l): buf=l; continue
    if l.startswith('R1 '): lines.append(l)
open(out+'/r1-lines.txt','w').write('\n'.join(lines)+'\n')
pts=[l for l in lines if l.startswith('R1 s=')]; ends=[l for l in lines if l.startswith('R1 end')]; ref=[l for l in lines if l.startswith('R1 refused')]
print(f"  R1 point lines: {len(pts)} (expect 4-5 per invocation); ok=1: {sum(' ok=1' in p for p in pts)}; bad!=0: {sum(' bad=0' not in p for p in pts)}; R1 end lines: {len(ends)} (expect one per invocation); refused: {len(ref)}")
for pat in (r'RESULT k800 retval=[0-9-]+', r'speedtest1-ran=\d+', r'released pool rc=\d+', r'released tables rc=\d+', r'HARD STOP|ENTRY-STALL|ABORT', r'domain halted[^\n]{0,60}'):
    m=find_all(pat, whole+wd if pat.startswith('HARD STOP') else flat); print(f"  {pat[:28]:28}: {len(m)}x {sorted(set(m))[:6]}")
print("  declared lines per invocation:", [int(x) for x in re.findall(r'lines=(\d+)', '\n'.join(ends))])
print("  boot banners after this run's load_image (must be 1):", max(flat.count('OpenSBI v'), flat.count('Linux version')))
PY
srs=${PIPESTATUS[0]}
# The marker says what happened, and the exit status agrees (ISSUES M-11): a refused launch is not "done".
if [ "$rc" -ne 0 ] && grep -Eq 'preflight:? BLOCKED' "$OUT/driver.log" 2>/dev/null; then
  echo refused > "$OUT/marker"; say "refused: preflight BLOCKED (runner rc=$rc)"; exit 1; fi
[ "$rc" -eq 0 ]  || { echo failed > "$OUT/marker"; say "failed: runner rc=$rc"; exit 1; }
[ "$srs" -eq 0 ] || { echo failed > "$OUT/marker"; say "failed: summary rc=$srs (no transcript after load_image, or an unparseable frame)"; exit 1; }
echo done > "$OUT/marker"; say "done"
