#!/bin/bash
# Boot sw80b -- P1 at -O2 (the level the protocol requires): cell 6 at -O2 (c506694f9f6f6889, Sublet, lookaside ON) at the 2 MiB
# arena, --arena 2097152, the same backing limit as cell 5's 2 MiB static heap. Same program, same rr host
# (2c9e82d101b48160) as sw74; only the host argument changes. QEMU (icount, 2026-09-14 20:21): 692,392,094
# instructions, HEAP 1344064 (the port carves 64.1 % of the arena for memsys5, so HEAP is not 2 MiB),
# sublet split=5568 mrev=37966 delin=32565, oracle 112006 38bb59fd; the 1,419,584 reference re-run the same
# minute reproduced 690,051,663 / HEAP 911104 / 5481 / 37874 exactly.
# Pre-registered, written before the boot:
#   arm 1  lpc + k800.dom                              RESULT k800 retval=4, or the boot is VOID
#   arm 2  cell 6, --arena 2097152 --tables 1750285, --size 1 (the boot's ONE Sublet workload):
#          112006 38bb59fd, HEAP 1344064, sublet 5568/37966/32565, cycles 2,803,660,693 +-1 %
#          (2,794,183,730 x 692,392,094/690,051,663; band 2,775.6 M .. 2,831.7 M); teardown probe:
#          SQ: released pool rc=1, RCLM 0 -> 1, no RCPR/RCSH/RCRE
#   arm 3  lpc + k800.dom                              retval=4
#   arm 4  probe --tail --arena 2097152 --tables 1750285 (representable, no rounding): ALEN:00200000, RR/done, RCLM 0 -> 1
# BUDGET 900 s per arm; watchdog 420 s.
set -u; R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd); U=${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}/unify
B=$R/capstone/caplifive-system/sw/buildroot; BM=$B/components/opensbi/lib/sbi/capstone-sbi
FW=${CAPSTONE_BR_FW:-$B/build-fpga/build/opensbi-custom/build/platform/fpga/ariane/firmware}
CC=$R/capstone/capstone-c
MEMLOCK=${MEMLOCK:-$HOME/bin/logs/machine-memory.lock}
OUT=$U/board-b80b; mkdir -p $OUT; LOG=$OUT/log; : > $LOG; rm -f $OUT/marker
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
IMG=${CELL_IMG:?set CELL_IMG=<path to the cell image>}
[ "$(sha256sum $IMG|cut -c1-16)" = c506694f9f6f6889 ] || fail "cell 6 -O2 image is not c506694f9f6f6889"
[ -f ${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}/qemu-pass/$(sha256sum $IMG|cut -d' ' -f1) ] || fail "no QEMU pass record for cell 6"
RRH=${R1_HOST:?set R1_HOST=<path to sqlite_host_rr.user, the readback host>}
[ "$(sha256sum $RRH|cut -c1-16)" = "${R1_HOST_HASH:-2c9e82d101b48160}" ] || fail "rr host is not ${R1_HOST_HASH:-2c9e82d101b48160}"
[ "$(strings $RRH | grep -c 'RR/share')" -ge 1 ] || fail "rr host lacks the revoke-reshare probe"
grep -aq 'SPEEDTEST1-CYCLES 338496909 HIGHWATER n/a HEAP 911104' ${QEMU_LOG_CELL:?} || fail "the -O2 QEMU run is not on record"
grep -aq 'SPEEDTEST1-CYCLES 340817186 HIGHWATER n/a HEAP 1344064' ${QEMU_LOG_CELL_2MIB:?} || fail "the -O2 QEMU run at the 2 MiB arena (the denominator this boot uses) is not on record"
LPC=${CAPSTONE_BR_OVERLAY:-$B/overlay/test-domains}/lpc
[ "$(sha256sum $LPC|cut -c1-16)" = 3b93a2b6e2adfa36 ] || fail "lpc on the overlay is not 3b93a2b6e2adfa36"
[ -f "$B/overlay/test-domains/k800.dom" ] || fail "k800.dom missing from the overlay"
say "pieces: cell 6 -O2 c506694f9f6f6889 (pass record present), rr host 2c9e82d101b48160, lpc 3b93a2b6e2adfa36; QEMU -O2 runs on record (338,496,909 at the 1,419,584 arena; 340,817,186 at the 2 MiB arena, 2026-09-15 00:10, same HEAP 1344064 and sublet 5568/37966/32565 as the -O0 2 MiB run — the denominator below)"
T=${CAPSTONE_BR_OVERLAY:-$B/overlay/test-domains}; TT=${CAPSTONE_BR_TARGET:-$B/build-fpga/target/test-domains}; mkdir -p "$T" "$TT"
for s in speedtest1.dom sqlite_host_rr.user; do cp -f "$T/$s" $OUT/$s.before 2>/dev/null; done
cp -f $IMG "$T/speedtest1.dom" || fail "stage cell 6"
cp -f $RRH "$T/sqlite_host_rr.user" || fail "stage rr host"
STASH=$OUT/retired; mkdir -p $STASH; RESTORED=0
RETIRE="rtpc bigregion.user trapctl.dom fillsd.dom fillwarm.dom fillcost.dom fillnop.dom speedtest1_seven.dom speedtest1_baseline"
restore(){ [ "$RESTORED" = 1 ] && return 0; RESTORED=1
  for s in speedtest1.dom sqlite_host_rr.user; do [ -f $OUT/$s.before ] && { cp -f $OUT/$s.before "$T/$s"; cp -f $OUT/$s.before "$TT/$s"; }; done
  for f in $RETIRE; do [ -f "$STASH/$f" ] && { cp -f "$STASH/$f" "$T/$f"; cp -f "$STASH/$f" "$TT/$f"; }; done
  bake restore && say "rebaked with the retired set back" || say "WARN: restore rebake failed -- the next boot MUST rebake"; }
trap restore EXIT
for f in $RETIRE; do [ -f "$T/$f" ] && cp -f "$T/$f" "$STASH/$f"; done
for f in $RETIRE; do rm -f "$T/$f" "$TT/$f"; done
for f in lpc sqlite_host.user sqlite_host_rr.user speedtest1.dom k800.dom; do cp -f "$T/$f" "$TT/$f"; done
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
want={'speedtest1.dom':'c506694f9f6f6889','lpc':'3b93a2b6e2adfa36','sqlite_host_rr.user':'2c9e82d101b48160'}
bad=0
for d in ('overlay/test-domains','build/target/test-domains'):
    for f,exp in want.items():
        p=os.path.join(L,d,f)
        got=hashlib.sha256(open(p,'rb').read()).hexdigest()[:16] if os.path.exists(p) else 'ABSENT'
        ok=got==exp; bad+=0 if ok else 1
        print(f"  {d:26} {f:20} {got}  {'ok' if ok else 'MISMATCH exp '+exp}")
cpio=open(L+'/build/images/rootfs.cpio','rb').read()
miss=[f for f in ['k800.dom','lpc','speedtest1.dom','sqlite_host_rr.user'] if cpio.find(open(L+'/overlay/test-domains/'+f,'rb').read()[:8192])<0]
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
BUDGET=900; export SQLITE_STAGE_TIMEOUT=$BUDGET SQLITE_IDLE_S=$BUDGET
D="/test-domains/lpc|k800:/test-domains/k800.dom"
D="$D,/test-domains/sqlite_host_rr.user|/test-domains/speedtest1.dom:--speedtest1 --arena 2097152 --tables 1750285 --testset main --size 1 --verify"
D="$D,/test-domains/lpc|k800:/test-domains/k800.dom"
D="$D,/test-domains/sqlite_host_rr.user|/test-domains/speedtest1.dom:--tail --arena 2097152 --tables 1750285"
export SQLITE_STAGE_DOMS="$D" PROBE_SCOPED_OUT=$OUT/boot.txt PROBE_RAW_OUT=$OUT/boot-raw.txt
say "=== boot sw80b = P1 at -O2: cell 6 -O2 at the 2 MiB arena, control, cell, control, probe ==="
say "=== pre-registered: retval=4; 112006 38bb59fd + HEAP 1344064 + sublet 5568/37966/32565 (the 2 MiB carve, as at -O0); cycles = 340.8 M x CPI (the 2 MiB-arena QEMU count 340,817,186): band 1.09 G .. 1.67 G (CPI 3.2..4.9; -O0 CPI 4.05 would give 1.371 G); teardown released pool rc=1, RCLM 0->1, no RCPR/RCSH/RCRE; probe ALEN:00200000, RR/done ==="
for i in $(seq 1 30); do c=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' "$FPGA_URL" 2>/dev/null); [ "$c" != "000" ] && break; say "console down ($i)"; sleep 60; done
cd $R/capstone/tests/rtl-smoke
T0=$(date +%s)
timeout $((4*BUDGET + 2400)) python3 -m fpga_driver.run_sqlite_stages_fpga > $OUT/driver.log 2>&1 &
RUNNER=$!
ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=${ENTRY_STALL_S:-420} \
  bash board-watchdog.sh "$OUT/driver.log" "${WD_IDLE:-900}" "$RUNNER" > $OUT/watchdog.log 2>&1 &
WD=$!
wait $RUNNER; rc=$?
kill $WD 2>/dev/null; wait $WD 2>/dev/null
say "runner ended after $(( $(date +%s) - T0 )) s with rc=$rc"
python3 - $OUT/driver.log <<'PY' | tee -a $LOG
import re,sys
s=open(sys.argv[1],'rb').read().decode('latin1'); i=s.rfind("monitor load_image"); s=s[i:] if i>=0 else s
for pat in (r'RESULT k800 retval=[0-9-]+', r'Verification Hash: \d+ [0-9a-f]{8}', r'SPEEDTEST1-CYCLES \d+', r'HEAP \d+ DROPPED \d+ RC \d+', r'sublet: split=\d+ mrev=\d+ delin=\d+', r'ALEN:[0-9A-F]{8}', r'SQ: released pool rc=\d+', r'RC(LM|PR|SH|RE|EN|CU):[0-9A-F]{8}', r'RR/[a-zA-Z-]+', r'SQ: share-trap=\d+', r'HARD STOP|ENTRY-STALL|ABORT'):
    f=re.findall(pat,s); print(f"  {pat[:30]:30}: {len(f)}x {f[-6:]}")
m=re.findall(r'SPEEDTEST1-CYCLES (\d+)', s)
if m: c=int(m[0]); print(f"  cycles {c:,}; CPI vs QEMU 340,817,186 (the 2 MiB-arena -O2 count): {c/340817186:.3f}; in band: {1090000000<=c<=1670000000}")
print("  boot banners after this run's load_image (must be 1):", len(re.findall(r'OpenSBI v', s)))
PY
echo done > $OUT/marker; say "done"
