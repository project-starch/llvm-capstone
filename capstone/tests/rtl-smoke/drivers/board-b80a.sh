#!/bin/bash
# Boot sw80a -- P1 at -O2, the optimisation level the protocol requires: cell 5 at -O2 (d61c8bf784f2bbd1:
# memsys5 + lookaside 1200,40, 2 MiB static heap, stack declaration 385,024) with the readback host
# 2af56927aaf907e9, and the native baseline at -O2 (b36eb3814c3cefce, lookaside 1200,40) in warm mode
# with --stats. QEMU (icount, 2026-09-14 21:0x): cell 5 -O2 330,723,308 instructions at the oracle with
# HEAP 2097152; native -O2 240,654,449 (BASELINE-WARM) with 25,122 lookasides. The -O0 rows of record are
# 678,572,868 / 535,283,834 on QEMU and 2,551,483,818 / 2,108,202,651 cycles on silicon (CPI 3.76 / 3.94).
# Pre-registered, written before the boot:
#   arm 1  lpc + k800.dom                       RESULT k800 retval=4, or the boot is VOID
#   arm 2  cell 5 -O2, --size 1                 112006 38bb59fd, HEAP 2097152 DROPPED 0 RC 0; cycles = 330.7 M x CPI,
#                                               band 1.06 G .. 1.62 G (CPI 3.2..4.9; the -O0 CPI 3.76 would give 1.244 G)
#   arm 3  lpc + k800.dom                       retval=4
#   arm 4  native -O2, warm --size 1 --stats    112006 38bb59fd, Successful lookasides 25122; BASELINE-WARM CYCLES = 240.65 M x CPI,
#                                               band 0.72 G .. 1.20 G (CPI 3.0..5.0; the -O0 CPI 3.94 would give 0.948 G)
# The ratio cell5-O2/native-O2 is P1's protection_cost denominator side at O2 (custom-spatial vs custom-plain).
# BUDGET 900 s per arm; watchdog 420 s.
set -u; R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd); U=${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}/unify
B=$R/capstone/caplifive-system/sw/buildroot; BM=$B/components/opensbi/lib/sbi/capstone-sbi
FW=${CAPSTONE_BR_FW:-$B/build-fpga/build/opensbi-custom/build/platform/fpga/ariane/firmware}
CC=$R/capstone/capstone-c
MEMLOCK=${MEMLOCK:-$HOME/bin/logs/machine-memory.lock}
OUT=$U/board-b80a; mkdir -p $OUT; LOG=$OUT/log; : > $LOG; rm -f $OUT/marker
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
[ "$(sha256sum $IMG|cut -c1-16)" = d61c8bf784f2bbd1 ] || fail "cell 5 -O2 image is not d61c8bf784f2bbd1"
[ -f ${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}/qemu-pass/$(sha256sum $IMG|cut -d' ' -f1) ] || fail "no QEMU pass record for cell 6"
RRH=${SQLITE_HOST_USER:?set SQLITE_HOST_USER=<path to sqlite_host.user>}
[ "$(sha256sum $RRH|cut -c1-16)" = 2af56927aaf907e9 ] || fail "host is not the readback build 2af56927aaf907e9"
NAT=${NATIVE_BASELINE:?set NATIVE_BASELINE=<path to speedtest1_baseline>}
[ "$(sha256sum $NAT|cut -c1-16)" = b36eb3814c3cefce ] || fail "native -O2 baseline is not b36eb3814c3cefce"
grep -aq 'SPEEDTEST1-CYCLES 330723308 HIGHWATER n/a HEAP 2097152' ${QEMU_LOG_CELL:?} || fail "the -O2 QEMU run of cell 5 is not on record"
grep -aq 'BASELINE-WARM CYCLES 240654449' ${QEMU_LOG_BASELINE:?} || fail "the -O2 QEMU run of the baseline is not on record"
LPC=${CAPSTONE_BR_OVERLAY:-$B/overlay/test-domains}/lpc
[ "$(sha256sum $LPC|cut -c1-16)" = 3b93a2b6e2adfa36 ] || fail "lpc on the overlay is not 3b93a2b6e2adfa36"
[ -f "$B/overlay/test-domains/k800.dom" ] || fail "k800.dom missing from the overlay"
say "pieces: cell 5 -O2 d61c8bf784f2bbd1 (pass record present), readback host 2af56927aaf907e9, native -O2 b36eb3814c3cefce, lpc 3b93a2b6e2adfa36; QEMU -O2 runs on record (330,723,308 / 240,654,449)"
T=${CAPSTONE_BR_OVERLAY:-$B/overlay/test-domains}; TT=${CAPSTONE_BR_TARGET:-$B/build-fpga/target/test-domains}; mkdir -p "$T" "$TT"
for s in speedtest1.dom sqlite_host.user speedtest1_baseline; do cp -f "$T/$s" $OUT/$s.before 2>/dev/null; done
cp -f $IMG "$T/speedtest1.dom" || fail "stage cell 5 -O2"
cp -f $RRH "$T/sqlite_host.user" || fail "stage host"
cp -f $NAT "$T/speedtest1_baseline" && chmod 0755 "$T/speedtest1_baseline" || fail "stage native -O2"
STASH=$OUT/retired; mkdir -p $STASH; RESTORED=0
RETIRE="rtpc bigregion.user trapctl.dom fillsd.dom fillwarm.dom fillcost.dom fillnop.dom speedtest1_seven.dom sqlite_host_rr.user"
restore(){ [ "$RESTORED" = 1 ] && return 0; RESTORED=1
  for s in speedtest1.dom sqlite_host.user speedtest1_baseline; do [ -f $OUT/$s.before ] && { cp -f $OUT/$s.before "$T/$s"; cp -f $OUT/$s.before "$TT/$s"; }; done
  for f in $RETIRE; do [ -f "$STASH/$f" ] && { cp -f "$STASH/$f" "$T/$f"; cp -f "$STASH/$f" "$TT/$f"; }; done
  bake restore && say "rebaked with the retired set back" || say "WARN: restore rebake failed -- the next boot MUST rebake"; }
trap restore EXIT
for f in $RETIRE; do [ -f "$T/$f" ] && cp -f "$T/$f" "$STASH/$f"; done
for f in $RETIRE; do rm -f "$T/$f" "$TT/$f"; done
for f in lpc sqlite_host.user speedtest1_baseline speedtest1.dom k800.dom; do cp -f "$T/$f" "$TT/$f"; done
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
want={'speedtest1.dom':'d61c8bf784f2bbd1','lpc':'3b93a2b6e2adfa36','sqlite_host.user':'2af56927aaf907e9','speedtest1_baseline':'b36eb3814c3cefce'}
bad=0
for d in ('overlay/test-domains','build/target/test-domains'):
    for f,exp in want.items():
        p=os.path.join(L,d,f)
        got=hashlib.sha256(open(p,'rb').read()).hexdigest()[:16] if os.path.exists(p) else 'ABSENT'
        ok=got==exp; bad+=0 if ok else 1
        print(f"  {d:26} {f:20} {got}  {'ok' if ok else 'MISMATCH exp '+exp}")
cpio=open(L+'/build/images/rootfs.cpio','rb').read()
miss=[f for f in ['k800.dom','lpc','speedtest1.dom','sqlite_host.user','speedtest1_baseline'] if cpio.find(open(L+'/overlay/test-domains/'+f,'rb').read()[:8192])<0]
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
export FPGA_BITSTREAM=${FPGA_BITSTREAM:-caplifive_r30r31_1bfff7776.bit} FPGA_BITSTREAM_UNVERIFIED=1
export PREFLIGHT_ALLOW_SHORT=1 PREFLIGHT_ALLOW_SLOTS=1
export ENTRY_STALL_S=420 EARLY_HALT_CONTROL=0 WEDGE_TRACER=0 HALT_MUX_READS=0
export SQLITE_HOST=/test-domains/sqlite_host.user
BUDGET=900; export SQLITE_STAGE_TIMEOUT=$BUDGET SQLITE_IDLE_S=$BUDGET
D="/test-domains/lpc|k800:/test-domains/k800.dom"
D="$D,/test-domains/speedtest1.dom:--speedtest1 --testset main --size 1 --verify"
D="$D,/test-domains/lpc|k800:/test-domains/k800.dom"
D="$D,/test-domains/speedtest1_baseline|warm:--testset main --size 1 --verify --stats"
export SQLITE_STAGE_DOMS="$D" PROBE_SCOPED_OUT=$OUT/boot.txt PROBE_RAW_OUT=$OUT/boot-raw.txt
say "=== boot sw80a = P1 at -O2: control, cell 5 -O2 (size 1), control, native -O2 warm --stats ==="
say "=== pre-registered: retval=4 twice; cell 5 -O2: 112006 38bb59fd, HEAP 2097152 DROPPED 0, cycles = 330.7 M x CPI, band 1.06 G .. 1.62 G (CPI 3.2..4.9; -O0 CPI 3.76 gives 1.244 G); native -O2: 112006 38bb59fd, Successful lookasides 25122, BASELINE-WARM cycles = 240.65 M x CPI, band 0.72 G .. 1.20 G (CPI 3.0..5.0; -O0 CPI 3.94 gives 0.948 G) ==="
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
python3 - $OUT/driver.log $OUT/watchdog.log <<'PY' | tee -a $LOG
import re,sys,os
from fpga_driver.transcript import read, scope_to_run, uart_text, strip_markers, find_all, require, arm_segments, TranscriptError
whole=read(sys.argv[1]); framed=scope_to_run(whole)
try: joined=require(uart_text(framed), f"UART after this run's load_image in {sys.argv[1]}")
except TranscriptError as e: print(f"  ERROR: {e}"); sys.exit(2)
s=strip_markers(joined)                      # the domain's lines (markers deleted); joined keeps the markers
wd=read(sys.argv[2]) if len(sys.argv)>2 and os.path.exists(sys.argv[2]) else ''
for pat in (r'RESULT k800 retval=[0-9-]+', r'Verification Hash: \d+ [0-9a-f]{8}', r'SPEEDTEST1-CYCLES \d+', r'BASELINE-WARM CYCLES \d+ INSTRS \d+', r'Successful lookasides:\s+\d+', r'HEAP \d+ DROPPED \d+ RC \d+', r'sublet: split=\d+ mrev=\d+ delin=\d+', r'ALEN:[0-9A-F]{8}', r'SQ: released pool rc=\d+', r'RC(LM|PR|SH|RE|EN|CU):[0-9A-F]{8}', r'RR/[a-zA-Z-]+', r'SQ: share-trap=\d+', r'HARD STOP|ENTRY-STALL|ABORT'):
    src = joined if pat.startswith(('ALEN:','RC(','RR/')) else (whole+wd if pat.startswith('HARD STOP') else s)
    f=find_all(pat,src); print(f"  {pat[:30]:30}: {len(f)}x {f[-6:]}")
m=re.findall(r'SPEEDTEST1-CYCLES (\d+) HIGHWATER', s)   # anchored on the next token
if m: c=int(m[0]); print(f"  cell 5 -O2 cycles {c:,}; CPI vs QEMU 330,723,308: {c/330723308:.3f}; in band: {1060000000<=c<=1620000000}")
n=re.findall(r'BASELINE-WARM CYCLES (\d+) INSTRS', s)
if n: b=int(n[-1]); print(f"  native -O2 warm cycles {b:,}; CPI vs QEMU 240,654,449: {b/240654449:.3f}; in band: {720000000<=b<=1200000000}; ratio cell5-O2/native-O2: {c/b:.4f}") if m else None
print("  boot banners after this run's load_image (must be 1):", max(len(find_all(r'OpenSBI v', s)), len(find_all(r'Linux version', s))))
PY
srs=${PIPESTATUS[0]}
# The marker says what happened, and the exit status agrees (ISSUES M-11): a refused launch is not "done".
if [ "$rc" -ne 0 ] && grep -Eq 'preflight:? BLOCKED' "$OUT/driver.log" 2>/dev/null; then
  echo refused > "$OUT/marker"; say "refused: preflight BLOCKED (runner rc=$rc)"; exit 1; fi
[ "$rc" -eq 0 ]  || { echo failed > "$OUT/marker"; say "failed: runner rc=$rc"; exit 1; }
[ "$srs" -eq 0 ] || { echo failed > "$OUT/marker"; say "failed: summary rc=$srs (no transcript after load_image, or an unparseable frame)"; exit 1; }
echo done > "$OUT/marker"; say "done"
