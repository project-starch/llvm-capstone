#!/bin/bash
# board-c6var.sh -- one boot of a cell-6 VARIANT image at the 2 MiB arena (control, cell, control, probe), parametrised:
#   C6_TAG=<name> C6_IMG=<dom> C6_HASH=<sha256/16> C6_QEMU_DEFAULT=<icount at the default arena> C6_QEMU_DEFAULT_LOG=<log>
#   C6_QEMU_2MIB=<icount at --arena 2097152> C6_QEMU_2MIB_LOG=<log> C6_EXPECT="<free text for the pre-registration>"
#   C6_ARENA=<bytes, default 2097152> C6_TABLES=<bytes, default 1750285> C6_DEFAULT_ARENA=<bytes, default 1419584>
#   C6_QEMU_DEFAULT_CFG=<host stdout capture of the measure run> C6_QEMU_2MIB_CFG=<same, 2 MiB run>
# TWO ARTEFACTS PER RECORD, BECAUSE THE TWO LINES ARE IN TWO STREAMS. `SPEEDTEST1-CYCLES` is emitted
# by the GUEST (speedtest1_measure.c:469) into the serial log the flow names with --log-file
# (run-speedtest1-measure.sh:238), while `== Sublet: pool ...` is a HOST shell echo (:167) written to
# the script's own stdout before QEMU starts, and the script never redirects its stdout into that log.
# So one file cannot carry both, and the first version of this gate (2026-09-16) pointed both greps at
# C6_QEMU_*_LOG and could not pass on any artefact the flow produces. Each grep now reads the file its
# own producer wrote. A COMBINED transcript was offered and declined on purpose: a concatenation is
# something no tool emitted natively, a later reader cannot tell it from a hand-assembled record, and
# that is the exact property this gate exists to refuse.
# THE ARENA AND TABLES ARE ONE SOURCE OF TRUTH FOR THE GATE AND THE BOOT, which they were not before
# 2026-09-16. The emulator gate matched only the SPEEDTEST1-CYCLES line, whose HEAP field is
# sublet_heap_len() = sublet_tables_len, computed in speedtest1_measure.c:105-113 from the GRANTED
# ARENA alone -- atoms = (end-base)/64 -- and never from the --tables argument, which the line does
# not emit at all. So `HEAP 1344064` says the arena was 2 MiB and says NOTHING about tables: a record
# produced by the committed flow at that arena carries tables 2523136 (both derive from
# SPEEDTEST1_POOL, atoms = POOL/65), the boot below runs tables 1750285, and the two are
# INDISTINGUISHABLE BY CONSTRUCTION -- a clean pass against a denominator from a different
# configuration. Found by the compiler lane while preparing F1's records and verified against the
# driver and the emitter by the board lane on apollo. The gate now also demands the flow's own
# configuration line, which names BOTH numbers, built from the same variables the boot uses.
# Written 2026-09-15 for the optnone bisection of the -O1/-O2 counter divergence (measurements §7s).
set -u; R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd); U=${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}/unify
B=$R/capstone/caplifive-system/sw/buildroot; BM=$B/components/opensbi/lib/sbi/capstone-sbi
FW=${CAPSTONE_BR_FW:-$B/build-fpga/build/opensbi-custom/build/platform/fpga/ariane/firmware}
CC=$R/capstone/capstone-c
MEMLOCK=${MEMLOCK:-$HOME/bin/logs/machine-memory.lock}
OUT=$U/board-${C6_TAG:?}; mkdir -p $OUT; LOG=$OUT/log; : > $LOG; rm -f $OUT/marker
say(){ echo "$(date +%T) $*" | tee -a $LOG; }; fail(){ say "FAIL: $*"; echo failed > $OUT/marker; exit 1; }
bake(){ say "bake $1: waiting for the machine memory lock (another lane may be measuring)"
  flock -w 7200 "$MEMLOCK" bash -c '
    B=$1; CC=$2; OUT=$3; tag=$4
    for a in modcapstone-rebuild linux-rebuild opensbi-rebuild; do
      ( cd $B && make build LINUX_PAYLOAD=1 A=$a CAPSTONE_CC_PATH=$CC ) > $OUT/bake-$tag-$a.log 2>&1 || exit 1
    done' _ "$B" "$CC" "$OUT" "$1"; }
pgrep -f 'python3 -m fpga_driver' >/dev/null && fail "a board runner is live"
[ "$(git -C $BM rev-parse --short HEAD)" = 4274268 ] || fail "monitor is not 4274268"
cd $B || fail "no FPGA buildroot copy"
[ "$(git rev-parse --short HEAD)" = d04bd83 ] || fail "FPGA copy is at $(git rev-parse --short HEAD), not d04bd83 (the #3 module)"
IMG=${C6_IMG:?}; export C6_HASH C6_QEMU_2MIB
[ "$(sha256sum $IMG|cut -c1-16)" = ${C6_HASH:?} ] || fail "cell 6 variant ${C6_TAG} image is not ${C6_HASH:?}"
[ -f ${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}/qemu-pass/$(sha256sum $IMG|cut -d' ' -f1) ] || fail "no QEMU pass record for cell 6"
RRH=${R1_HOST:?set R1_HOST=<path to sqlite_host_rr.user, the readback host>}
[ "$(sha256sum $RRH|cut -c1-16)" = "${R1_HOST_HASH:-2c9e82d101b48160}" ] || fail "rr host is not ${R1_HOST_HASH:-2c9e82d101b48160}"
[ "$(strings $RRH | grep -c 'RR/share')" -ge 1 ] || fail "rr host lacks the revoke-reshare probe"
grep -aq "SPEEDTEST1-CYCLES ${C6_QEMU_DEFAULT:?} HIGHWATER n/a HEAP 911104" ${C6_QEMU_DEFAULT_LOG:?} || fail "the variant's QEMU run at the default arena is not on record"
grep -aq "SPEEDTEST1-CYCLES ${C6_QEMU_2MIB:?} HIGHWATER n/a HEAP 1344064" ${C6_QEMU_2MIB_LOG:?} || fail "the variant's QEMU run at the 2 MiB arena (the denominator this boot uses) is not on record"
# WHY THE DEFAULT GATE DEMANDS HEAP 911104 AND NOT THE 910008 THE FORMULA GIVES: the grant is rounded
# up from the request. 1419584 is 22181 atoms and would give 910008; the granted region is 1421312,
# 22208 atoms, which gives 911104 -- and 911104 over 1421312 is attested by a separate 2026-09-15 run.
# The rounding granularity is NOT settled: 2048 and 4096 both reproduce 1421312 and this data cannot
# separate them, because 347 pages of 4096 is 694 of 2048, and the 2 MiB point is 512 exact pages at
# every candidate so it constrains nothing. Both agree at the two arenas in use, so nothing here
# depends on it -- but a prediction at a NEW arena differs between them (a 1700000 request gives
# 1701888 under 2048 and 1703936 under 4096). Reading the grant path would settle it; nobody has.
# DO NOT "fix" the 911104 to match the formula on the requested arena: that value is what real runs
# produce, and a gate moved to 910008 would fail on every genuine record while looking tightened.
C6_ARENA=${C6_ARENA:-2097152}; C6_TABLES=${C6_TABLES:-1750285}; C6_DEFAULT_ARENA=${C6_DEFAULT_ARENA:-1419584}
# The configuration line run-speedtest1-measure.sh:167 emits. It names arena AND tables, so unlike the
# HEAP field it can tell the two configurations apart -- and a hand-assembled record that cannot show
# its configuration fails here, which is the point: refusing is correct, passing unverifiably is not.
CFG2="== Sublet: pool ${C6_ARENA} bytes (arena, REV_BORROWED), tables ${C6_TABLES} bytes"
CFGD="== Sublet: pool ${C6_DEFAULT_ARENA} bytes (arena, REV_BORROWED), tables ${C6_TABLES} bytes"
grep -aqF "$CFGD" ${C6_QEMU_DEFAULT_CFG:?set C6_QEMU_DEFAULT_CFG=<host stdout of the default-arena measure run>} || fail "the default-arena record's host stdout does not show its configuration: expected '$CFGD'"
grep -aqF "$CFG2" ${C6_QEMU_2MIB_CFG:?set C6_QEMU_2MIB_CFG=<host stdout of the 2 MiB measure run>} || fail "the 2 MiB record's host stdout does not show the configuration this boot runs: expected '$CFG2' (HEAP alone cannot distinguish it -- see the header)"
LPC=${CAPSTONE_BR_OVERLAY:-$B/overlay/test-domains}/lpc
[ "$(sha256sum $LPC|cut -c1-16)" = 3b93a2b6e2adfa36 ] || fail "lpc on the overlay is not 3b93a2b6e2adfa36"
[ -f "$B/overlay/test-domains/k800.dom" ] || fail "k800.dom missing from the overlay"
say "pieces: cell 6 variant ${C6_TAG} ${C6_HASH:?} (pass record present), rr host 2c9e82d101b48160, lpc 3b93a2b6e2adfa36; QEMU -O1 runs on record (${C6_QEMU_DEFAULT} at the 1,419,584 arena; ${C6_QEMU_2MIB} at the 2 MiB arena, 2026-09-15 00:10, same HEAP 1344064 and sublet 5568/37966/32565 as the -O0 2 MiB run — the denominator below)"
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
want={'speedtest1.dom':os.environ['C6_HASH'],'lpc':'3b93a2b6e2adfa36','sqlite_host_rr.user':'2c9e82d101b48160'}
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
export FPGA_BITSTREAM=${FPGA_BITSTREAM:-caplifive_r30r31_1bfff7776.bit} FPGA_BITSTREAM_UNVERIFIED=1
export PREFLIGHT_ALLOW_SHORT=1 PREFLIGHT_ALLOW_SLOTS=1
export ENTRY_STALL_S=420 EARLY_HALT_CONTROL=0 WEDGE_TRACER=0 HALT_MUX_READS=0
export SQLITE_HOST=/test-domains/sqlite_host_rr.user
BUDGET=900; export SQLITE_STAGE_TIMEOUT=$BUDGET SQLITE_IDLE_S=$BUDGET
D="/test-domains/lpc|k800:/test-domains/k800.dom"
D="$D,/test-domains/sqlite_host_rr.user|/test-domains/speedtest1.dom:--speedtest1 --arena ${C6_ARENA} --tables ${C6_TABLES} ${C6_ARGS:---testset main --size 1 --verify}"
D="$D,/test-domains/lpc|k800:/test-domains/k800.dom"
D="$D,/test-domains/sqlite_host_rr.user|/test-domains/speedtest1.dom:--tail --arena ${C6_ARENA} --tables ${C6_TABLES}"
export SQLITE_STAGE_DOMS="$D" PROBE_SCOPED_OUT=$OUT/boot.txt PROBE_RAW_OUT=$OUT/boot-raw.txt
say "=== boot ${C6_TAG} = cell 6 variant ${C6_TAG} (${C6_HASH:?}) at the 2 MiB arena, control, cell, control, probe ==="
say "=== pre-registered: retval=4 twice; 112006 38bb59fd; ${C6_EXPECT:-counters EITHER the emulator's 5568/37966/32565/37966/5401 OR sw80b's 8654/41293/32638/41287/8649}; cycles = ${C6_QEMU_2MIB} x CPI 3.2..4.9; probe ALEN:00200000, RR/done ==="
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
for pat in (r'RESULT k800 retval=[0-9-]+', r'Verification Hash: \d+ [0-9a-f]{8}', r'SPEEDTEST1-CYCLES \d+', r'Successful lookasides:\s+\d+', r'Lookaside size faults:\s+\d+', r'HEAP \d+ DROPPED \d+ RC \d+', r'sublet: split=\d+ mrev=\d+ delin=\d+', r'ALEN:[0-9A-F]{8}', r'SQ: released pool rc=\d+', r'RC(LM|PR|SH|RE|EN|CU):[0-9A-F]{8}', r'RR/[a-zA-Z-]+', r'SQ: share-trap=\d+', r'HARD STOP|ENTRY-STALL|ABORT'):
    src = joined if pat.startswith(('ALEN:','RC(','RR/')) else (whole+wd if pat.startswith('HARD STOP') else s)
    f=find_all(pat,src); print(f"  {pat[:30]:30}: {len(f)}x {f[-6:]}")
m=re.findall(r'SPEEDTEST1-CYCLES (\d+) HIGHWATER', s)   # the value reads anchored on the NEXT field, so a cut number cannot read as the count
import os; q=int(os.environ['C6_QEMU_2MIB'])
if m: c=int(m[0]); print(f"  cycles {c:,}; CPI vs this image's emulator count {q:,} at 2 MiB: {c/q:.3f}; in band: {q*3.2<=c<=q*4.9}")
print("  boot banners after this run's load_image (must be 1):", max(len(find_all(r'OpenSBI v', s)), len(find_all(r'Linux version', s))))
PY
srs=${PIPESTATUS[0]}
# The marker says what happened, and the exit status agrees (ISSUES M-11): a refused launch is not "done".
if [ "$rc" -ne 0 ] && grep -Eq 'preflight:? BLOCKED' "$OUT/driver.log" 2>/dev/null; then
  echo refused > "$OUT/marker"; say "refused: preflight BLOCKED (runner rc=$rc)"; exit 1; fi
[ "$rc" -eq 0 ]  || { echo failed > "$OUT/marker"; say "failed: runner rc=$rc"; exit 1; }
[ "$srs" -eq 0 ] || { echo failed > "$OUT/marker"; say "failed: summary rc=$srs (no transcript after load_image, or an unparseable frame)"; exit 1; }
echo done > "$OUT/marker"; say "done"
