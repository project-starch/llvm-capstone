#!/bin/bash
# E1 of the 2026-09-14 Sublet-paper plan: the S1/S2 safety matrix ON HARDWARE. One boot = one group of
# the s2/3 branch's probe images (cells.tsv, column 5), each linked at its own entry VA (R-3 / C15),
# every returning cell first and the boot's one FAULT cell LAST: on this RTL a capability fault inside
# a domain is a wedge (M-1), so it takes the core and ends the boot. REP and BOOT select the group.
# Pre-registered per cell (cells.tsv column 4): the mark the emulator returned (low 24 bits of
# `ngx retval`), `GOOD` for the PostgreSQL tests, and for the FAULT cell NO `ngx retval` line, a wedge
# read by the tracer with a capability cause in the trap log. Control k800 = 4 first, or VOID.
set -u; R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd); U=${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}/unify
B=$R/capstone/caplifive-system/sw/buildroot; BM=$B/components/opensbi/lib/sbi/capstone-sbi
FW=${CAPSTONE_BR_FW:-$B/build-fpga/build/opensbi-custom/build/platform/fpga/ariane/firmware}
CC=$R/capstone/capstone-c
E=${E1_DIR:?set E1_DIR=<the E1 images directory>}
MEMLOCK=${MEMLOCK:-$HOME/bin/logs/machine-memory.lock}
REP=${REP:?rep 1..3}; BOOT=${BOOT:?boot 1..3}
OUT=$U/board-b78-r${REP}b${BOOT}; mkdir -p $OUT; LOG=$OUT/log; : > $LOG; rm -f $OUT/marker
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

# The cell table: name arm stop want boot va entry hash rc line. Returning cells first, FAULT last.
[ -f $E/cells.tsv ] || fail "no cells.tsv"
CELLS=$(awk -F'\t' -v b="$BOOT" '$5==b && $4!="FAULT"' $E/cells.tsv; awk -F'\t' -v b="$BOOT" '$5==b && $4=="FAULT"' $E/cells.tsv)
[ -n "$CELLS" ] || fail "no cells for boot $BOOT"
nf=$(printf '%s\n' "$CELLS" | awk -F'\t' '$4=="FAULT"' | wc -l); [ "$nf" -le 1 ] || fail "$nf FAULT cells in one boot"
T=${CAPSTONE_BR_OVERLAY:-$B/overlay/test-domains}; TT=${CAPSTONE_BR_TARGET:-$B/build-fpga/target/test-domains}; mkdir -p "$T" "$TT"
STAGED=""; D="/test-domains/lpc|k800:/test-domains/k800.dom"; PRE=""
dom_of(){ case $1 in subpool) echo $E/cells/subpool/ngx-subpool.dom;; pgsub) echo $E/cells/pgsub/pg_subpool_test.dom;; pghier) echo $E/cells/pghier/sublet/pg_hierarchy.dom;; *) echo $E/cells/$1/ngx-uaf-$1.dom;; esac; }
while IFS=$'\t' read -r name arm stop want boot va ent hash rc line; do
  src=$(dom_of $name); [ -f "$src" ] || fail "$name: no image $src"
  [ "$(sha256sum $src|cut -c1-16)" = "$hash" ] || fail "$name: image hash is not cells.tsv's $hash"
  if [ "$rc" != 0 ]; then
    grep -q 'Assertion' $E/cells/$name.log 2>/dev/null && say "NOTE $name: the emulator aborts on this cell (capstone-qemu PR #4); board-only, no QEMU pass" || fail "$name: QEMU rc=$rc, not staged"
  fi
  case $name in
    subpool) dst=ngx-subpool.dom; host=/test-domains/ngx-guest; sel="--arena-linear";;
    pgsub)   dst=pg_subpool_test.dom; host=/test-domains/pg_host_sub.user; sel="/test-domains/nothing.bin --tail --linear-arena";;
    pghier)  dst=pg_hierarchy.dom; host=/test-domains/pg_host_hier.user; sel="/test-domains/nothing.bin --tail --linear-arena --scratch 1048576";;
    *)       dst=ngx-uaf-$name.dom; host=/test-domains/ngx-guest; sel=""; [ "$arm" = sublet ] && sel="--arena-linear";;
  esac
  cp -f "$src" "$T/$dst" && cp -f "$src" "$TT/$dst" || fail "stage $dst"
  STAGED="$STAGED $dst"; D="$D,$host|/test-domains/$dst${sel:+:$sel}"
  PRE="$PRE  $name($arm stop $stop) -> $want"
done <<< "$CELLS"
# hosts: one ngx-guest (identical across cells, checked), the two PostgreSQL hosts, the one-byte file
# hosts, only the ones this boot's cells use: one ngx-guest (identical across cells, checked), the
# PostgreSQL hosts (one per test: their arena sizes differ), the one-byte file
if grep -q 'ngx-guest' <<<"$D"; then
  # SAME PROGRAM, not same bytes: the per-cell builds embed libcapstone.c's __FILE__ (an assert
  # string), which is the worktree-symlink path in one pass and the main path in a rerun, so
  # .rodata and every offset into it shift. Pin the SOURCES (unmodified in git, ngx-guest.c = the
  # s2/3 blob) and allow the builds to differ in that one string only.
  PK=$R/capstone/caplifive-buildroot; [ -z "$(git -C $PK status --short package/modcapstone)" ] || fail "modcapstone userspace sources are modified in git"
  [ "$(git -C $R hash-object ${CAPSTONE_ARTIFACTS:-$HOME/capstone-artifacts}/wt-s2/capstone/ports/nginx/tools/ngx-guest.c)" = "$(git -C $R rev-parse origin/s2/3-manager-hierarchy:capstone/ports/nginx/tools/ngx-guest.c)" ] || fail "ngx-guest.c in the worktree is not the s2/3 blob"
  G=$E/share/p1/ngx-guest
  for c in $E/share/*/ngx-guest; do
    d=$(diff <(strings -n 6 $G) <(strings -n 6 $c) | grep -E '^[<>]' | grep -vE 'modcapstone/userspace/lib/libcapstone\.c$' | wc -l)
    [ "$d" -eq 0 ] || fail "ngx-guest differs across cells beyond the libcapstone.c path string: $c"
  done
  say "ngx-guest: staging p1's build $(sha256sum $G|cut -c1-16); the other cells' builds differ only in the embedded libcapstone.c path (libcapstone.c $(git -C $PK log -1 --format=%h -- package/modcapstone/userspace/lib/libcapstone.c))"
  cp -f $G $T/ngx-guest && cp -f $G $TT/ngx-guest && chmod 0755 $T/ngx-guest $TT/ngx-guest || fail "stage ngx-guest"; STAGED="$STAGED ngx-guest"
fi
for pg in sub:pgsub hier:pghier; do tag=${pg%%:*}; cell=${pg##*:}
  grep -q "pg_host_$tag.user" <<<"$D" || continue
  cp -f $E/share/$cell/pg_host.user $T/pg_host_$tag.user && cp -f $E/share/$cell/pg_host.user $TT/pg_host_$tag.user || fail "stage pg_host_$tag"
  grep -q 'PG: result=' $T/pg_host_$tag.user || fail "pg_host_$tag.user is not the PostgreSQL host (no PG: result= string)"
  printf '\0' > $T/nothing.bin; cp -f $T/nothing.bin $TT/nothing.bin; STAGED="$STAGED pg_host_$tag.user nothing.bin"
done
say "pieces: monitor 4274268, module d04bd83; staged for boot $BOOT rep $REP:$STAGED"
say "pre-registered:$PRE"
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
for f in lpc k800.dom; do cp -f "$T/$f" "$TT/$f"; done
say "overlay: $(ls $T | tr '\n' ' ')"
python3 - "$B" <<'PY' | tee -a $LOG
import sys,glob,struct
L=sys.argv[1]; seen={}
for p in sorted(glob.glob(L+'/overlay/test-domains/*')):
    d=open(p,'rb').read()
    if d[:4]!=b'\x7fELF' or not p.endswith('.dom'): continue   # DOMAIN images only: hosts are Linux processes
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
python3 - "$B" "$STAGED" <<'PY' | tee -a $LOG
import sys,os
L=sys.argv[1]; files=sys.argv[2].split()
cpio=open(L+'/build/images/rootfs.cpio','rb').read()
miss=[f for f in files+['k800.dom','lpc'] if cpio.find(open(L+'/overlay/test-domains/'+f,'rb').read()[:4096])<0]
print('initramfs membership:', 'all present' if not miss else 'MISSING '+','.join(miss))
ko=open(L+'/build/target/capstone.ko','rb').read()
print('initramfs carries the #3 module:', 'yes' if cpio.find(ko[:8192])>=0 and b'domain declares dom_data' in cpio else 'NO')
sys.exit(1 if (miss or cpio.find(ko[:8192])<0) else 0)
PY
[ ${PIPESTATUS[0]} -eq 0 ] || fail "initramfs check"

export FPGA_URL="$(cat "${CAPSTONE_FPGA_URL_FILE:-$HOME/.claude-kisp/secrets/fpga-console-url}")"; export FPGA_FW=$FW/fw_payload.bin
# The resident name is a KNOB with the current silicon as its default, so a reflash costs one
# exported variable rather than an edit to each of the seven drivers. Set it to the string the
# console REPORTS as flash_state.nv_bitstream_name after the flash, never to the filename you
# happened to upload. A wrong value hard-stops the boot, which is the safe direction.
export FPGA_BITSTREAM=${FPGA_BITSTREAM:-caplifive_r30r31_1bfff7776.bit} FPGA_BITSTREAM_UNVERIFIED=1
export PREFLIGHT_ALLOW_SHORT=1 PREFLIGHT_ALLOW_SLOTS=1
export ENTRY_STALL_S=420 EARLY_HALT_CONTROL=0 WEDGE_TRACER=0 HALT_MUX_READS=0
export SQLITE_HOST=/test-domains/sqlite_host.user
BUDGET=${BUDGET:-300}; export SQLITE_STAGE_TIMEOUT=$BUDGET SQLITE_IDLE_S=$BUDGET
export SQLITE_STAGE_DOMS="$D" PROBE_SCOPED_OUT=$OUT/boot.txt PROBE_RAW_OUT=$OUT/boot-raw.txt
say "=== boot sw78 r${REP}b${BOOT} = E1 (S1/S2 on hardware), repetition $REP, group $BOOT ==="
say "=== stage list: $D"
for i in $(seq 1 30); do c=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' "$FPGA_URL" 2>/dev/null); [ "$c" != "000" ] && break; say "console down ($i)"; sleep 60; done
cd $R/capstone/tests/rtl-smoke
T0=$(date +%s)
timeout $((8*BUDGET + 2400)) python3 -m fpga_driver.run_sqlite_stages_fpga > $OUT/driver.log 2>&1 &
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
try: arms=require(arm_segments(framed), "'[stages] --> TEST' arms after this run's load_image")
except TranscriptError as e: print(f"  ERROR: {e}"); sys.exit(2)
print("  control:", find_all(r'RESULT k800 retval=[0-9-]+', s))
print("  boot banners after this run's load_image (must be 1):", max(len(find_all(r'OpenSBI v', s)), len(find_all(r'Linux version', s))))
# per arm: the seam-joined UART between its --> and <-- lines (a.uart), the runner's own lines in a.framed
for a in arms:
    m=re.search(r'ngx retval = (\d+)\n', a.uart)          # terminator REQUIRED: a number cut by a chunk seam cannot read as a mark
    pg=re.search(r'__CAPSTONE_PG_([A-Z]+)_(GOOD|BAD|FAILED)__', a.uart)
    tl=re.search(r'TRAP LOG \{seen,mcause\[6:0\]\}\s+(0x[0-9a-f]+)', a.framed)
    v=(f"mark 0x{int(m.group(1))&0xFFFFFF:06X}" if m else f"PG {pg.group(1)} {pg.group(2)}" if pg
       else ('WEDGED trap-log '+(tl.group(1) if tl else '?') if a.returned is False else 'no result line'))
    print(f"  {a.label[:70]:70} -> {v}")
PY
srs=${PIPESTATUS[0]}
# The marker says what happened, and the exit status agrees (ISSUES M-11): a refused launch is not "done".
if [ "$rc" -ne 0 ] && grep -Eq 'preflight:? BLOCKED' "$OUT/driver.log" 2>/dev/null; then
  echo refused > "$OUT/marker"; say "refused: preflight BLOCKED (runner rc=$rc)"; exit 1; fi
[ "$rc" -eq 0 ]  || { echo failed > "$OUT/marker"; say "failed: runner rc=$rc"; exit 1; }
[ "$srs" -eq 0 ] || { echo failed > "$OUT/marker"; say "failed: summary rc=$srs (no transcript after load_image, or an unparseable frame)"; exit 1; }
echo done > "$OUT/marker"; say "done"
