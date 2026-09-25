#!/usr/bin/env bash
# Run the tshark domain images on Capstone QEMU and judge them.
#
#   run-qemu.sh stages                   M1-M4, then M5, on dhcp.pcap (TSAPP_STAGES_CAPTURE=<name>
#                                        for another capture, or <name>.flip): each image in
#                                        its own section, ascending, one boot
#   run-qemu.sh oracle <capture>...      M5 on each capture (a name under test/captures, or
#                                        <name>.flip for the one-byte-flipped copy): one boot
#   run-qemu.sh safety <fixture>...      the safety fixtures (src/tsapp-safety.c), in the order
#                                        given, one boot, judged by host/safety-verdict.py against
#                                        host/safety-expect.txt
#
# TSAPP_HEAP=level0|shrink picks the heap arm's images (build-domain.sh), from $TS_WORK/domain or
# $TS_WORK/domain-shrink; TSAPP_DOMAIN_DIR overrides the directory.
#
# SAFETY MODE. A capability fault inside a domain ends the emulator, so a boot holds at most one
# fixture predicted to FAULT on its arm, and it must come last: the runner refuses any other
# order. Each fixture's output goes straight to the serial log between __TSAPP_BEGIN_FX<n>__ and
# __TSAPP_END_FX<n>__, because a fault takes the boot down before anything could be copied back.
#
# The domain runs `/tmp/tshark -r /tmp/input.pcap -V -n` with TZ=UTC, HOME=/tmp and an empty
# WIRESHARK_CONFIG_DIR, given by /tmp/domain.argv and /tmp/domain.env (deps/domain_entry.c). The
# guest writes those files and copies each capture to /tmp/input.pcap before each run, so one
# image serves them all.
#
# The host is musl-capstone's generic libc_test_host.c: it services the domain's hostcalls, prints
# its stdout, and ends with one `LT-RESULT <image> status=<n> ...` line. A staged image prints
# `TSAPP-STAGE n` and exits with 100 + n (patch 0006). A domain that halts is `FAIL HALTED`.
#
# Boot environment:
# - a PRIVATE rootfs (host/make-rootfs.sh), fresh from buildroot's rootfs.tar, with the CMA-capable
#   module at TSAPP_MODULE (default: $TS_WORK/cma-module/fix.ko, a74a856). The shared rootfs.ext2
#   is never used.
# - cma= sized as (runs + 1) x the image's block: the module never frees a domain block.
#
# Verdicts, printed per section by host-side checks, never by the runner's exit status:
# - stages: `STAGE n REACHED` when the status is 100 + n AND the stage's line is there (M5: status
#   0 and no stage line), then `output MATCH|DIFFERS` for what the stage printed;
# - oracle: `<capture>: stdout MATCH|DIFFERS, stderr MATCH|DIFFERS`.
#   - stdout against native STOCK tshark (TSAPP_STOCK): the same command and pinned environment
#     as host/oracle.sh, the reference generated here;
#   - stderr against the NATIVE MINIMAL build (TSAPP_MINIMAL, native-min-pa: the same whitelist,
#     byte-identical generated dissectors.c, the same patches), because the minimal build writes
#     194 registration notices there (patch 0002: dissector tables the whitelist leaves out) and
#     stock writes none. TSAPP_MINIMAL=none skips this check.
#   The runtime sends the domain's fd 1 and fd 2 to the host's stdout, which also carries the
#   loader's banner. host/domain-stdout.py removes exactly the banner, domain_entry's echoes and
#   the TSAPP-HEAP and LT-RESULT lines, then requires the rest to begin with the expected stderr.
#   Any other extra byte shows as a difference: the split can turn a match into DIFFERS, never the
#   reverse.
#
# Every boot writes its share's SHA256SUMS next to its log, and its share is its own (mktemp). A
# second invocation cannot swap a running boot's images (the FFmpeg runners' race, 2026-09-24).
set -euo pipefail
APP=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$APP/deps/env.sh" > /dev/null
MODE=${1:?stages | oracle <capture>... | safety <fixture>...}; shift || true
HEAP=${TSAPP_HEAP:-level0}
case $HEAP in level0) d=domain ;; shrink) d=domain-shrink ;; *) echo "TSAPP_HEAP must be level0 or shrink" >&2; exit 2 ;; esac
OUT=${TSAPP_DOMAIN_DIR:-$TS_WORK/$d} CAPS=$TS_WORK/xsrc/test/captures RUNS=$TS_WORK/runs
EXPECT=$APP/host/safety-expect.txt
STOCK=${TSAPP_STOCK:-$TS_WORK/native-stock/run/tshark}
MINIMAL=${TSAPP_MINIMAL:-$TS_WORK/native-min-pa/run/tshark}
ROOTFS=${TSAPP_ROOTFS:-$TS_WORK/br-clean}
MODULE=${TSAPP_MODULE:-$TS_WORK/cma-module/fix.ko}
mkdir -p "$RUNS"
[ -f "$OUT/tshark_m5.dom" ] || { echo "no images; run host/build-domain.sh" >&2; exit 2; }
if [ ! -f "$ROOTFS/build/images/rootfs.ext2" ]; then
  [ -f "$MODULE" ] || { echo "no module at $MODULE (build it as results/2026-09-24-qemu-cma-domain-block/build.sh does)" >&2; exit 2; }
  bash "$APP/host/make-rootfs.sh" "$ROOTFS" "$MODULE"
fi

# The fabricated gp stays NONLIN, the type the entry glue gives it. Without this QEMU re-fabricates
# a LINEAR gp at every call, a `movc` of a live code capability moves it and nulls the source, and
# the next cjalr through it faults deep in the run (ports/sqlite/run-sqlite-speedtest1.sh:52-64;
# CPython and Postgres set it too). Printed into every log.
export CAPSTONE_GP_NONLIN=${CAPSTONE_GP_NONLIN:-1}

SHARE=$(mktemp -d "$TS_WORK/share.XXXXXX"); trap 'rm -rf "$SHARE"' EXIT
LT_DIR="$CAPSTONE_REPO_ROOT/capstone/ports/musl-capstone/libc-test"
"${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}" -O2 \
  -I"$LT_DIR" -I"$CAPSTONE_REPO_ROOT/capstone/ports/musl-capstone/runtime" \
  -I"$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib" \
  -I"$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/hostcall-stdout-probe" \
  -I"$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu" \
  -o "$SHARE/lt.user" "$LT_DIR/libc_test_host.c" "$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib/libcapstone.c"
printf '%s\n' /tmp/tshark -r /tmp/input.pcap -V -n > "$SHARE/domain.argv"
printf '%s\n' TZ=UTC HOME=/tmp WIRESHARK_CONFIG_DIR=/tmp/wsconf > "$SHARE/domain.env"

stage_cap() {  # name -> copies the capture (or its flipped twin) into the share as <name>.pcap
  local n=$1 base=${1%.flip}
  [ -f "$CAPS/$base.pcap" ] || { echo "no capture $CAPS/$base.pcap" >&2; exit 2; }
  if [ "$n" = "$base" ]; then cp "$CAPS/$base.pcap" "$SHARE/$n.pcap"
  else python3 -c 'import sys; b=bytearray(open(sys.argv[1],"rb").read()); b[len(b)-8]^=1; open(sys.argv[2],"wb").write(b)' \
         "$CAPS/$base.pcap" "$SHARE/$n.pcap"; fi
}
# Time budgets. Every tshark run measured so far took 15-30 s, and setup (the copies from the share)
# usually about 30 s. So: the host program's alarm at 120 s per run, and the whole guest command at
# 150 s per section plus 600 s for setup (see the setup watchdog). A guest that stalls holds the
# SHARED QEMU lock until this expires: with 960 s per section a 9p-copy stall held it for 85 minutes
# against other lanes' queued work (2026-09-25). The alarm cannot end a guest stall, only a run that
# keeps making hostcalls.
RUN_ALARM=${TSAPP_RUN_ALARM:-120}
G="$SHARE/guest.sh"
# The setup watchdog. Boots on 2026-09-24/25 stalled in the copies below, before any domain started,
# and each held the shared QEMU lock until the whole guest budget ran out. What it caught once it
# could look: a cp waiting in p9_virtio_zc_request, a 9p zero-copy read, and the copy still MOVING,
# only slowly -- one finished during the watchdog's own report, after more than five minutes. So:
# if setup has not finished within TSAPP_SETUP_BUDGET seconds, the guest prints where the copy
# stands (what reached /tmp, each cp's kernel stack and wait channel, memory and CMA), then checks
# every 30 s and powers itself off only if nothing moved in that time. A slow copy completes and its
# boot counts; a stopped one ends the boot within a minute of being seen. A guest that wedges
# outright shows neither line, and only the outer budget below ends it.
SETUP_BUDGET=${TSAPP_SETUP_BUDGET:-300}
{ echo "( sleep $SETUP_BUDGET; [ -e /tmp/.ts_setup_done ] && exit 0; echo __TS_SETUP_SLOW__; ls -l /tmp;" \
       "for p in \$(pidof cp); do echo cp \$p wchan=\$(cat /proc/\$p/wchan); cat /proc/\$p/stack; done;" \
       "grep -E 'MemFree|MemAvailable|Shmem:|CmaTotal|CmaFree' /proc/meminfo; echo __TS_SETUP_SLOW_END__;" \
       "while :; do a=\$(du -sk /tmp | cut -f1); sleep 30; [ -e /tmp/.ts_setup_done ] && exit 0;" \
       "b=\$(du -sk /tmp | cut -f1); echo __TS_SETUP_PROGRESS__ \$a \$b;" \
       "[ \"\$a\" = \"\$b\" ] && { echo __TS_SETUP_STALLED__; poweroff -f; }; done ) &"
  echo 'echo MODULE-MD5 $(md5sum /capstone.ko); cat /proc/cmdline'
  echo 'cp /mnt/host/lt.user /tmp/lt.user && chmod 0755 /tmp/lt.user; cp /mnt/host/domain.argv /mnt/host/domain.env /tmp/; mkdir -p /tmp/wsconf'
  echo 'cp /mnt/host/*.dom /tmp/; dmesg -c > /dev/null; touch /tmp/.ts_setup_done; echo __TS_SETUP_DONE__'; } > "$G"
SECTIONS=()
case $MODE in
  stages)
    SC=${TSAPP_STAGES_CAPTURE:-dhcp}
    stage_cap "$SC"
    for n in 1 2 3 4 5; do
      cp "$OUT/tshark_m$n.dom" "$SHARE/"
      { echo "echo __TS_BEGIN_M${n}__; rm -f /tmp/input.pcap"
        echo "cp '/mnt/host/$SC.pcap' /tmp/input.pcap && /tmp/lt.user /tmp/tshark_m$n.dom $RUN_ALARM > /tmp/out-m$n.txt 2> /tmp/err-m$n.txt; echo __TS_RC_M${n}__ \$?"
        echo "tail -n 1 /tmp/out-m$n.txt; tail -n 3 /tmp/err-m$n.txt; cp /tmp/out-m$n.txt /tmp/err-m$n.txt /mnt/host/"
        echo "dmesg -c > /tmp/dmesg-m$n.txt; grep -v 'remote fence' /tmp/dmesg-m$n.txt | sed 's/^/DMESG M$n: /'; echo __TS_END_M${n}__"; } >> "$G"
      SECTIONS+=("M$n")
    done ;;
  oracle)
    [ $# -gt 0 ] || { echo "oracle needs captures" >&2; exit 2; }
    cp "$OUT/tshark_m5.dom" "$SHARE/"
    for c in "$@"; do
      stage_cap "$c"
      { echo "echo __TS_BEGIN_$c""__; rm -f /tmp/input.pcap"
        echo "cp '/mnt/host/$c.pcap' /tmp/input.pcap && /tmp/lt.user /tmp/tshark_m5.dom $RUN_ALARM > '/tmp/out-$c.txt' 2> '/tmp/err-$c.txt'; echo __TS_RC_$c""__ \$?"
        echo "tail -n 1 /tmp/out-$c.txt; cp /tmp/out-$c.txt /tmp/err-$c.txt /mnt/host/"
        echo "dmesg -c > /tmp/dmesg-$c.txt; grep -v 'remote fence' /tmp/dmesg-$c.txt | sed 's/^/DMESG $c: /'; echo __TS_END_$c""__"; } >> "$G"
      SECTIONS+=("$c")
    done ;;
  safety)
    [ $# -gt 0 ] || { echo "safety needs fixtures" >&2; exit 2; }
    i=0
    for fx in "$@"; do
      i=$((i + 1))
      [ -f "$OUT/tsapp_fx$fx.dom" ] || { echo "no image $OUT/tsapp_fx$fx.dom" >&2; exit 2; }
      grep -qE "^$HEAP +$fx +" "$EXPECT" || { echo "fixture $fx has no prediction for $HEAP" >&2; exit 2; }
      if [ "$i" -lt $# ] && grep -qE "^$HEAP +$fx +FAULT" "$EXPECT"; then
        echo "fixture $fx is predicted to FAULT on $HEAP and is not last; the rest would never run" >&2; exit 2
      fi
      cp "$OUT/tsapp_fx$fx.dom" "$SHARE/"
      echo "echo __TSAPP_BEGIN_FX${fx}__; /tmp/lt.user /tmp/tsapp_fx$fx.dom $RUN_ALARM; echo __TSAPP_END_FX${fx}__" >> "$G"
      SECTIONS+=("fx$fx")
    done ;;
  *) echo "mode must be stages, oracle or safety" >&2; exit 2 ;;
esac
echo 'echo __TS_ALL_DONE__' >> "$G"

# cma: (runs + 1) blocks of the largest image this boot runs, each twice the size build-domain.sh
# prints, because a74a856 doubles a block whose declared data does not survive the monitor's split
# (M-infra item 2). The module never frees a domain block (capstone.c has no dma_free_pages for it),
# so every run takes a new one.
BLOCK_MB=$(python3 - "${TSAPP_STACK_BYTES:-$((1 << 20))}" "$SHARE"/*.dom <<'PY2'
import subprocess, sys
mb = 4
for img in sys.argv[2:]:
    out = subprocess.run(['readelf', '-lW', img], capture_output=True, text=True).stdout
    memsz = max(int(l.split()[5], 16) for l in out.splitlines() if l.split()[:1] == ['LOAD'])
    pages = (memsz + 8192 + int(sys.argv[1]) - 1) // 4096 + 1
    mb = max(mb, (1 << (pages - 1).bit_length()) * 4096 >> 20)
print(mb)
PY2
)
CMA=$(( (${#SECTIONS[@]} + 1) * BLOCK_MB * 2 ))
# The largest cma a boot has been seen to take (the six-capture oracle, 2026-09-24). Beyond it the
# reservation is untested, and it may not fit below the 4 GiB the kernel places CMA under; a boot
# that fails there would read as a stall. Split the runs instead.
[ "$CMA" -le 1792 ] || { echo "cma=${CMA}M for ${#SECTIONS[@]} runs is more than the 1792M measured to boot; use fewer runs per boot" >&2; exit 2; }
tag=$MODE; if [ "$MODE" = safety ] || [ "$HEAP" != level0 ]; then tag=$MODE-$HEAP; fi
LOG=$(mktemp "$RUNS/qemu-$tag-$(date +%Y%m%d-%H%M%S)-XXXX.log")
( cd "$SHARE" && sha256sum lt.user domain.argv domain.env ./*.dom $(find . -maxdepth 1 -name '*.pcap' | sort) ) > "$LOG.sha256"
echo "run-qemu: ${#SECTIONS[@]} runs, block ${BLOCK_MB} MiB, cma=${CMA}M, CAPSTONE_GP_NONLIN=$CAPSTONE_GP_NONLIN, log $LOG" | tee "$LOG.head"
set +e
# A safety fixture runs in about 10 s (2026-09-25), so its sections get 60 s. Setup gets 600 s: a
# slow 9p copy has taken more than five minutes and then completed, and the watchdog above ends a
# copy that stops. A guest that wedges outright, which nothing inside it can end, then costs at
# most 16 minutes for six fixtures.
PER_SECTION=150; [ "$MODE" = safety ] && PER_SECTION=60
CAPSTONE_GUEST_COMMAND_TIMEOUT=${CAPSTONE_GUEST_COMMAND_TIMEOUT:-$(( ${#SECTIONS[@]} * PER_SECTION + 600 ))} \
capstone_with_qemu_lock python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE" --buildroot-dir "$ROOTFS" --qemu-binary "$CAPSTONE_QEMU_BINARY" \
  --kernel-arg "cma=${CMA}M" --guest-command "sh /mnt/host/guest.sh" \
  --success-marker __TS_ALL_DONE__ --log-file "$LOG" > "$LOG.runner" 2>&1
echo "run-domain-smoke exit status $? (not the verdict)"
set -e
mkdir "$LOG.out"; cp "$SHARE"/out-*.txt "$SHARE"/err-*.txt "$LOG.out/" 2>/dev/null || true
# The images this boot ran, kept beside its log: build-domain.sh rewrites $TS_WORK/domain on every
# run, and the share goes when this script exits, so otherwise a counted result would be left with
# only the hashes of what it ran (2026-09-24).
mkdir "$LOG.images"; cp "$SHARE"/*.dom "$SHARE"/lt.user "$LOG.images/"

if [ "$MODE" = safety ]; then
  # The verdict is per fixture, from the serial log; its exit status is this script's. Redirected,
  # not piped, so nothing stands between the verdict and that status.
  set +e; python3 "$APP/host/safety-verdict.py" "$LOG" "$EXPECT" "$HEAP" "$@" > "$LOG.verdict"; rc=$?; set -e
  cat "$LOG.verdict"; exit "$rc"
fi

# Verdicts.
# The references, made on the host from the same capture bytes the guest was given. An empty one is
# an ERROR, never a comparison: an empty expected stderr would match anything (audit, 2026-09-24).
refs() {  # section capture -> $LOG.out/ref-<section>.txt (stock stdout), referr-<section>.txt (minimal stderr)
  local cfg; cfg=$(mktemp -d)
  HOME=$cfg WIRESHARK_CONFIG_DIR=$cfg TZ=UTC "$STOCK" -r "$2" -V -n > "$LOG.out/ref-$1.txt" 2> /dev/null || true
  if [ "$MINIMAL" != none ]; then
    HOME=$cfg WIRESHARK_CONFIG_DIR=$cfg TZ=UTC "$MINIMAL" -r "$2" -V -n > /dev/null 2> "$LOG.out/referr-$1.txt" || true
  fi
  rm -rf "$cfg"
  [ -s "$LOG.out/ref-$1.txt" ] || { echo "stock wrote no stdout"; return 1; }
  [ "$MINIMAL" = none ] || [ -s "$LOG.out/referr-$1.txt" ] || { echo "the minimal build wrote no stderr"; return 1; }
}
export DOMAIN_ARGV="$SHARE/domain.argv" DOMAIN_ENV="$SHARE/domain.env"
for s in "${SECTIONS[@]}"; do
  if [ "$MODE" = stages ]; then o="$LOG.out/out-m${s#M}.txt"; else o="$LOG.out/out-$s.txt"; fi
  if [ ! -f "$o" ]; then
    # A domain that halts on a capability fault takes QEMU down with it (the boot ends there), so
    # its section never copies anything back. Say which it was, from the serial log.
    m=$s; [ "$MODE" = stages ] && m="M${s#M}"
    python3 - "$LOG" "$m" <<'PYH'
import re, sys
log = open(sys.argv[1], 'rb').read().decode('latin-1').replace('\r', '')
sec = sys.argv[2]
b = re.search(r'^__TS_BEGIN_%s__$' % re.escape(sec), log, re.M)
if not b:
    print(f'{sec}: NOT RUN (the boot ended before this section)'); sys.exit()
nxt = re.search(r'^__TS_(BEGIN|END)_', log[b.end():], re.M)
part = log[b.end(): b.end() + nxt.start()] if nxt else log[b.end():]
h = re.search(r'domain halted by capability fault: cause = (\d+), pc = (0x[0-9a-f]+)', part)
if h:
    print(f'{sec}: HALTED (capability fault, cause {h.group(1)}, pc {h.group(2)}; QEMU ended the boot here)')
else:
    print(f'{sec}: ERROR no output (the section started; nothing was copied back and no halt was logged)')
PYH
    continue
  fi
  lt=$(grep -a '^LT-RESULT ' "$o" | tail -1 || true)
  [ -n "$lt" ] || { echo "$s: ERROR no LT-RESULT line (the host never finished)"; continue; }
  st=$(printf '%s' "$lt" | sed -n 's/.* status=\(-\?[0-9]*\) .*/\1/p')
  dom="$LOG.out/dom-$s.txt"
  if [ "$MODE" = stages ]; then
    # REACHED needs both the stage's own exit code (100 + n, outside tshark's 1..18 and 256) and
    # its TSAPP-STAGE line; M5 needs status 0 and no stage line. A status alone is not enough: a
    # failed cf_open exits with 3 (WS_EXIT_INVALID_FILE) before stage 3 (audit, 2026-09-24).
    n=${s#M}
    if [ "$n" = 5 ]; then
      if [ "$st" = 0 ] && ! grep -aq '^TSAPP-STAGE ' "$o"; then echo "$s: STAGE 5 REACHED ($lt)"; else echo "$s: NOT REACHED ($lt)"; continue; fi
    else
      if [ "$st" = $((100 + n)) ] && grep -axq "TSAPP-STAGE $n" "$o"; then echo "$s: STAGE $n REACHED ($lt)"; else echo "$s: NOT REACHED ($lt)"; continue; fi
    fi
    # What the stage printed, against what it must have printed ($SC): M1 its stage line only;
    # M2 and M3 the minimal build's stderr and the stage line; M4 the same stderr, then stock's
    # first frame exactly, then the stage line; M5 the stderr, then stock's stdout exactly.
    [ "$MINIMAL" != none ] || { echo "$s: output not checked (TSAPP_MINIMAL=none)"; continue; }
    [ -f "$LOG.out/ref-$SC.txt" ] || why=$(refs "$SC" "$SHARE/$SC.pcap") || { echo "$s: output ERROR ($why)"; continue; }
    errarg=("$LOG.out/referr-$SC.txt"); [ "$n" = 1 ] && errarg=()
    e=$(python3 "$APP/host/domain-stdout.py" "$o" "$dom" "${errarg[@]}") \
      || { echo "$s: output ERROR $(tail -1 "$dom.err" 2>/dev/null)"; continue; }
    python3 - "$s" "$dom" "$LOG.out/ref-$SC.txt" "$e" <<'PY3'
import sys
s, dom, ref, e = sys.argv[1], open(sys.argv[2], 'rb').read(), open(sys.argv[3], 'rb').read(), sys.argv[4]
n = int(s[1:])
tail = b'TSAPP-STAGE %d\n' % n if n < 5 else b''
if not dom.endswith(tail):
    print(f'{s}: output DIFFERS ({e}; no TSAPP-STAGE {n} line at the end)'); sys.exit()
body = dom[:len(dom) - len(tail)]
if n < 4:
    ok, what = body == b'', 'nothing else'
elif n == 4:
    # Frame 1 exactly: -V ends every frame with one blank line and has none inside one (checked on
    # dhcp, dns_port, http and arp: as many "\n\n" as frames).
    first = ref[:ref.index(b'\n\n') + 2] if b'\n\n' in ref else None
    ok, what = first is not None and body == first, f'stock stdout\'s first frame exactly ({len(first or b"")} bytes)'
else:
    ok, what = body == ref, 'stock stdout exactly'
print(f'{s}: output {"MATCH" if ok else "DIFFERS"} ({e}; expected {what})')
PY3
  else
    why=$(refs "$s" "$SHARE/$s.pcap") || { echo "$s: ERROR ($why)"; continue; }
    errarg=(); [ "$MINIMAL" = none ] || errarg=("$LOG.out/referr-$s.txt")
    e=$(python3 "$APP/host/domain-stdout.py" "$o" "$dom" "${errarg[@]}") \
      || { echo "$s: ERROR $(tail -1 "$dom.err" 2>/dev/null)"; continue; }
    [ "$MINIMAL" != none ] || e="stderr not checked, $e"
    # The run's own outcome first: output that compares equal from a run that did not exit 0 is
    # not a pass (audit, 2026-09-24).
    head=""; [ "$st" = 0 ] || head="FAILED (exit status $st), "
    if cmp -s "$LOG.out/ref-$s.txt" "$dom"; then echo "$s: ${head}stdout MATCH, $e ($lt)"
    else echo "$s: ${head}stdout DIFFERS ($(diff "$LOG.out/ref-$s.txt" "$dom" | grep -c '^[<>]' || true) diff lines), $e ($lt)"; fi
  fi
done
# The flip control (host/oracle.sh's): a flipped capture proves something only if it changes
# stock's output. Reported for every <c>.flip whose <c> ran in the same boot.
if [ "$MODE" = oracle ]; then
  for s in "${SECTIONS[@]}"; do
    case $s in *.flip) ;; *) continue ;; esac
    a="$LOG.out/ref-${s%.flip}.txt" b="$LOG.out/ref-$s.txt"
    [ -s "$a" ] && [ -s "$b" ] || { echo "$s: flip control ERROR (no stock output for both)"; continue; }
    n=$(diff "$a" "$b" | grep -c '^[<>]' || true)
    if [ "$n" -gt 0 ]; then echo "$s: flip control FIRES (stock changed $n lines)"; else echo "$s: flip control DID NOT FIRE"; fi
  done
fi
