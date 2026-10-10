#!/usr/bin/env bash
# This corpus on stock CheriBSD purecap, with libc revocation ON.
#
#   CHERI_SDK=~/cheri/output/sdk CHERI_SYSROOT=~/cheri/rootfs-purecap \
#   CHERI_IMAGE=~/cheri/output/cheribsd-riscv64-purecap.img \
#     bash runners/run-cheribsd.sh [outdir]
#
# Modelled on ../../ffmpeg/plain-heap-repros/runners/run-cheribsd.sh. The three
# things that are load-bearing, each learned by getting it wrong:
#
#  1. -O0. clang folds a constant out-of-bounds access that gcc keeps.
#  2. The probe is DEFINED ONCE in shared/driver.c, not static in the header, so
#     `supervise` can resolve wsh_read_probe from the ELF and the fault can be
#     required to land inside it. PROBE_SYMBOL is compile-time.
#  3. The platform's OWN abi probe for --abi-probe. Passing anything else makes
#     cheribsd-abi and cheribsd-bounds FAIL, and a suite whose platform controls
#     did not fire is not a reading however plausible its case rows look.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$HERE/..
REPO=$(cd "$ROOT/../../../.." && pwd)
CAP=$REPO/capstone

SDK=${CHERI_SDK:-$HOME/cheri/output/sdk}
SYSROOT=${CHERI_SYSROOT:-$HOME/cheri/rootfs-purecap}
IMAGE=${CHERI_IMAGE:-$HOME/cheri/output/cheribsd-riscv64-purecap.img}
PORT=${CHERI_PORT:-10499}
OUT=${1:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/ws-plain-heap-cheribsd}

for p in "$SDK/bin/clang" "$SYSROOT" "$IMAGE"; do
  [ -e "$p" ] || { echo "CONTROL-FAILED missing $p" >&2; exit 75; }
done
mkdir -p "$OUT/bin"

CFLAGS="--target=riscv64-unknown-freebsd13 -march=rv64imafdcxcheri -mabi=l64pc128d"
CFLAGS="$CFLAGS -mno-relax -B$SDK/bin --sysroot=$SYSROOT -std=gnu11 -O0 -fuse-ld=lld"
# Optional, for another arm on the same cases: CHERI_EXTRA_CFLAGS (e.g. -Xclang
# -cheri-bounds=subobject-safe for cheribsd-subobject) and CHERI_REVOCATION=off (PoisonCap mode 0).
# Both default to this runner's own arm, so an unset environment builds and boots exactly as before.
CFLAGS="$CFLAGS ${CHERI_EXTRA_CFLAGS:-}"
REVOCATION=${CHERI_REVOCATION:-on}
case "$REVOCATION" in on|off) ;; *) echo "CONTROL-FAILED CHERI_REVOCATION=$REVOCATION" >&2; exit 75;; esac

for dir in "$ROOT"/[0-9][0-9]_*/; do
  n=$(basename "$dir" | cut -c1-2)
  "$SDK/bin/clang" $CFLAGS -I"$ROOT/shared" "$dir/case.c" "$ROOT/shared/driver.c" \
    -o "$OUT/bin/wsh-$n" || { echo "CONTROL-FAILED purecap build case $n" >&2; exit 75; }
done
"$SDK/bin/clang" $CFLAGS -DPROBE_SYMBOL='"wsh_read_probe"' \
  "$CAP/bug-corpora/cpython/pymalloc-repros/observe/supervise.c" -lutil \
  -o "$OUT/bin/supervise" || { echo "CONTROL-FAILED supervise build" >&2; exit 75; }
"$SDK/bin/clang" $CFLAGS "$CAP/ports/common/host/cheribsd/abi-probe.c" \
  -o "$OUT/bin/cheribsd-abi-probe" || { echo "CONTROL-FAILED abi-probe build" >&2; exit 75; }

# PREDICTIONS, written before the run. Most rows CATCH: their crossing leaves the
# USABLE allocation, which is what malloc bounds the capability to. ABSORBED names
# the exception and is NOT a prediction of safety -- a crossing shorter than the gap
# to the next size class stays inside the same allocation, so no capability check
# can see it. Declaring CATCH for such a row makes a correct reading look like a
# failure, which is how a measured absorption gets mistaken for a bug; that happened
# on the 2026-10-08 run, where wsh-03 read FAIL for exactly this reason.
#
# ABSORBED is taken from tools/size-class-audit.py, which reproduces four in-guest
# usable sizes recorded in memcached/plain-heap-repros/00's case.json.
python3 - "$OUT/bin" "$OUT/cases.json" "$ROOT" <<'PY'
import json, pathlib, sys
BIN, OUT, CORPUS = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
SIGPROT = 34
# case -> request bytes, crossing, expectation. Every case directory must appear;
# the gate below refuses the run otherwise, because a case with no row here is one
# the run would be silent about.
CASES = {
    0:  (8192, '65471 bytes past a g_malloc(8192)',                       'FAULT'),
    1:  (16,   '16 bytes past a 16-byte array of pointers',               'FAULT'),
    2:  (16,   '3 bytes past a 16-byte string buffer',                    'FAULT'),
    3:  (1,    '1 byte past a 1-byte calloc; usable 16, so 15 bytes of '
               'slack ABSORB it -- the trigger is an EMPTY string, so the '
               'request cannot be sized onto a class boundary',           'COMPLETE'),
    4:  (16,   '1 byte past a 16-byte APP_TEXT payload',                  'FAULT'),
    5:  (16,   '1 byte past a 16-byte packet buffer',                     'FAULT'),
    6:  (16,   '1 byte past a 16-byte input buffer',                      'FAULT'),
    7:  (16,   '1 byte past a 16-byte decode buffer',                     'FAULT'),
    8:  (16,   '8 bytes past a 16-byte option string',                    'FAULT'),
    9:  (16,   'past a 16-byte buffer; the real index wraps to 4294967295 '
               'so the probe touches the first byte outside instead',     'FAULT'),
    10: (16,   '1 byte past a 16-byte output buffer',                     'FAULT'),
    11: (16,   '1 byte past a 16-byte output buffer',                     'FAULT'),
}
present = {int(d.name[:2]) for d in CORPUS.glob('[0-9][0-9]_*') if d.is_dir()}
missing, extra = sorted(present - set(CASES)), sorted(set(CASES) - present)
if missing or extra:
    msg = []
    if missing: msg.append(f'cases {missing} exist in the corpus but have no row in CASES')
    if extra:   msg.append(f'CASES names {extra}, which is not a case directory')
    sys.exit('CONTROL-FAILED cases.json: ' + '; '.join(msg))

cases = []
for p in sorted(BIN.glob('wsh-[0-9][0-9]')):
    n = int(p.name.split('-')[1])
    request, what, expect = CASES[n]
    cases.append(dict(name=f'{p.name}-fixed', program=str(p), args=['fixed', str(n)],
                      timeout=300, expect_regex=r'VERDICT FIXED .*', exit=0))
    if expect == 'FAULT':
        cases.append(dict(name=f'{p.name}-buggy', program=str(BIN/'supervise'),
                          args=['./target','buggy',str(n)], inputs={'target': str(p)},
                          timeout=300, expect=f'SUPERVISE exit signalled={SIGPROT}',
                          exit=128+SIGPROT))
    else:
        cases.append(dict(name=f'{p.name}-buggy', program=str(BIN/'supervise'),
                          args=['./target','buggy',str(n)], inputs={'target': str(p)},
                          timeout=300, expect='SUPERVISE exit status=0', exit=0))
OUT.write_text(json.dumps(cases, indent=2) + '\n')
print(f'  wrote {OUT}: {len(cases)} cases, covering all {len(present)} case directories')
for n, (request, what, expect) in sorted(CASES.items()):
    print(f"    case {n:>2}: {'CATCH   ' if expect=='FAULT' else 'COMPLETE'} -- {what}")
PY
[ -s "$OUT/cases.json" ] || { echo "CONTROL-FAILED cases.json" >&2; exit 75; }
. "$CAP/bug-corpora/tools/cheribsd-subobj-control.sh"
subobj_control_add "$OUT" "$SDK" ${CFLAGS/${CHERI_EXTRA_CFLAGS:-@@none@@}/} || exit 75


# A marker whose mtime bounds THIS invocation. The control check below requires each
# control's record to be newer than it, because $OUT/run survives a previous run and a
# stale record would otherwise satisfy the check for a boot that never happened.
STAMP="$OUT/.run-started"
: > "$STAMP"

python3 "$CAP/ports/common/host/cheribsd/run.py" "$OUT/run" \
  --sdk "$SDK" --rootfs "$SYSROOT" --image "$IMAGE" --port "$PORT" \
  --abi-probe "$OUT/bin/cheribsd-abi-probe" \
  --runtime-revocation "$REVOCATION" --cases "$OUT/cases.json" --continue-on-failure
rc=$?

# The suite's status cannot distinguish "an oracle did not hold" -- which is DATA
# -- from "the boot produced nothing". The platform controls decide that.
for c in cheribsd-abi cheribsd-bounds; do
  [ -s "$OUT/run/$c/stdout.txt" ] \
    || { echo "CONTROL-FAILED $c produced no output: not a reading" >&2; exit 75; }
  # ... and it must come from THIS boot, not from a directory a previous run left behind.
  [ "$OUT/run/$c/stdout.txt" -nt "$STAMP" ] || {
    echo "CONTROL-FAILED $c's record predates this run: it is evidence about an earlier boot," >&2
    echo "               so this suite is not a reading. Use a fresh output directory." >&2
    exit 75; }
done
WANT_REV=1; [ "$REVOCATION" = off ] && WANT_REV=0
grep -q "runtime_revocation=$WANT_REV" "$OUT/run/cheribsd-abi/stdout.txt" \
  || { echo "CONTROL-FAILED cheribsd-abi did not report runtime_revocation=$WANT_REV" >&2; exit 75; }
grep -q "CHERI_BOUNDARY_READY" "$OUT/run/cheribsd-bounds/stdout.txt" \
  || { echo "CONTROL-FAILED cheribsd-bounds did not report ready" >&2; exit 75; }
subobj_control_check "$OUT" || exit 75

echo "run-cheribsd: controls fired; suite exit $rc (non-zero = an oracle did not hold, which is data)"
exit "$rc"
