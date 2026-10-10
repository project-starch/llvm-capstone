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
#     `supervise` can resolve mch_write_probe from the ELF and the fault can be
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
PORT=${CHERI_PORT:-10501}
OUT=${1:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/mc-plain-heap-cheribsd}

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
    -o "$OUT/bin/mch-$n" || { echo "CONTROL-FAILED purecap build case $n" >&2; exit 75; }
done
# One supervisor per labelled probe, chosen per case below from the probe its case.c calls. Until
# 2026-10-10 a single supervisor was built for mch_write_probe, so the READ defects 06-08 printed
# "SUPERVISE expect mch_write_probe" and their records said the fault lay in that probe's extent; it
# lies in mch_read_probe (attributed with llvm-nm, results/2026-10-08-cheribsd/attribution-2026-10-10.txt).
for sym in mch_write_probe mch_read_probe; do
  "$SDK/bin/clang" $CFLAGS -DPROBE_SYMBOL="\"$sym\"" \
    "$CAP/bug-corpora/cpython/pymalloc-repros/observe/supervise.c" -lutil \
    -o "$OUT/bin/supervise-$sym" || { echo "CONTROL-FAILED supervise build ($sym)" >&2; exit 75; }
done
"$SDK/bin/clang" $CFLAGS "$CAP/ports/common/host/cheribsd/abi-probe.c" \
  -o "$OUT/bin/cheribsd-abi-probe" || { echo "CONTROL-FAILED abi-probe build" >&2; exit 75; }

# PREDICTION, written before the run: NOT CAUGHT, and this is a REFUTED earlier
# prediction rather than a guess. The crossing is one byte past calloc(1, 9), and
# CheriBSD's malloc bounds to the allocator's USABLE size: measured in-guest,
# request 9 yields a capability of length 16, so offset 9 is inside the bounds
# and no fault is possible. Contrast the wireshark sibling, whose 65471-byte
# crossing past an exactly-size-classed 8192 does leave the allocation.
python3 - "$OUT/bin" "$OUT/cases.json" "$ROOT" <<'CASESPY'
import json, pathlib, sys
BIN, OUT, CORPUS = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
SIGPROT = 34
# The buggy arm's expectation is PER CASE, because the cases differ in whether the
# crossing leaves the allocator's USABLE size. malloc bounds a capability to that
# size and not to the request, so a crossing shorter than the gap to the next size
# class stays inside the same allocation and NO capability check can see it.
# Encoding one expectation for all of them would make a true reading look like a
# failure, which is how a measured absorption gets mistaken for a bug.
#
# The usable sizes are MEASURED in this guest and recorded in case 00's case.json:
# calloc(1,1) and calloc(1,9) both return length 16, calloc(1,17) returns 32,
# calloc(1,8192) returns exactly 8192. tools/size-class-audit.py reproduces all four.
CASES = {
    0: (9,  '1 byte past calloc(1, 9), whose capability is 16 long -- ABSORBED', 'COMPLETE'),
    1: (64, '1 byte past malloc(64), and 64 is exactly a size class',            'FAULT'),
    2: (64, '1 byte past a 64-byte realloc of the suffix freelist',              'FAULT'),
    3: (64, '1 byte past a 64-byte object-cache freelist array',                 'FAULT'),
    4: (64, '1 byte past a 64-byte stats buffer',                                'FAULT'),
    5: (16, '1 byte past a 16-byte connection write buffer',                     'FAULT'),
    6: (16, '1 byte past a 16-byte proxy key buffer',                            'FAULT'),
    7: (16, '1 byte past a 16-byte binary-protocol key',                         'FAULT'),
    8: (16, '1 element (8 bytes) past a 16-byte slab page list',                 'FAULT'),
}
# A case directory with no row here is one the run would be silent about. The
# FFmpeg sibling had exactly that: a table of 4 while the corpus held 25.
present = {int(d.name[:2]) for d in CORPUS.glob('[0-9][0-9]_*') if d.is_dir()}
missing, extra = sorted(present - set(CASES)), sorted(set(CASES) - present)
if missing or extra:
    msg = []
    if missing: msg.append('cases %s exist in the corpus but have no row in CASES' % missing)
    if extra:   msg.append('CASES names %s, which is not a case directory' % extra)
    sys.exit('CONTROL-FAILED cases.json: ' + '; '.join(msg))

def probe_of(n):
    src = next(CORPUS.glob('%02d_*' % n)) / 'case.c'
    text = src.read_text()
    reads, writes = 'read_probe(' in text, 'write_probe(' in text
    if reads == writes:
        sys.exit('CONTROL-FAILED cases.json: case %d calls %s labelled probe' % (n, 'both' if reads else 'no'))
    return 'mch_read_probe' if reads else 'mch_write_probe'

cases = []
for p in sorted(BIN.glob('mch-[0-9][0-9]')):
    n = int(p.name.split('-')[1])
    request, what, expect = CASES[n]
    supervise = BIN / ('supervise-' + probe_of(n))
    cases.append(dict(name='%s-fixed' % p.name, program=str(p), args=['fixed', str(n)],
                      timeout=300, expect_regex=r'VERDICT FIXED .*', exit=0))
    if expect == 'FAULT':
        cases.append(dict(name='%s-buggy' % p.name, program=str(supervise),
                          args=['./target','buggy',str(n)], inputs={'target': str(p)},
                          timeout=300, expect='SUPERVISE exit signalled=%d' % SIGPROT,
                          exit=128+SIGPROT))
    else:
        cases.append(dict(name='%s-buggy' % p.name, program=str(supervise),
                          args=['./target','buggy',str(n)], inputs={'target': str(p)},
                          timeout=300, expect='SUPERVISE exit status=0', exit=0))
OUT.write_text(json.dumps(cases, indent=2) + '\n')
print('  wrote %s: %d cases, covering all %d case directories' % (OUT, len(cases), len(present)))
for n, (request, what, expect) in sorted(CASES.items()):
    print('    case %2d: %s -- %s' % (n, 'CATCH   ' if expect == 'FAULT' else 'COMPLETE', what))
CASESPY
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
