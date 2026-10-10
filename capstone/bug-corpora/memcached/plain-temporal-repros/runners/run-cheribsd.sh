#!/usr/bin/env bash
# This corpus on stock CheriBSD purecap, with libc revocation ON.
#
#   CHERI_SDK=~/cheri/output/sdk CHERI_SYSROOT=~/cheri/rootfs-purecap \
#   CHERI_IMAGE=~/cheri/output/cheribsd-riscv64-purecap.img \
#     bash runners/run-cheribsd.sh <fresh outdir>
#
# WHAT THIS ASKS. These objects come straight from malloc, so -- unlike every nested-allocator
# corpus in this tree -- their release REACHES free(), and CheriBSD's libc holds the chunk in
# quarantine until the revoker has swept every capability to it. Each case frees, re-allocates the
# same size and follows the stale pointer, with NO sweep forced in between. Three outcomes are
# distinguishable, and the case prints which:
#
#   SIGPROT                   the stale capability had been revoked: CAUGHT
#   VERDICT DEFECT-REPRODUCED the chunk was reissued and the stale pointer read the NEW object
#   VERDICT NOT-REISSUED      no fault and no aliasing: quarantine withheld the chunk
#
# PREDICTION, written before the first run: NOT-REISSUED for every buggy arm. Nothing triggers a
# sweep between the free and the access, so the stale capability keeps its tag; but the chunk sits
# in quarantine, so the fresh allocation is a different one and nothing aliases.
#
# THE REVOCATION CONTROL IS REQUIRED, not optional. Without it a completion is indistinguishable
# from a revoker that was never doing anything. It mallocs, frees, FORCES a sweep, re-allocates and
# reads the old pointer; it must report tag_after_sweep=0 and then fault. It is borrowed unchanged
# from ports/memcached/allocators/security-tests/cheribsd/revocation-control.c, the same control the
# FFmpeg pool, memcached and wmem readings used, so the columns are comparable.
#
# An infrastructure failure exits 75 and prints no verdict.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$HERE/..
REPO=$(cd "$ROOT/../../../.." && pwd)
CAP=$REPO/capstone

SDK=${CHERI_SDK:-$HOME/cheri/output/sdk}
SYSROOT=${CHERI_SYSROOT:-$HOME/cheri/rootfs-purecap}
IMAGE=${CHERI_IMAGE:-$HOME/cheri/output/cheribsd-riscv64-purecap.img}
PORT=${CHERI_PORT:-10477}
OUT=${1:?usage: run-cheribsd.sh <fresh outdir>}

for p in "$SDK/bin/clang" "$SYSROOT" "$IMAGE"; do
  [ -e "$p" ] || { echo "CONTROL-FAILED missing $p" >&2; exit 75; }
done
[ -e "$OUT/run" ] && { echo "CONTROL-FAILED $OUT/run exists: use a fresh output directory" >&2; exit 75; }
mkdir -p "$OUT/bin"

# -O0: an optimiser may fold a store into a freed object, and the aliasing the case is built to
# observe would stop being observable for a reason unrelated to the defect.
CFLAGS="--target=riscv64-unknown-freebsd13 -march=rv64imafdcxcheri -mabi=l64pc128d"
CFLAGS="$CFLAGS -mno-relax -B$SDK/bin --sysroot=$SYSROOT -std=gnu11 -O0 -fuse-ld=lld"
# Optional, for another arm on the same cases: CHERI_EXTRA_CFLAGS, and CHERI_REVOCATION=off
# (a revocation-off reading). Unset, this builds and boots exactly as before.
CFLAGS="$CFLAGS ${CHERI_EXTRA_CFLAGS:-}"
REVOCATION=${CHERI_REVOCATION:-on}
case "$REVOCATION" in on|off) ;; *) echo "CONTROL-FAILED CHERI_REVOCATION=$REVOCATION" >&2; exit 75;; esac

for dir in "$ROOT"/[0-9][0-9]_*/; do
  n=$(basename "$dir" | cut -c1-2)
  "$SDK/bin/clang" $CFLAGS -I"$ROOT/shared" "$dir/case.c" "$ROOT/shared/driver.c" \
    -o "$OUT/bin/mct-$n" || { echo "CONTROL-FAILED purecap build case $n" >&2; exit 75; }
done

# One supervise per probe symbol: PROBE_SYMBOL is compile-time.
for sym in mct_read_probe mct_write_probe mc_defect_read; do
  "$SDK/bin/clang" $CFLAGS -DPROBE_SYMBOL="\"$sym\"" \
    "$CAP/bug-corpora/cpython/pymalloc-repros/observe/supervise.c" -lutil \
    -o "$OUT/bin/supervise-$sym" || { echo "CONTROL-FAILED supervise build $sym" >&2; exit 75; }
done
"$SDK/bin/clang" $CFLAGS "$CAP/ports/common/host/cheribsd/abi-probe.c" \
  -o "$OUT/bin/cheribsd-abi-probe" || { echo "CONTROL-FAILED abi-probe build" >&2; exit 75; }
"$SDK/bin/clang" $CFLAGS "$CAP/ports/memcached/allocators/security-tests/cheribsd/revocation-control.c" \
  -o "$OUT/bin/revocation-control" || { echo "CONTROL-FAILED revocation-control build" >&2; exit 75; }

python3 - "$OUT/bin" "$OUT/cases.json" "$ROOT" <<'CASESPY'
import json, pathlib, sys
BIN, OUT, CORPUS = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
SIGPROT = 34
# case -> (probe symbol of its stale access, prediction for the buggy arm)
CASES = {
    0: ('mct_read_probe', 'NOT-REISSUED'),
    1: ('mct_write_probe', 'NOT-REISSUED'),
    2: ('mct_read_probe', 'NOT-REISSUED'),
}
present = {int(d.name[:2]) for d in CORPUS.glob('[0-9][0-9]_*') if d.is_dir()}
missing, extra = sorted(present - set(CASES)), sorted(set(CASES) - present)
if missing or extra:
    msg = []
    if missing: msg.append('cases %s exist in the corpus but have no row in CASES' % missing)
    if extra:   msg.append('CASES names %s, which is not a case directory' % extra)
    sys.exit('CONTROL-FAILED cases.json: ' + '; '.join(msg))

cases = [dict(name='revocation-control', program=str(BIN / 'supervise-mc_defect_read'),
              args=['./target'], inputs={'target': str(BIN / 'revocation-control')},
              timeout=300, expect='SUPERVISE exit signalled=%d' % SIGPROT, exit=128 + SIGPROT)]
for n, (sym, prediction) in sorted(CASES.items()):
    prog = BIN / ('mct-%02d' % n)
    cases.append(dict(name='mct-%02d-fixed' % n, program=str(prog), args=['fixed', str(n)],
                      timeout=300, expect_regex=r'VERDICT FIXED .*', exit=0))
    # run.py matches `expect` against a WHOLE line and `expect_regex` with fullmatch. A VERDICT
    # line carries text after its keyword, so it must be a regex: passed as a plain `expect` it can
    # never match, and a CORRECT prediction then reads FAIL -- which the first run of this runner did,
    # 26 times, while every case printed exactly the predicted verdict.
    key, marker, code = {
        'NOT-REISSUED': ('expect_regex', r'VERDICT NOT-REISSUED .*', 1),
        'CAUGHT':       ('expect', 'SUPERVISE exit signalled=%d' % SIGPROT, 128 + SIGPROT),
        'ALIASED':      ('expect_regex', r'VERDICT DEFECT-REPRODUCED .*', 0)}[prediction]
    row = dict(name='mct-%02d-buggy' % n, program=str(BIN / ('supervise-' + sym)),
               args=['./target', 'buggy', str(n)], inputs={'target': str(prog)},
               timeout=300, exit=code)
    row[key] = marker
    cases.append(row)
OUT.write_text(json.dumps(cases, indent=2) + '\n')
print('  wrote %s: %d cases, covering all %d case directories' % (OUT, len(cases), len(present)))
for n, (sym, prediction) in sorted(CASES.items()):
    print('    case %2d: %-12s (probe %s)' % (n, prediction, sym))
CASESPY
[ -s "$OUT/cases.json" ] || { echo "CONTROL-FAILED cases.json" >&2; exit 75; }

# A marker whose mtime bounds THIS invocation; every control record must be newer than it.
STAMP="$OUT/.run-started"
: > "$STAMP"

python3 "$CAP/ports/common/host/cheribsd/run.py" "$OUT/run" \
  --sdk "$SDK" --rootfs "$SYSROOT" --image "$IMAGE" --port "$PORT" \
  --abi-probe "$OUT/bin/cheribsd-abi-probe" \
  --runtime-revocation "$REVOCATION" --cases "$OUT/cases.json" --continue-on-failure
rc=$?

for c in cheribsd-abi cheribsd-bounds revocation-control; do
  out="$OUT/run/$c/stdout.txt"
  [ -s "$out" ] || { echo "CONTROL-FAILED $c produced no output: not a reading" >&2; exit 75; }
  [ "$out" -nt "$STAMP" ] || {
    echo "CONTROL-FAILED $c's record predates this run: it is evidence about an earlier boot" >&2
    exit 75; }
done
WANT_REV=1; [ "$REVOCATION" = off ] && WANT_REV=0
grep -q "runtime_revocation=$WANT_REV" "$OUT/run/cheribsd-abi/stdout.txt" \
  || { echo "CONTROL-FAILED cheribsd-abi did not report runtime_revocation=$WANT_REV" >&2; exit 75; }
grep -q "CHERI_BOUNDARY_READY" "$OUT/run/cheribsd-bounds/stdout.txt" \
  || { echo "CONTROL-FAILED cheribsd-bounds did not report ready" >&2; exit 75; }
# The temporal reading is void unless the revoker demonstrably sweeps in THIS guest -- when
# revocation is on. With it off nothing can sweep, and the reading is the
# unprotected one; the requirement below does not apply.
if [ "$REVOCATION" = on ]; then
grep -q "tag_after_sweep=0" "$OUT/run/revocation-control/stdout.txt" \
  || { echo "CONTROL-FAILED revocation-control: the stale capability kept its tag after a forced sweep" >&2; exit 75; }
grep -q "SUPERVISE exit signalled=34" "$OUT/run/revocation-control/stdout.txt" \
  || { echo "CONTROL-FAILED revocation-control did not fault after the sweep: the revoker is not shown to act" >&2; exit 75; }
fi

echo "run-cheribsd: controls fired, including the revocation control; suite exit $rc (non-zero = a prediction did not hold, which is data)"
echo "run-cheribsd: per-case output in $OUT/run/"
exit "$rc"
