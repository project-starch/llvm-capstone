#!/usr/bin/env bash
# This corpus on stock CheriBSD purecap, with libc revocation ON.
#
# Committed because the 2026-10-06 reading in results/20261006-cheribsd/ was
# taken with an ad-hoc command line, which left the bundle citing evidence with
# no in-tree reproduction path. Everything the run needs is here.
#
#   CHERI_SDK=~/cheri/output/sdk CHERI_SYSROOT=~/cheri/rootfs-purecap \
#   CHERI_IMAGE=~/cheri/output/cheribsd-riscv64-purecap.img \
#     bash runners/run-cheribsd.sh [outdir]
#
# THREE THINGS THAT ARE LOAD-BEARING, each learned by getting it wrong:
#
#  1. -O0. clang folds a constant out-of-bounds access that gcc keeps, which made
#     a sibling corpus's case read INCONCLUSIVE on purecap with every control
#     passing. The native arms use gcc and are unaffected, so this only ever
#     shows up here.
#  2. One supervise per probe symbol. PROBE_SYMBOL is compile-time, and the
#     cases use three different probes. The probes are DEFINED ONCE in
#     shared/driver.c (not static in the header) precisely so the symbol is
#     resolvable from the ELF and a fault can be attributed to it.
#  3. The REAL abi probe, not this corpus's size probe, for --abi-probe. Passing
#     the wrong one makes cheribsd-abi and cheribsd-bounds FAIL, and a suite
#     whose platform controls did not fire is not a reading however plausible
#     its case rows look. That happened on the first attempt at this run.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$HERE/..
REPO=$(cd "$ROOT/../../../.." && pwd)
CAP=$REPO/capstone

SDK=${CHERI_SDK:-$HOME/cheri/output/sdk}
SYSROOT=${CHERI_SYSROOT:-$HOME/cheri/rootfs-purecap}
IMAGE=${CHERI_IMAGE:-$HOME/cheri/output/cheribsd-riscv64-purecap.img}
PORT=${CHERI_PORT:-10477}
OUT=${1:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/ff-plain-heap-cheribsd}

for p in "$SDK/bin/clang" "$SYSROOT" "$IMAGE"; do
  [ -e "$p" ] || { echo "CONTROL-FAILED missing $p" >&2; exit 75; }
done
mkdir -p "$OUT/bin"

CFLAGS="--target=riscv64-unknown-freebsd13 -march=rv64imafdcxcheri -mabi=l64pc128d"
CFLAGS="$CFLAGS -mno-relax -B$SDK/bin --sysroot=$SYSROOT -std=gnu11 -O0 -fuse-ld=lld"
# Optional, for another arm on the same cases: CHERI_EXTRA_CFLAGS (e.g. -Xclang
# -cheri-bounds=subobject-safe for cheribsd-subobject) and CHERI_REVOCATION=off (a revocation-off reading).
# Both default to this runner's own arm, so an unset environment builds and boots exactly as before.
CFLAGS="$CFLAGS ${CHERI_EXTRA_CFLAGS:-}"
REVOCATION=${CHERI_REVOCATION:-on}
case "$REVOCATION" in on|off) ;; *) echo "CONTROL-FAILED CHERI_REVOCATION=$REVOCATION" >&2; exit 75;; esac

# One program per case.
for dir in "$ROOT"/[0-9][0-9]_*/; do
  n=$(basename "$dir" | cut -c1-2)
  "$SDK/bin/clang" $CFLAGS -I"$ROOT/shared" "$dir/case.c" "$ROOT/shared/driver.c" \
    -o "$OUT/bin/ffh-$n" || { echo "CONTROL-FAILED purecap build case $n" >&2; exit 75; }
done

# One supervise per probe symbol (see note 2).
for sym in ffh_read_probe ffh_read_probe_u8 ffh_write_probe_u8; do
  "$SDK/bin/clang" $CFLAGS -DPROBE_SYMBOL="\"$sym\"" \
    "$CAP/bug-corpora/cpython/pymalloc-repros/observe/supervise.c" -lutil \
    -o "$OUT/bin/supervise-$sym" \
    || { echo "CONTROL-FAILED supervise build $sym" >&2; exit 75; }
done

# The platform's own ABI probe, which also serves cheribsd-bounds (see note 3).
"$SDK/bin/clang" $CFLAGS "$CAP/ports/common/host/cheribsd/abi-probe.c" \
  -o "$OUT/bin/cheribsd-abi-probe" \
  || { echo "CONTROL-FAILED abi-probe build" >&2; exit 75; }

# The per-case capability-length probe. The lengths are read IN-GUEST for this
# corpus's own request sizes, because CheriBSD's malloc bounds to the
# allocator's USABLE size and a size class is a step function -- the table
# measured for a sibling corpus is not transferable, which is what refuted one
# of this corpus's own catch predictions.
"$SDK/bin/clang" $CFLAGS "$ROOT/shared/cap-bounds.c" -o "$OUT/bin/cap-bounds" \
  || { echo "CONTROL-FAILED cap-bounds build" >&2; exit 75; }

python3 "$HERE/cheribsd-cases.py" "$OUT/bin" "$OUT/cases.json" \
  || { echo "CONTROL-FAILED cases.json" >&2; exit 75; }
. "$CAP/bug-corpora/tools/cheribsd-subobj-control.sh"
subobj_control_add "$OUT" "$SDK" ${CFLAGS/${CHERI_EXTRA_CFLAGS:-@@none@@}/} || exit 75


# --continue-on-failure so a refuted prediction does not hide the rows after it.
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

# The suite exits non-zero when any arm's oracle does not hold. That is DATA, not
# infrastructure -- a refuted CATCH prediction is exactly such an arm -- so the
# runner's status alone cannot say whether the run counts. What decides that is
# whether the PLATFORM controls fired, and that is checked here rather than left
# to a reader: a suite whose controls did not fire is exit 75, not a reading.
for c in cheribsd-abi cheribsd-bounds; do
  out="$OUT/run/$c/stdout.txt"
  [ -s "$out" ] || { echo "CONTROL-FAILED $c produced no output: not a reading" >&2; exit 75; }
  # ... and it must come from THIS boot, not from a directory a previous run left behind.
  [ "$out" -nt "$STAMP" ] || {
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
echo "run-cheribsd: per-case output in $OUT/run/"
exit "$rc"
