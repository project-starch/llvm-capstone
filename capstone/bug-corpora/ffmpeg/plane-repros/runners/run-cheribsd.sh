#!/usr/bin/env bash
# This corpus on stock CheriBSD purecap, with libc revocation ON.
#
#   CHERI_SDK=~/cheri/output/sdk CHERI_SYSROOT=~/cheri/rootfs-purecap \
#   CHERI_IMAGE=~/cheri/output/cheribsd-riscv64-purecap.img \
#   FFPURECAP=/tmp/capstone/ffmpeg-purecap/build \
#     bash runners/run-cheribsd.sh [outdir]
#
# WHAT MAKES THIS CORPUS DIFFERENT FROM ITS SIBLINGS, and why its CheriBSD arm went unrun so long:
# the cases link REAL libavutil. av_frame_alloc/av_frame_get_buffer/av_frame_free are what carve the
# planes, and the row's claim is "real allocator, model consumer", so the allocator cannot be
# modelled away. plain-heap-repros links nothing, so copying its runner is NOT sufficient --
# a purecap libavutil has to exist first. Build one with the script beside this one,
# runners/build-libavutil-purecap.sh (out of tree, so the native build the native arm uses is
# untouched), and check it really is purecap rather than trusting the command line:
#
#   llvm-readobj --file-headers libavutil.a  ->  EF_RISCV_CAP_MODE, EF_RISCV_CHERIABI, EM_RISCV
#
# The other load-bearing points are the siblings' and are repeated because each was learned by
# getting it wrong:
#
#  1. -O0. clang folds a constant out-of-bounds access that gcc keeps, which made a sibling's case
#     read INCONCLUSIVE on purecap with every control passing. The native arms use gcc.
#  2. The probe is DEFINED ONCE in shared/driver.c, non-static, so `supervise` can resolve
#     ffp_read_probe from the ELF and a fault can be required to land inside it. It used to be
#     static in corpus.h, which would have made every arm here UNRESOLVED.
#  3. The platform's OWN abi probe for --abi-probe. Passing anything else makes cheribsd-abi and
#     cheribsd-bounds FAIL, and a suite whose platform controls did not fire is not a reading
#     however plausible its case rows look.
#
# PREDICTED READING, written before the first run: this case COMPLETES on both arms; it is NOT
# caught. The crossing leaves the alpha PLANE but stays inside the frame's single AVBuffer --
# measured slack 1024 bytes against a logical 160 -- and CHERI bounds the allocation, not the plane.
# A FAULT here would mean the reduction is not doing what it claims, not that CHERI improved. The
# expected zero is the point of the row: it is the nested-spatial cell, where a per-allocation
# bound is blind by construction.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$HERE/..
REPO=$(cd "$ROOT/../../../.." && pwd)
CAP=$REPO/capstone

SDK=${CHERI_SDK:-$HOME/cheri/output/sdk}
SYSROOT=${CHERI_SYSROOT:-$HOME/cheri/rootfs-purecap}
IMAGE=${CHERI_IMAGE:-$HOME/cheri/output/cheribsd-riscv64-purecap.img}
PORT=${CHERI_PORT:-10483}
FFPURECAP=${FFPURECAP:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/ffmpeg-purecap/build}
FFSRC=${FFPLANE_SRC:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/ffmpeg-native-vidstab/src}
OUT=${1:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/ff-plane-cheribsd}

LIB=$FFPURECAP/libavutil/libavutil.a
for p in "$SDK/bin/clang" "$SYSROOT" "$IMAGE" "$LIB" "$FFSRC/libavutil/frame.h"; do
  [ -e "$p" ] || { echo "CONTROL-FAILED missing $p" >&2; exit 75; }
done

# The library must be PURECAP, not a native one that happens to sit at that path. Checked, because
# linking a native archive would fail confusingly late.
#
# CAPTURED, NOT PIPED INTO grep -q, and that is not style. `grep -q` exits at the first match and
# closes the pipe; llvm-readobj then dies of SIGPIPE, and under `set -o pipefail` the PIPELINE
# inherits that non-zero. So the obvious `readobj | grep -q` reports "not purecap" for a library
# that is purecap -- it did exactly that on the first run of this script, while the same line
# pasted into an interactive shell (no pipefail) passed. Never put a pipe between a check and its
# exit status.
FLAGS=$("$SDK/bin/llvm-readobj" --file-headers "$LIB" 2>/dev/null || true)
case "$FLAGS" in
  *EF_RISCV_CHERIABI*) : ;;
  *) echo "CONTROL-FAILED $LIB is not a purecap archive (no EF_RISCV_CHERIABI)" >&2; exit 75 ;;
esac

mkdir -p "$OUT/bin"
CFLAGS="--target=riscv64-unknown-freebsd13 -march=rv64imafdcxcheri -mabi=l64pc128d"
CFLAGS="$CFLAGS -mno-relax -B$SDK/bin --sysroot=$SYSROOT -std=gnu11 -O0 -fuse-ld=lld"
# Optional, for another arm on the same cases: CHERI_EXTRA_CFLAGS (e.g. -Xclang
# -cheri-bounds=subobject-safe for cheribsd-subobject) and CHERI_REVOCATION=off (a revocation-off reading).
# Both default to this runner's own arm, so an unset environment builds and boots exactly as before.
CFLAGS="$CFLAGS ${CHERI_EXTRA_CFLAGS:-}"
REVOCATION=${CHERI_REVOCATION:-on}
case "$REVOCATION" in on|off) ;; *) echo "CONTROL-FAILED CHERI_REVOCATION=$REVOCATION" >&2; exit 75;; esac

for dir in "$ROOT"/[0-9][0-9]_*/; do
  n=$(basename "$dir" | cut -c1-2)
  "$SDK/bin/clang" $CFLAGS -I"$ROOT/shared" -I"$FFSRC" -I"$FFPURECAP" \
    "$dir/case.c" "$ROOT/shared/driver.c" "$LIB" -lm \
    -o "$OUT/bin/ffp-$n" || { echo "CONTROL-FAILED purecap build case $n" >&2; exit 75; }
done

"$SDK/bin/clang" $CFLAGS -DPROBE_SYMBOL='"ffp_read_probe"' \
  "$CAP/bug-corpora/cpython/pymalloc-repros/observe/supervise.c" -lutil \
  -o "$OUT/bin/supervise" || { echo "CONTROL-FAILED supervise build" >&2; exit 75; }

"$SDK/bin/clang" $CFLAGS "$CAP/ports/common/host/cheribsd/abi-probe.c" \
  -o "$OUT/bin/cheribsd-abi-probe" || { echo "CONTROL-FAILED abi-probe build" >&2; exit 75; }

# Both arms are expected to COMPLETE (exit 0) -- see the predicted reading above. The arms are run
# directly rather than under supervise: supervise exists to attribute a FAULT to the probe, and the
# prediction here is that there is no fault. If an arm faults, the suite reports it and the
# attribution run is the follow-up, not this.
python3 - "$OUT/bin" "$OUT/cases.json" <<'PY'
import json, pathlib, sys
BIN, OUT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
cases = []
for p in sorted(BIN.glob('ffp-[0-9][0-9]')):
    n = int(p.name.split('-')[1])
    cases.append(dict(name=f'{p.name}-fixed', program=str(p), args=['fixed', str(n)],
                      timeout=300, expect_regex=r'VERDICT FIXED .*', exit=0))
    cases.append(dict(name=f'{p.name}-buggy', program=str(p), args=['buggy', str(n)],
                      timeout=300, expect_regex=r'VERDICT DEFECT-REPRODUCED .*', exit=0))
OUT.write_text(json.dumps(cases, indent=2) + '\n')
PY
[ -s "$OUT/cases.json" ] || { echo "CONTROL-FAILED cases.json" >&2; exit 75; }
. "$CAP/bug-corpora/tools/cheribsd-subobj-control.sh"
subobj_control_add "$OUT" "$SDK" ${CFLAGS/${CHERI_EXTRA_CFLAGS:-@@none@@}/} || exit 75


python3 "$CAP/ports/common/host/cheribsd/run.py" "$OUT/run" \
  --sdk "$SDK" --rootfs "$SYSROOT" --image "$IMAGE" --port "$PORT" \
  --abi-probe "$OUT/bin/cheribsd-abi-probe" \
  --runtime-revocation "$REVOCATION" --cases "$OUT/cases.json" --continue-on-failure
rc=$?

# The suite's status cannot distinguish "an oracle did not hold" -- which is DATA -- from "the boot
# produced nothing". The platform controls decide that, and they are checked here rather than left
# to a reader.
for c in cheribsd-abi cheribsd-bounds; do
  out="$OUT/run/$c/stdout.txt"
  [ -s "$out" ] || { echo "CONTROL-FAILED $c produced no output: not a reading" >&2; exit 75; }
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
