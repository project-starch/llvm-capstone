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

for dir in "$ROOT"/[0-9][0-9]_*/; do
  n=$(basename "$dir" | cut -c1-2)
  "$SDK/bin/clang" $CFLAGS -I"$ROOT/shared" "$dir/case.c" "$ROOT/shared/driver.c" \
    -o "$OUT/bin/mch-$n" || { echo "CONTROL-FAILED purecap build case $n" >&2; exit 75; }
done
"$SDK/bin/clang" $CFLAGS -DPROBE_SYMBOL='"mch_write_probe"' \
  "$CAP/bug-corpora/cpython/pymalloc-repros/observe/supervise.c" -lutil \
  -o "$OUT/bin/supervise" || { echo "CONTROL-FAILED supervise build" >&2; exit 75; }
"$SDK/bin/clang" $CFLAGS "$CAP/ports/common/host/cheribsd/abi-probe.c" \
  -o "$OUT/bin/cheribsd-abi-probe" || { echo "CONTROL-FAILED abi-probe build" >&2; exit 75; }

# PREDICTION, written before the run: NOT CAUGHT, and this is a REFUTED earlier
# prediction rather than a guess. The crossing is one byte past calloc(1, 9), and
# CheriBSD's malloc bounds to the allocator's USABLE size: measured in-guest,
# request 9 yields a capability of length 16, so offset 9 is inside the bounds
# and no fault is possible. Contrast the wireshark sibling, whose 65471-byte
# crossing past an exactly-size-classed 8192 does leave the allocation.
python3 - "$OUT/bin" "$OUT/cases.json" <<'PY'
import json, pathlib, sys
BIN, OUT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
SIGPROT = 34
# The buggy arm expects a COMPLETION, because that is what was MEASURED and the
# CATCH prediction was refuted. Encoding the refuted prediction instead would
# make this suite fail every time and a genuine change invisible. SIGPROT is
# still imported so the contrast with the wireshark sibling is readable.
cases = [dict(name='mch-00-fixed', program=str(BIN/'mch-00'), args=['fixed','0'],
              timeout=300, expect_regex=r'VERDICT FIXED .*', exit=0),
         dict(name='mch-00-buggy', program=str(BIN/'supervise'), args=['./target','buggy','0'],
              inputs={'target': str(BIN/'mch-00')}, timeout=300,
              expect='SUPERVISE exit status=0', exit=0)]
assert SIGPROT == 34
OUT.write_text(json.dumps(cases, indent=2) + '\n')
PY
[ -s "$OUT/cases.json" ] || { echo "CONTROL-FAILED cases.json" >&2; exit 75; }

python3 "$CAP/ports/common/host/cheribsd/run.py" "$OUT/run" \
  --sdk "$SDK" --rootfs "$SYSROOT" --image "$IMAGE" --port "$PORT" \
  --abi-probe "$OUT/bin/cheribsd-abi-probe" \
  --runtime-revocation on --cases "$OUT/cases.json" --continue-on-failure
rc=$?

# The suite's status cannot distinguish "an oracle did not hold" -- which is DATA
# -- from "the boot produced nothing". The platform controls decide that.
for c in cheribsd-abi cheribsd-bounds; do
  [ -s "$OUT/run/$c/stdout.txt" ] \
    || { echo "CONTROL-FAILED $c produced no output: not a reading" >&2; exit 75; }
done
grep -q "runtime_revocation=1" "$OUT/run/cheribsd-abi/stdout.txt" \
  || { echo "CONTROL-FAILED cheribsd-abi did not report runtime_revocation=1" >&2; exit 75; }
grep -q "CHERI_BOUNDARY_READY" "$OUT/run/cheribsd-bounds/stdout.txt" \
  || { echo "CONTROL-FAILED cheribsd-bounds did not report ready" >&2; exit 75; }

echo "run-cheribsd: controls fired; suite exit $rc (non-zero = an oracle did not hold, which is data)"
exit "$rc"
