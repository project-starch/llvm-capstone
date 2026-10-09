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

# PREDICTION, written before the run: CAUGHT. The crossing is 65471 bytes past a
# g_malloc(8192), and 8192 is exactly a size class, so it leaves the USABLE
# allocation by far more than any slack. Contrast the memcached sibling, whose
# 1-byte crossing past a 9-byte request is absorbed by a 16-byte capability.
python3 - "$OUT/bin" "$OUT/cases.json" <<'PY'
import json, pathlib, sys
BIN, OUT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
SIGPROT = 34
cases = []
for p in sorted(BIN.glob('wsh-[0-9][0-9]')):
    n = int(p.name.split('-')[1])
    cases.append(dict(name=f'{p.name}-fixed', program=str(p), args=['fixed', str(n)],
                      timeout=300, expect_regex=r'VERDICT FIXED .*', exit=0))
    cases.append(dict(name=f'{p.name}-buggy', program=str(BIN/'supervise'),
                      args=['./target','buggy',str(n)], inputs={'target': str(p)},
                      timeout=300, expect=f'SUPERVISE exit signalled={SIGPROT}',
                      exit=128+SIGPROT))
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
