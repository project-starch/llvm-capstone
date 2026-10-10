#!/usr/bin/env bash
# This corpus on CheriBSD purecap, one boot per arm:
#
#   CHERI_SDK=... CHERI_SYSROOT=... CHERI_IMAGE=... bash runners/run-cheribsd.sh <arm> <fresh outdir>
#
#   cheribsd-revocation   stock CheriBSD, libc revocation on
#   cheribsd-subobject    the same, cases built with -Xclang -cheri-bounds=subobject-safe
#   cheribsd-carve-bounds the same, cases built with -DFFC_CARVE_BOUNDS: ffc_carve() narrows each
#                         carved region with cheri_bounds_set, the remedy at the carving code
#   poisoncap-spatial     the PoisonCap image (CHERI_* pointing at it), revocation off (mode 0)
#   poisoncap-protected   the PoisonCap image, revocation on (mode 1)
#
# -O0, as every CheriBSD runner here: clang folds a constant out-of-bounds access gcc keeps.
# A fault is read from outside the process by supervise, one build per probe symbol. The platform
# controls (cheribsd-abi, cheribsd-bounds) must fire in the same boot, and so must the carve control:
# FAULT on the carve-bounds arm, COMPLETE on every other. An infrastructure failure exits 75.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$HERE/..
REPO=$(cd "$ROOT/../../../.." && pwd)
CAP=$REPO/capstone
ARM=${1:?usage: run-cheribsd.sh <arm> <fresh outdir>}
OUT=${2:?usage: run-cheribsd.sh <arm> <fresh outdir>}
SDK=${CHERI_SDK:-$HOME/cheri/output/sdk}
SYSROOT=${CHERI_SYSROOT:-$HOME/cheri/rootfs-purecap}
IMAGE=${CHERI_IMAGE:-$HOME/cheri/output/cheribsd-riscv64-purecap.img}
PORT=${CHERI_PORT:-10491}
case "$ARM" in
  cheribsd-revocation)   EXTRA= ;                                       REV=on;  EXPECT=complete ;;
  cheribsd-subobject)    EXTRA="-Xclang -cheri-bounds=subobject-safe";  REV=on;  EXPECT=complete ;;
  cheribsd-carve-bounds) EXTRA=-DFFC_CARVE_BOUNDS;                      REV=on;  EXPECT=fault ;;
  poisoncap-spatial)     EXTRA= ;                                       REV=off; EXPECT=complete ;;
  poisoncap-protected)   EXTRA= ;                                       REV=on;  EXPECT=complete ;;
  *) echo "CONTROL-FAILED unknown arm $ARM" >&2; exit 75 ;;
esac
for p in "$SDK/bin/clang" "$SYSROOT" "$IMAGE"; do
  [ -e "$p" ] || { echo "CONTROL-FAILED missing $p" >&2; exit 75; }
done
[ -e "$OUT" ] && { echo "CONTROL-FAILED $OUT exists: use a fresh output directory" >&2; exit 75; }
mkdir -p "$OUT/bin"

CFLAGS="--target=riscv64-unknown-freebsd13 -march=rv64imafdcxcheri -mabi=l64pc128d"
CFLAGS="$CFLAGS -mno-relax -B$SDK/bin --sysroot=$SYSROOT -std=gnu11 -O0 -fuse-ld=lld"
for dir in "$ROOT"/[0-9][0-9]_*/; do
  n=$(basename "$dir" | cut -c1-2)
  "$SDK/bin/clang" $CFLAGS $EXTRA -I"$ROOT/shared" "$dir/case.c" "$ROOT/shared/driver.c" \
    -o "$OUT/bin/ffc-$n" || { echo "CONTROL-FAILED purecap build case $n" >&2; exit 75; }
done
"$SDK/bin/clang" $CFLAGS $EXTRA -I"$ROOT/shared" "$ROOT/shared/carve-control.c" "$ROOT/shared/driver.c" \
  -o "$OUT/bin/carve-control" || { echo "CONTROL-FAILED carve-control build" >&2; exit 75; }
for sym in ffc_read_probe_u32 ffc_write_probe_u8 ffc_write_probe_u32; do
  "$SDK/bin/clang" $CFLAGS -DPROBE_SYMBOL="\"$sym\"" \
    "$CAP/bug-corpora/cpython/pymalloc-repros/observe/supervise.c" -lutil \
    -o "$OUT/bin/supervise-$sym" || { echo "CONTROL-FAILED supervise build $sym" >&2; exit 75; }
done
"$SDK/bin/clang" $CFLAGS "$CAP/ports/common/host/cheribsd/abi-probe.c" -o "$OUT/bin/cheribsd-abi-probe" \
  || { echo "CONTROL-FAILED abi-probe build" >&2; exit 75; }
python3 "$HERE/cheribsd-cases.py" "$OUT/bin" "$OUT/cases.json" "$EXPECT" \
  || { echo "CONTROL-FAILED cases.json" >&2; exit 75; }
# The field-bounds arm also carries the shared field-crossing control (it must die by SIGPROT).
if [ "$ARM" = cheribsd-subobject ]; then
  export CHERI_EXTRA_CFLAGS=$EXTRA
  . "$CAP/bug-corpora/tools/cheribsd-subobj-control.sh"
  subobj_control_add "$OUT" "$SDK" $CFLAGS || exit 75
fi
echo "$ARM" > "$OUT/arm"

STAMP="$OUT/.run-started"; : > "$STAMP"
python3 "$CAP/ports/common/host/cheribsd/run.py" "$OUT/run" \
  --sdk "$SDK" --rootfs "$SYSROOT" --image "$IMAGE" --port "$PORT" \
  --abi-probe "$OUT/bin/cheribsd-abi-probe" \
  --runtime-revocation "$REV" --cases "$OUT/cases.json" --continue-on-failure
rc=$?
# A boot that died partway is not a reading: --continue-on-failure keeps a failing CASE from
# stopping the run, but only run.py's ran_all_cases says every selected case actually executed.
# Without this, a guest that died after the controls read as "suite exit 1, which is data".
python3 - "$OUT/run/summary.json" <<'RANALL' || { echo "INFRA: the boot did not run every case (ran_all_cases is not true): not a reading" >&2; exit 75; }
import json, sys
sys.exit(0 if json.load(open(sys.argv[1])).get("ran_all_cases") is True else 1)
RANALL
for c in cheribsd-abi cheribsd-bounds carve-control-fixed carve-control-buggy; do
  out="$OUT/run/$c/stdout.txt"
  [ -s "$out" ] || { echo "CONTROL-FAILED $c produced no output: not a reading" >&2; exit 75; }
  [ "$out" -nt "$STAMP" ] || { echo "CONTROL-FAILED $c's record predates this run" >&2; exit 75; }
done
WANT_REV=1; [ "$REV" = off ] && WANT_REV=0
grep -q "runtime_revocation=$WANT_REV" "$OUT/run/cheribsd-abi/stdout.txt" \
  || { echo "CONTROL-FAILED cheribsd-abi did not report runtime_revocation=$WANT_REV" >&2; exit 75; }
grep -q "CHERI_BOUNDARY_READY" "$OUT/run/cheribsd-bounds/stdout.txt" \
  || { echo "CONTROL-FAILED cheribsd-bounds did not report ready" >&2; exit 75; }
python3 - "$OUT/run/summary.json" "$EXPECT" <<'PY' || { echo "CONTROL-FAILED the carve control did not read as this arm requires" >&2; exit 75; }
import json, sys
rows = {r.get('name'): r for r in json.load(open(sys.argv[1]))['results'] if isinstance(r, dict)}
want = 162 if sys.argv[2] == 'fault' else 0
ok = rows.get('carve-control-fixed', {}).get('exit') == 0 and rows.get('carve-control-buggy', {}).get('exit') == want
sys.exit(0 if ok else 1)
PY
[ "$ARM" = cheribsd-subobject ] && { subobj_control_check "$OUT" || exit 75; }
echo "run-cheribsd: $ARM controls fired; suite exit $rc (non-zero = a row differs from the arm's prediction, which is data)"
exit "$rc"
