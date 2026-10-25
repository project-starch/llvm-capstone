#!/usr/bin/env bash
# This corpus on CheriBSD purecap: every case, fixed then buggy, against the buffer-pool port built
# for purecap (cmake --preset cheribsd), as the 2026-10-06 reading was taken.
#
#   CHERI_SDK=... CHERI_SYSROOT=... CHERI_IMAGE=... bash runners/run-cheribsd.sh <fresh outdir>
#
# Optional, for another arm on the same cases (defaults reproduce cheribsd-revocation):
#   CHERI_EXTRA_CFLAGS  e.g. "-Xclang -cheri-bounds=subobject-safe" -- the cheribsd-subobject arm,
#                       applied to the cases, whose member accesses are what it narrows;
#   CHERI_REVOCATION    on|off (off: a revocation-off reading).
#
# The fixed arm runs directly. The BUGGY arm runs under supervise (cpython/pymalloc-repros/observe/
# supervise.c, anchored on ff2_case_run): the status is the same 162 = 128 + SIGPROT, and the fault pc is
# kept, so tools/attribute-cheribsd-faults.py can name the function it landed in (attribution.tsv).
# Until 2026-10-10 the buggy arm ran directly and a catch carried no pc at all. An infrastructure
# failure exits 75.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/.." && pwd)
REPO=$(cd "$ROOT/../../../.." && pwd)
CAP=$REPO/capstone
PORT_DIR=$CAP/ports/ffmpeg/buffer-pool
SDK=${CHERI_SDK:-$HOME/cheri/output/sdk}
SYSROOT=${CHERI_SYSROOT:-$HOME/cheri/rootfs-purecap}
IMAGE=${CHERI_IMAGE:-$HOME/cheri/output/cheribsd-riscv64-purecap.img}
PORT=${CHERI_PORT:-10485}
OUT=${1:?usage: run-cheribsd.sh <fresh outdir>}
for p in "$SDK/bin/clang" "$SYSROOT" "$IMAGE"; do
  [ -e "$p" ] || { echo "CONTROL-FAILED missing $p" >&2; exit 75; }
done
[ -e "$OUT" ] && { echo "CONTROL-FAILED $OUT exists: use a fresh output directory" >&2; exit 75; }
mkdir -p "$OUT/bin"
REVOCATION=${CHERI_REVOCATION:-on}
case "$REVOCATION" in on|off) ;; *) echo "CONTROL-FAILED CHERI_REVOCATION=$REVOCATION" >&2; exit 75;; esac

# The port, built purecap. Its own objects do NOT take CHERI_EXTRA_CFLAGS: the arm under test is
# about the cases' member accesses, and the library is the platform it runs on.
LIB=$OUT/port
CHERI_SDK=$SDK CHERI_SYSROOT=$SYSROOT cmake --preset cheribsd -S "$PORT_DIR" -B "$LIB" > "$OUT/port.log" 2>&1 \
  && cmake --build "$LIB" --target ffmpeg-pool -j "${JOBS:-8}" >> "$OUT/port.log" 2>&1 \
  || { echo "CONTROL-FAILED purecap port build (see $OUT/port.log)" >&2; exit 75; }
case "$("$SDK/bin/llvm-readobj" --file-headers "$LIB/libffmpeg-pool.a" 2>/dev/null)" in
  *EF_RISCV_CHERIABI*) : ;;
  *) echo "CONTROL-FAILED $LIB/libffmpeg-pool.a is not a purecap archive" >&2; exit 75 ;;
esac

# -O0: case 0's index is a compile-time constant one past its member, and -O1 folds the store away
# (results/20261006-cheribsd/README.md).
CFLAGS="--target=riscv64-unknown-freebsd13 -march=rv64imafdcxcheri -mabi=l64pc128d"
CFLAGS="$CFLAGS -mno-relax -B$SDK/bin --sysroot=$SYSROOT -std=gnu11 -O0 -fuse-ld=lld"
for dir in "$ROOT"/[0-9][0-9]_*/; do
  n=$(basename "$dir" | cut -c1-2)
  "$SDK/bin/clang" $CFLAGS ${CHERI_EXTRA_CFLAGS:-} -I"$ROOT/shared" -I"$PORT_DIR/src/shared" \
    -I"$LIB/sources/ffmpeg-ported" -I"$PORT_DIR/cmake/replay-config" -I"$CAP/runtime/include" \
    "$dir/case.c" "$ROOT/shared/driver.c" -L"$LIB" -lffmpeg-pool \
    -o "$OUT/bin/so-$n" || { echo "CONTROL-FAILED purecap build case $n" >&2; exit 75; }
done
"$SDK/bin/clang" $CFLAGS "$CAP/ports/common/host/cheribsd/abi-probe.c" \
  -o "$OUT/bin/cheribsd-abi-probe" || { echo "CONTROL-FAILED abi-probe build" >&2; exit 75; }
"$SDK/bin/clang" $CFLAGS -DPROBE_SYMBOL='"ff2_case_run"' \
  "$CAP/bug-corpora/cpython/pymalloc-repros/observe/supervise.c" -lutil \
  -o "$OUT/bin/supervise" || { echo "CONTROL-FAILED supervise build" >&2; exit 75; }

python3 - "$OUT/bin" "$OUT/cases.json" "$ROOT" <<'PY'
import json, pathlib, sys
BIN, OUT, CORPUS = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
present = sorted(int(d.name[:2]) for d in CORPUS.glob('[0-9][0-9]_*') if d.is_dir())
built = sorted(int(p.name.split('-')[1]) for p in BIN.glob('so-[0-9][0-9]'))
if present != built:
    sys.exit(f'CONTROL-FAILED cases.json: corpus has {present}, built {built}')
cases = []
for n in built:
    p = BIN / f'so-{n:02d}'
    # The expectation is only what run.py needs to record the arm; the reading is taken from the
    # status and the VERDICT line afterwards, so a refuted prediction is data, not a lost row.
    cases.append(dict(name=f'so-{n:02d}-fixed', program=str(p), args=['fixed', str(n)],
                      timeout=300, expect_regex=r'VERDICT FIXED .*', exit=0))
    cases.append(dict(name=f'so-{n:02d}-buggy', program=str(BIN / 'supervise'),
                      args=['./target', 'buggy', str(n)], inputs={'target': str(p)},
                      timeout=300, expect_regex=r'SUPERVISE exit .*', exit=0))
OUT.write_text(json.dumps(cases, indent=2) + '\n')
print(f'  cases.json: {len(cases)} arms covering all {len(present)} case directories')
PY
[ -s "$OUT/cases.json" ] || { echo "CONTROL-FAILED cases.json" >&2; exit 75; }
. "$CAP/bug-corpora/tools/cheribsd-subobj-control.sh"
subobj_control_add "$OUT" "$SDK" ${CFLAGS/${CHERI_EXTRA_CFLAGS:-@@none@@}/} || exit 75


STAMP="$OUT/.run-started"; : > "$STAMP"
python3 "$CAP/ports/common/host/cheribsd/run.py" "$OUT/run" \
  --sdk "$SDK" --rootfs "$SYSROOT" --image "$IMAGE" --port "$PORT" \
  --abi-probe "$OUT/bin/cheribsd-abi-probe" \
  --runtime-revocation "$REVOCATION" --cases "$OUT/cases.json" --continue-on-failure
rc=$?
# A boot that died partway is not a reading: --continue-on-failure keeps a failing CASE from
# stopping the run, but only run.py's ran_all_cases says every selected case actually executed.
# Without this, a guest that died after the controls read as "suite exit 1, which is data".
python3 - "$OUT/run/summary.json" <<'RANALL' || { echo "INFRA: the boot did not run every case (ran_all_cases is not true): not a reading" >&2; exit 75; }
import json, sys
sys.exit(0 if json.load(open(sys.argv[1])).get("ran_all_cases") is True else 1)
RANALL
for c in cheribsd-abi cheribsd-bounds; do
  out="$OUT/run/$c/stdout.txt"
  [ -s "$out" ] || { echo "CONTROL-FAILED $c produced no output: not a reading" >&2; exit 75; }
  [ "$out" -nt "$STAMP" ] || { echo "CONTROL-FAILED $c's record predates this run" >&2; exit 75; }
done
WANT_REV=1; [ "$REVOCATION" = off ] && WANT_REV=0
grep -q "runtime_revocation=$WANT_REV" "$OUT/run/cheribsd-abi/stdout.txt" \
  || { echo "CONTROL-FAILED cheribsd-abi did not report runtime_revocation=$WANT_REV" >&2; exit 75; }
grep -q "CHERI_BOUNDARY_READY" "$OUT/run/cheribsd-bounds/stdout.txt" \
  || { echo "CONTROL-FAILED cheribsd-bounds did not report ready" >&2; exit 75; }
subobj_control_check "$OUT" || exit 75
# Where each buggy fault landed, against the case's own declared sites (case.json fault_sites) or its
# labelled probe. A fault anywhere else is data, and says so in the table.
mapfile -t ATTR < <(python3 - "$ROOT" "$OUT/bin" <<'PY'
import json, pathlib, re, sys
root, bin_ = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
for d in sorted(root.glob('[0-9][0-9]_*')):
    n = d.name[:2]
    print('--arm'); print(f'so-{n}-buggy={bin_}/so-{n}')
    sites = json.loads((d / 'case.json').read_text()).get('fault_sites') or []
    probes = sorted(set(re.findall(r'\b((?:[a-z0-9]+_)?(?:read|write)_probe)\(', (d / 'case.c').read_text())))
    if sites or probes:
        print('--sites'); print(f'so-{n}-buggy=' + ','.join(sites + probes))
PY
)
python3 "$CAP/bug-corpora/tools/attribute-cheribsd-faults.py" --run "$OUT/run" --nm "$SDK/bin/llvm-nm" \
  --sysroot "$SYSROOT" --anchor ff2_case_run "${ATTR[@]}" > "$OUT/attribution.tsv"
arc=$?
echo "run-cheribsd: controls fired; suite exit $rc (non-zero = an arm's status was not the recorded one, which is data);" \
     "attribution exit $arc (non-zero = a fault not at a declared site or probe, which is data) -- $OUT/attribution.tsv"
exit "$rc"
