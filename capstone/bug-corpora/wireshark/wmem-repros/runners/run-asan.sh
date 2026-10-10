#!/usr/bin/env bash
# The native-detect arm: every case under AddressSanitizer, wmem itself included, buggy and fixed.
#
#   bash runners/run-asan.sh <fresh outdir>
#
# wmem is built hosted (ports/wireshark/wmem/src/shared/system.c): g_malloc and g_free ARE the host's
# malloc and free, at the requested size, as a stock build gets them from glib.
# So ASan sees every block wmem asks the system for, every jumbo, and every g_free -- what it would
# see in an upstream fuzz build under the wmem allocator override -- and a chunk wmem carves inside
# one of those blocks is still invisible to it, which is the reading.
#
# Until 2026-10-10 this arm ran on the hosted bump arena (one 384 MiB aligned_alloc, nothing ever
# handed back), where no reading but "silent" was possible and the only controls were separate
# programs. Now the positive control is wmem's own: control 01 (controls/01_control_uaf_jumbo), a
# jumbo the packet reset frees with g_free and then reads, built with the SAME options in this run,
# must report heap-use-after-free; and the plain asan-control must report one byte past a block.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/.." && pwd)
PORT=$(cd -- "$ROOT/../../../ports/wireshark/wmem" && pwd)
TOOLS=$(cd -- "$ROOT/../../tools" && pwd)
OUT=${1:?usage: run-asan.sh <fresh outdir>}
SAN="-fsanitize=address -fno-omit-frame-pointer -g -O0"
OPTS=(-DCMAKE_C_FLAGS="$SAN" -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address)
bash "$HERE/run-native.sh" "$OUT" "${OPTS[@]}" > "$OUT.native.log" 2>&1
rc=$?; [ $rc -eq 75 ] && { echo "CONTROL-FAILED ASan build (see $OUT.native.log)" >&2; exit 75; }
CTL="$OUT-controls"
cmake --preset native -S "$PORT" -B "$CTL" -DWM_CORPUS_DIR="$ROOT/controls" \
  "${OPTS[@]}" > "$CTL.configure.log" 2>&1 && cmake --build "$CTL" -j "${JOBS:-8}" > "$CTL.build.log" 2>&1 \
  || { echo "CONTROL-FAILED controls build (see $CTL.build.log)" >&2; exit 75; }
jumbo=$(ls "$CTL"/bin/01-* 2>/dev/null | head -1)
[ -x "$jumbo" ] || { echo "CONTROL-FAILED control 01 not built" >&2; exit 75; }
${CC:-cc} $SAN -o "$OUT/bin/asan-control" "$TOOLS/asan-control.c" || exit 75
python3 "$TOOLS/run-native-asan.py" --corpus "$ROOT" --bin "$OUT/bin/case-{nn}" --out "$OUT/run" \
  --buggy-args "0 {n} buggy" --fixed-args "0 {n} fixed" \
  --build-note "$(${CC:-cc} --version | head -1); $SAN; wmem and the port built with the same flags; g_malloc is malloc" \
  --control "$jumbo 0 1 buggy=heap-use-after-free" \
  --control "$OUT/bin/asan-control past 4096=heap-buffer-overflow"
