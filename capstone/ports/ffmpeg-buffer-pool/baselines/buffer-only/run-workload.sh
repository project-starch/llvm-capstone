#!/usr/bin/env bash
set -euo pipefail
WORK=${FFPOOL_WORK:-/tmp/capstone/ffmpeg-buffer-pool}
OUT=${1:?usage: run-workload.sh RESULT-DIR [duration-seconds] [size]}
DURATION=${2:-300}
SIZE=${3:-1280x720}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
for name in input.mkv stock.framemd5 traced.framemd5 original.bin workload.json; do
    [[ ! -e "$OUT/$name" ]] || { echo "refusing to overwrite $OUT/$name" >&2; exit 2; }
done
STOCK="$WORK/workload-stock/ffmpeg"
TRACED="$WORK/workload-traced/ffmpeg"
COMMON=(-nostdin -hide_banner -loglevel warning -threads 1 -filter_threads 1 -filter_complex_threads 1)
"$STOCK" "${COMMON[@]}" -f lavfi \
    -i "testsrc2=size=$SIZE:rate=30:duration=$DURATION" -c:v mpeg4 -q:v 3 -y "$OUT/input.mkv" \
    > "$OUT/generate.stdout" 2> "$OUT/generate.stderr"
"$STOCK" "${COMMON[@]}" -i "$OUT/input.mkv" -f framemd5 "$OUT/stock.framemd5" \
    > "$OUT/stock.stdout" 2> "$OUT/stock.stderr"
FFPOOL_TRACE="$OUT/original.bin" "$TRACED" "${COMMON[@]}" \
    -i "$OUT/input.mkv" -f framemd5 "$OUT/traced.framemd5" \
    > "$OUT/traced.stdout" 2> "$OUT/traced.stderr"
cmp "$OUT/stock.framemd5" "$OUT/traced.framemd5"
sha256sum "$OUT/input.mkv" "$OUT/stock.framemd5" "$OUT/traced.framemd5" \
    "$OUT/original.bin" "$STOCK" "$TRACED" > "$OUT/workload.sha256"
python3 - "$OUT" "$DURATION" "$SIZE" <<'PY'
import json, pathlib, platform, sys
out = pathlib.Path(sys.argv[1])
frames = sum(not line.startswith('#') for line in (out / 'stock.framemd5').read_text().splitlines())
expected = int(sys.argv[2]) * 30
if frames != expected:
    raise SystemExit(f'frame count mismatch: {frames} != {expected}')
info = dict(ffmpeg='9.0.1', duration_seconds=int(sys.argv[2]), dimensions=sys.argv[3],
            frames=frames, fps=30, workload='MPEG-4 decode to framemd5',
            output_identical=True, machine=platform.machine(),
            trace_scope='AVBufferPool CREATE/GET/final RETURN/CLOSE',
            scheduling='native pthreads and atomics; pool operations serialized by recorder')
(out / 'workload.json').write_text(json.dumps(info, indent=2) + '\n')
print(f'FFWORKLOAD frames={frames} output=identical trace={out / "original.bin"}')
PY
