#!/usr/bin/env bash
# Record both pool APIs and keep the decoder output oracle beside the trace.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORK=${FFPOOL_WORK:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/ffmpeg-buffer-pool}
OUT=${1:?usage: record-workload.sh RESULT-DIR [duration-seconds] [size]}
DURATION=${2:-300}
SIZE=${3:-1280x720}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
for name in input.mkv stock.framemd5 traced.framemd5 recorded.bin commands.bin workload.json; do
    [[ ! -e "$OUT/$name" ]] || { echo "refusing to overwrite $OUT/$name" >&2; exit 2; }
done
STOCK="$WORK/workload-stock/ffmpeg"
TRACED="$WORK/combined-workload/ffmpeg"
COMMON=(-nostdin -hide_banner -loglevel warning -threads 1 -filter_threads 1 -filter_complex_threads 1)
"$STOCK" "${COMMON[@]}" -f lavfi \
    -i "testsrc2=size=$SIZE:rate=30:duration=$DURATION" -c:v mpeg4 -q:v 3 -y "$OUT/input.mkv" \
    > "$OUT/generate.stdout" 2> "$OUT/generate.stderr"
"$STOCK" "${COMMON[@]}" -i "$OUT/input.mkv" -f framemd5 "$OUT/stock.framemd5" \
    > "$OUT/stock.stdout" 2> "$OUT/stock.stderr"
FFPOOL_TRACE="$OUT/recorded.bin" "$TRACED" "${COMMON[@]}" \
    -i "$OUT/input.mkv" -f framemd5 "$OUT/traced.framemd5" \
    > "$OUT/traced.stdout" 2> "$OUT/traced.stderr"
cmp "$OUT/stock.framemd5" "$OUT/traced.framemd5"
python3 "$HERE/../../analysis/trace-tools.py" commands "$OUT/recorded.bin" "$OUT/commands.bin"
python3 - "$OUT" "$DURATION" "$SIZE" "$STOCK" "$TRACED" <<'PY'
import hashlib, json, pathlib, sys
out = pathlib.Path(sys.argv[1])
frames = sum(not line.startswith('#') for line in (out / 'stock.framemd5').read_text().splitlines())
if frames != int(sys.argv[2]) * 30:
    raise SystemExit('frame count mismatch')
paths = [out / name for name in ('input.mkv', 'stock.framemd5', 'traced.framemd5', 'recorded.bin', 'commands.bin')]
paths += [pathlib.Path(sys.argv[4]), pathlib.Path(sys.argv[5])]
hashes = {str(p): hashlib.file_digest(p.open('rb'), 'sha256').hexdigest() for p in paths}
info = dict(ffmpeg='9.0.1', duration_seconds=int(sys.argv[2]), dimensions=sys.argv[3],
            frames=frames, fps=30, output_identical=True, sha256=hashes,
            trace_scope='AVBufferPool and AVRefStructPool with nested allocator callback effects',
            scheduling='native pthreads and atomics; pool operations serialized by recorder')
(out / 'workload.json').write_text(json.dumps(info, indent=2) + '\n')
print(f'FF2_WORKLOAD frames={frames} output=identical')
PY
