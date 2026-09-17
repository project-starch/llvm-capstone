#!/usr/bin/env bash
# Given a complete recording, regenerate commands without recorded outcomes,
# run both replay arms, reject an invalid final RETURN, then compare and plot.
# Requires native/capstone build-replay.sh outputs and the platform environment.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORK=${FFPOOL_WORK:-/tmp/capstone/ffmpeg-buffer-pool}
RESULT=${1:?usage: check-replay.sh RESULT-DIR}
PYTHON=${PYTHON:-python3}
"$PYTHON" "$HERE/trace-tools.py" prepare "$RESULT/original.bin" "$RESULT"
"$WORK/native/replay" "$RESULT/commands.bin" "$RESULT/native.bin"
status=0
"$WORK/native/replay" "$RESULT/invalid-return.bin" "$RESULT/native-invalid-return.bin" \
    > "$RESULT/native-invalid-return.log" || status=$?
[[ "$status" == 1 ]]
grep -q '^FFREPLAY status=18 events=' "$RESULT/native-invalid-return.log"
bash "$HERE/run-replay-qemu.sh" "$RESULT/commands.bin" "$RESULT"
bash "$HERE/run-replay-qemu.sh" "$RESULT/invalid-return.bin" "$RESULT/negative-control" 18
"$PYTHON" "$HERE/trace-tools.py" compare "$RESULT/original.bin" \
    "$RESULT/native.bin" "$RESULT/capstone.bin" "$RESULT"
"$PYTHON" "$HERE/plot-replay.py" "$RESULT"
