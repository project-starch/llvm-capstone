#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../tests/capstone-test-env.sh"
WORK=${FFPOOL_WORK:-$CAPSTONE_TMP_ROOT/ffmpeg-buffer-pool}
INPUT=${1:?usage: run-replay-qemu.sh TRACE RESULT-DIR}
RESULT=${2:?missing result directory}
EXPECTED_STATUS=${3:-0}
[[ "$EXPECTED_STATUS" =~ ^[0-9]+$ ]] || { echo "invalid expected status" >&2; exit 2; }
PYTHON=${PYTHON:-python3}
"$PYTHON" -c 'import pexpect'
mkdir -p "$RESULT"
RESULT=$(cd "$RESULT" && pwd)
RUN=$(mktemp -d "$RESULT/qemu.XXXXXX")
SHARE="$RUN/share"
mkdir -p "$SHARE"
cp "$INPUT" "$SHARE/trace.bin"
cp "$WORK/capstone/replay.dom" "$SHARE/replay.dom"
cp "$WORK/capstone/replay-host.user" "$SHARE/replay-host.user"
printf 'EXPECTED_STATUS=%s\n' "$EXPECTED_STATUS" > "$SHARE/expected.sh"
cat > "$SHARE/run.sh" <<'EOF'
#!/bin/sh
set -e
cp /mnt/host/replay.dom /tmp/ffreplay.dom
cp /mnt/host/replay-host.user /tmp/ffreplay-host
cp /mnt/host/trace.bin /tmp/ffreplay-trace.bin
. /mnt/host/expected.sh
status=0
/tmp/ffreplay-host /tmp/ffreplay.dom /tmp/ffreplay-trace.bin /tmp/ffreplay-result.bin > /tmp/ffreplay.out || status=$?
cat /tmp/ffreplay.out
if [ "$EXPECTED_STATUS" = 0 ]; then
    test "$status" = 0
else
    test "$status" = 1
    grep -q "^FFREPLAY return=$EXPECTED_STATUS status=$EXPECTED_STATUS events=" /tmp/ffreplay.out
fi
cp /tmp/ffreplay-result.bin /mnt/host/capstone.bin
echo FFREPLAY_DONE
EOF
sha256sum "$SHARE/trace.bin" "$SHARE/replay.dom" "$SHARE/replay-host.user" \
    > "$RUN/inputs.sha256"
echo "run records: $RUN"
flock -x -w 45 "$CAPSTONE_QEMU_LOCK" \
    "$PYTHON" "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
    --share-dir "$SHARE" --log-file "$RUN/serial.log" \
    --kernel-arg cma=128M --timeout-multiplier 4 \
    --guest-command 'sh /mnt/host/run.sh' --success-marker FFREPLAY_DONE
cp "$SHARE/capstone.bin" "$RESULT/capstone.bin"
printf '%s\n' "$RUN" > "$RESULT/qemu-run-path.txt"
