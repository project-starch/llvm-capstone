#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../tests/capstone-test-env.sh"
WORK=${FFPOOL_WORK:-$CAPSTONE_TMP_ROOT/ffmpeg-buffer-pool}
INPUT=${1:?usage: run-qemu.sh TRACE RESULT-DIR bounds|backing|sublet [EXPECTED-STATUS]}
RESULT=${2:?missing result directory}
PROTECTION=${3:?missing protection: bounds, backing or sublet}
case "$PROTECTION" in
    bounds|0) MODE=0 ;;
    backing|1) MODE=1 ;;
    sublet|2) MODE=2 ;;
    *) echo "expected bounds, backing or sublet" >&2; exit 2 ;;
esac
EXPECTED_STATUS=${4:-0}
[[ "$MODE" =~ ^[012]$ && "$EXPECTED_STATUS" =~ ^[0-9]+$ ]] || exit 2
PYTHON=${PYTHON:-python3}
"$PYTHON" -c 'import pexpect'
mkdir -p "$RESULT"
RESULT=$(cd "$RESULT" && pwd)
RUN=$(mktemp -d "$RESULT/qemu.XXXXXX")
SHARE="$RUN/share"
mkdir -p "$SHARE"
cp "$INPUT" "$SHARE/trace.bin"
cp "$WORK/combined-port-capstone/replay.dom" "$SHARE/replay.dom"
cp "$WORK/combined-port-capstone/host.user" "$SHARE/host.user"
printf 'EXPECTED_STATUS=%s\nMODE=%s\n' "$EXPECTED_STATUS" "$MODE" > "$SHARE/expected.sh"
cat > "$SHARE/run.sh" <<'EOF'
#!/bin/sh
set -e
cp /mnt/host/replay.dom /tmp/ff2.dom
cp /mnt/host/host.user /tmp/ff2-host
cp /mnt/host/trace.bin /tmp/ff2-trace.bin
. /mnt/host/expected.sh
status=0
/tmp/ff2-host /tmp/ff2.dom /tmp/ff2-trace.bin /tmp/ff2-result.bin "$MODE" > /tmp/ff2.out || status=$?
cat /tmp/ff2.out
cp /tmp/ff2-result.bin /mnt/host/capstone.bin
if [ "$EXPECTED_STATUS" = 0 ]; then
    test "$status" = 0
else
    test "$status" = 1
    grep -q "^FF2 return=$EXPECTED_STATUS status=$EXPECTED_STATUS events=" /tmp/ff2.out
fi
echo FF2_DONE
EOF
sha256sum "$SHARE/trace.bin" "$SHARE/replay.dom" "$SHARE/host.user" > "$RUN/inputs.sha256"
echo "run records: $RUN"
printf '%s\n' "CAPSTONE_REV_NODES=${CAPSTONE_REV_NODES:-65536}" > "$RUN/node-budget.txt"
flock -x -w 45 "$CAPSTONE_QEMU_LOCK" \
    "$PYTHON" "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
    --share-dir "$SHARE" --log-file "$RUN/serial.log" \
    --kernel-arg cma=512M --timeout-multiplier 12 \
    --guest-command 'sh /mnt/host/run.sh' --success-marker FF2_DONE
cp "$SHARE/capstone.bin" "$RESULT/capstone.bin"
printf '%s\n' "$RUN" > "$RESULT/qemu-run-path.txt"
