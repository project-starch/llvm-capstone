#!/bin/bash
# Run the R1 harness image on capstone-qemu through the SQLite host's --speedtest1 path, one
# invocation = one (arm, series, pattern). The emulator gives the logic an oracle (survivors, the
# region back linear, node counts); its cycle figures are icount, not timing.
#   R1_DOM=<image> R1_HOST=<sqlite_host(_rr).user> OUT=<dir> bash run-r1-qemu.sh "<args>" [arena bytes]
set -u
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
source "$REPO_ROOT/capstone/tests/capstone-test-env.sh" >/dev/null 2>&1
ARGS=${1:?"the harness arguments, e.g. '--arm S --series nodes --pattern shared --reps 5'"}
# the host refuses a --speedtest1 line without --testset (sqlite_host.c, a precondition for the SQLite
# workload); the harness ignores the token and its value
ARGS="--testset r1 $ARGS"
ARENA=${2:-8388608}
: "${R1_DOM:?}" "${R1_HOST:?}" "${OUT:?}"
mkdir -p "$OUT/share"; cp -f "$R1_HOST" "$OUT/share/h.user"; cp -f "$R1_DOM" "$OUT/share/d.dom"
GC="cp /mnt/host/h.user /tmp/h && chmod 0755 /tmp/h; /tmp/h /mnt/host/d.dom --speedtest1 --arena $ARENA '$ARGS'; echo R1_RC=\$?; echo R1_END"
flock -w 7200 "$CAPSTONE_QEMU_LOCK" python3 "$REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$OUT/share" --log-file "$OUT/boot.log" --timeout-multiplier 6 \
  --qemu-extra-arg=-icount --qemu-extra-arg="shift=0,sleep=off" \
  --qemu-extra-arg=-append --qemu-extra-arg="root=/dev/vda ro cma=256M" \
  --guest-command "$GC" --success-marker R1_END > "$OUT/smoke.out" 2>&1; rc=$?
echo "smoke rc=$rc  image $(sha256sum "$R1_DOM" | cut -c1-16)  host $(sha256sum "$R1_HOST" | cut -c1-16)"
grep -aoE 'R1 [^\r\n]*|SQ: speedtest1-ran=[0-9]+|SQ: [a-z -]*(fault|abort|did not run|trap)[^\r\n]{0,80}|R1_RC=[0-9]+|domain halted[^\r\n]{0,120}|cause = [0-9]+, pc = 0x[0-9a-f]+[^\r\n]{0,60}' "$OUT/boot.log" | sed 's/^/  /'
