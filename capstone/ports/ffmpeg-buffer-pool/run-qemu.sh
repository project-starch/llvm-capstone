#!/usr/bin/env bash
# Run the compiled unprotected pilot after a known-result domain control.
# Set CAPSTONE_LLVM_BUILD_DIR, CAPSTONE_BUILDROOT_DIR, CAPSTONE_QEMU_BINARY
# to the installed toolchain and initialized platform checkout before calling.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../tests/capstone-test-env.sh"
WORK=${FFPOOL_WORK:-$CAPSTONE_TMP_ROOT/ffmpeg-buffer-pool}
PYTHON=${PYTHON:-python3}
"$PYTHON" -c 'import pexpect'
DOM="$WORK/capstone/ffpool.dom"
[[ -f "$DOM" ]] || { echo "run build.sh capstone first" >&2; exit 2; }
RUN=$(mktemp -d "$WORK/qemu-run.XXXXXX")
SHARE="$RUN/share"
mkdir -p "$SHARE"
cp "$DOM" "$SHARE/ffpool.dom"
RUNTIME="$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu"
export CAPSTONE_TEST_SRC="$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/capstone-test.c"
export LIBCAPSTONE_SRC="$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib/libcapstone.c"
export CAPSTONE_INCLUDE_DIR="$CAPSTONE_BUILDROOT_DIR/package/modcapstone/include"
bash "$RUNTIME/build-capstone-test-user.sh" "$SHARE/capstone-test.user"
bash "$RUNTIME/build-domain.sh" "$RUNTIME/domains/write_42.c" "$SHARE/control.dom"
cat > "$SHARE/pilot.sh" <<'EOF'
#!/bin/sh
set -e
echo FFPOOL_COPY
cp /mnt/host/capstone-test.user /tmp/ffpool-loader
cp /mnt/host/control.dom /tmp/ffpool-control.dom
cp /mnt/host/ffpool.dom /tmp/ffpool.dom
echo FFPOOL_CONTROL
/tmp/ffpool-loader /tmp/ffpool-control.dom > /tmp/ffpool-control.out
cat /tmp/ffpool-control.out
grep -qx 'Called dom (1-th time) retval = 42' /tmp/ffpool-control.out
echo FFPOOL_PROBE
/tmp/ffpool-loader /tmp/ffpool.dom > /tmp/ffpool-probe.out
cat /tmp/ffpool-probe.out
grep -qx 'Called dom (1-th time) retval = 42042' /tmp/ffpool-probe.out
echo FFPOOL_DONE
EOF
sha256sum "$SHARE/ffpool.dom" "$SHARE/control.dom" "$SHARE/capstone-test.user" > "$RUN/images.sha256"
echo "run records: $RUN"
# Use the existing shared lock. The runner uses a snapshot of the rootfs.
flock -x -w 45 "$CAPSTONE_QEMU_LOCK" \
    "$PYTHON" "$RUNTIME/run-domain-smoke.py" \
    --share-dir "$SHARE" --log-file "$RUN/serial.log" \
    --timeout-multiplier 2 \
    --guest-command 'sh /mnt/host/pilot.sh' \
    --success-marker 'FFPOOL_DONE'
