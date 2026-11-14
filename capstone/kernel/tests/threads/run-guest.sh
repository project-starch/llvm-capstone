#!/usr/bin/env bash
# Use the first-program initramfs recipe with the prebuilt libc thread probes.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../tests/capstone-test-env.sh"
q=${1:?QEMU} fw=${2:?firmware} image=${3:?kernel Image} dom=${4:?smoke directory}
out=${5:-$CAPSTONE_TMP_ROOT/thread-guest}
mkdir -p "$out"
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
mkdir "$work/root"
(cd "$work/root" && zcat "${TRAPTEST_BASE_INITRAMFS:?busybox initramfs}" | cpio -id 2>/dev/null)
for name in threads-smoke threads-disabled futex-stale; do
    install -m 0755 "$dom/$name.dom" "$work/root/"
done
install -m 0755 "$here/init" "$work/root/init"
(cd "$work/root" && find . | cpio -o -H newc 2>/dev/null | gzip -9) > "$work/initramfs.cpio.gz"
python3 "$here/../trap-route/boot.py" "$q" "$fw" "$image" "$work/initramfs.cpio.gz" "$out/threads.log" off
grep 'TRAPTEST:' "$out/threads.log"
for verdict in 'threads-smoke status 0' 'threads-disabled status 1' 'futex-stale status 0'; do
    grep -q "TRAPTEST: $verdict" "$out/threads.log"
done
if grep -q 'Kernel panic' "$out/threads.log"; then
    exit 1
fi
echo 'PASS: two threads join, disabled-thread control fails, stale futex accesses return EFAULT'
