#!/usr/bin/env bash
# Boot the M-infra gate: one module per boot, each from a PRIVATE copy of the guest rootfs with
# /capstone.ko replaced (the shared rootfs.ext2 is never written). A live rmmod/insmod swap hung
# the guest (predictions.txt, attempt 2), which is why a boot carries exactly one module.
#   run.sh [OUT]        after build.sh [OUT]
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../../../tests/capstone-test-env.sh"
OUT=${1:-$CAPSTONE_TMP_ROOT/cma-domain-block}; S=$OUT/share
IMG=${CAPSTONE_BUILDROOT_DIR:?}/build/images
cp "$HERE"/gate?.sh "$S/"
rootfs() {  # module -> $OUT/br-<module>
  local D=$OUT/br-$1/build/images; mkdir -p "$D"
  ln -sf "$IMG/fw_jump.elf" "$D/fw_jump.elf"; ln -sf "$IMG/Image" "$D/Image"
  cp --sparse=always "$IMG/rootfs.ext2" "$D/rootfs.ext2"
  debugfs -w -R "rm /capstone.ko" "$D/rootfs.ext2" > /dev/null 2>&1
  debugfs -w -R "write $S/$1.ko /capstone.ko" "$D/rootfs.ext2" > /dev/null 2>&1
  debugfs -w -R "sif /capstone.ko mode 0100755" "$D/rootfs.ext2" > /dev/null 2>&1
  debugfs -R "dump /capstone.ko $OUT/br-$1.check" "$D/rootfs.ext2" > /dev/null 2>&1
  cmp -s "$S/$1.ko" "$OUT/br-$1.check" || { echo "rootfs $1: /capstone.ko did not read back" >&2; return 1; }
}
boot() {  # name module gate marker [kernel-arg]
  local extra=(); [ -n "${5:-}" ] && extra=(--kernel-arg "$5")
  CAPSTONE_GUEST_COMMAND_TIMEOUT=1500 capstone_with_qemu_lock python3 \
    "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" --share-dir "$S" \
    --buildroot-dir "$OUT/br-$2" --qemu-binary "$CAPSTONE_QEMU_BINARY" "${extra[@]}" \
    --guest-command "sh /mnt/host/$3" --success-marker "$4" --log-file "$OUT/$1.log" > "$OUT/$1.runner" 2>&1
  echo "$1: runner rc=$?"
  tr -d '\r' < "$OUT/$1.log" | grep -aE "^(MODULE-MD5|RUN|EXIT|DMESG|GATE)|retval =|domain halted|Cap mem access" \
    > "$OUT/$1.lines"
}
rootfs cma && rootfs fix || exit 1
boot bootB cma gateB.sh GATEB-END cma=1G     # edge64 is predicted to FAULT: last in its boot
boot bootE fix gateE.sh GATEE-END cma=1G
boot bootD fix gateD.sh GATED-END
