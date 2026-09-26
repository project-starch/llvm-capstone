#!/usr/bin/env bash
# Run an mruby program image (mruby, mrbtest) as a domain under QEMU.
#
#   bash run-mruby-domain.sh <image> <work dir> <seconds> [<file>...] -- [<argv[1]>...]
#
# The files are staged in the share the runner mounts at /mnt/host, under
# /mnt/host/files/<basename>; the arguments go to /mnt/host/dom-args, which
# toolchain/domain_entry.c reads as argv[1..]. Prints the domain's output (from
# MRBD-RUN-BEGIN to MRBD-RUN-END) and the host's LT-RESULT line or the monitor's
# halt line; exit 0 when the program exited 0.
#
# Environment: CAPSTONE_LLVM_BUILD_DIR, CAPSTONE_BUILDROOT_DIR (a snapshot whose
# kernel module has CMA-backed domain blocks; default /tmp/capstone/br-snap),
# CAPSTONE_QEMU_BINARY, RUNTIME_REPO (the tree whose host helper, libc_test_host,
# serves the domain), MRBD_CMA_MB (default 640: an image with the 64 MiB arena
# needs a block beyond the buddy allocator, taken from CMA), MRBD_ENV (the
# domain's environment, NAME=value words; default HOME=/mnt/host). CAPSTONE_* switches
# of the QEMU (CAPSTONE_TAGWATCH, CAPSTONE_MOVC_NULL_SCALAR, ...) pass through.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMG=${1:?usage: run-mruby-domain.sh <image> <work dir> <seconds> [files...] -- [args...]}
WORK=${2:?} LIMIT=${3:?}; shift 3
export CAPSTONE_BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR:-/tmp/capstone/br-snap}
source "$SCRIPT_DIR/../../../tests/capstone-test-env.sh" >/dev/null
RT=${RUNTIME_REPO:-$CAPSTONE_REPO_ROOT}
SHARE=$WORK/share
rm -rf "$WORK"; mkdir -p "$SHARE/bin" "$SHARE/files"
while [[ $# -gt 0 && $1 != -- ]]; do cp "$1" "$SHARE/files/"; shift; done
[[ ${1:-} == -- ]] && shift
printf '%s\n' "$@" > "$SHARE/dom-args"
# The environment: MRBD_ENV, NAME=value words; HOME by default, as a shell would
# have it (File.expand_path("~") reads it).
printf '%s\n' ${MRBD_ENV:-HOME=/mnt/host} > "$SHARE/dom-env"
BR=$CAPSTONE_BUILDROOT_DIR/build
MUSL_PORT=$RT/capstone/ports/musl-capstone
MODSRC=$RT/capstone/caplifive-buildroot/package/modcapstone
"$CAPSTONE_LLVM_BIN/llvm-objcopy" --strip-debug "$IMG" "$SHARE/bin/mruby.dom" || exit 2
"$BR/host/bin/riscv64-buildroot-linux-gnu-gcc" -O2 -I"$MUSL_PORT/libc-test" -I"$MUSL_PORT/runtime" \
  -I"$MODSRC/userspace/lib" -I"$RT/capstone/tests/runtime-qemu/hostcall-stdout-probe" \
  -I"$RT/capstone/tests/runtime-qemu" -o "$SHARE/lt.user" \
  "$MUSL_PORT/libc-test/libc_test_host.c" "$MODSRC/userspace/lib/libcapstone.c" || exit 2
cat > "$SHARE/dom-run.sh" <<RUN
cp /mnt/host/bin/mruby.dom /tmp/mruby.dom && echo MRBD-IMAGE-COPIED
echo MRBD-RUN-BEGIN
/tmp/lt.user /tmp/mruby.dom $LIMIT 2>&1
echo MRBD-RUN-END rc=\$?
RUN
GUEST="echo __BOOT_OK__; cp /mnt/host/lt.user /tmp/lt.user && chmod 0755 /tmp/lt.user; sh /mnt/host/dom-run.sh 2>&1 | sed 's/#[ ]/#_/g'; echo __ALL_DONE__"
python3 -c "import pexpect" 2>/dev/null \
  || { echo "run-domain-smoke.py needs pexpect in this python3 (activate the venv)" >&2; exit 2; }
# gp is re-fabricated LINEAR at every cjalr without CAPSTONE_GP_NONLIN=1 (the CPython runner's note).
export CAPSTONE_GP_NONLIN=${CAPSTONE_GP_NONLIN:-1}
CAPSTONE_QEMU_LOGIN_TIMEOUT=${CAPSTONE_QEMU_LOGIN_TIMEOUT:-240} CAPSTONE_GUEST_COMMAND_TIMEOUT=$((LIMIT + 330)) \
capstone_with_qemu_lock python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE" --log-file "$WORK/qemu.log" --timeout-multiplier 8 --kernel-arg cma=${MRBD_CMA_MB:-640}M \
  --guest-command "$GUEST" --success-marker __BOOT_OK__ --success-marker __ALL_DONE__ > "$WORK/runner.out" 2>&1
sed -n '/MRBD-RUN-BEGIN/,/MRBD-RUN-END/p' "$WORK/qemu.log" | grep -av 'remote fence' | grep -av 'echo MRBD'
halt=$(grep -a -m1 'domain halted' "$WORK/qemu.log" || true)
result=$(grep -a -m1 '^LT-RESULT' "$WORK/qemu.log" || true)
[[ -n $halt ]] && echo "$halt"
[[ -n $result ]] || { echo "no LT-RESULT line: the domain did not finish (see $WORK/qemu.log)" >&2; exit 1; }
echo "$result"
[[ $result == *" PASS"* ]]
