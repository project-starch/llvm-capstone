#!/usr/bin/env bash
# Run the linked CPython interpreter in a domain under QEMU, on one script.
#
#   run-cpython-domain.sh <script.py> [seconds]
#
# The image is $CPY_ROOT/link/python.dom from link-cpython-capstone.py; the
# script becomes /mnt/host/main.py, which toolchain/domain_entry.c runs with
# PYTHONHOME=/mnt/host/pyhome. Prints the host's LT-RESULT line and the
# domain's output; exit 0 when the interpreter exited 0.
#
# Three things are staged next to the image, each built here:
#   pyhome/lib/python313.zip  the stdlib, by make-stdlib-zip.py with the native
#                             3.13 that prepare-cpython-capstone.sh built
#   lt.user                   the host process, musl-capstone's libc_test_host.c
#                             (a generic musl-domain host: file service, stdout,
#                             exit status)
#   capstone-cma.ko           the kernel module with CMA-backed domain blocks. The
#                             image needs ~60 MiB (48 MiB of it the heap arena);
#                             the pinned module stops at 4 MiB. Built from
#                             $CPY_MODCAPSTONE_SRC, which must carry that change
#                             (caplifive-buildroot domain/1-cma-large-domains).
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../../tests/capstone-test-env.sh" >/dev/null

# CPY_QEMU_MONITOR=<path> adds a second QEMU monitor on that UNIX socket, so a
# stalled run can be asked where the vCPU is (`info registers`) from outside.
SCRIPT=${1:?usage: run-cpython-domain.sh <script.py> [seconds]}
SECONDS_LIMIT=${2:-900}
CPY_ROOT=${CPY_ROOT:-$CAPSTONE_TMP_ROOT/cpython-interpreter}
DOM=${CPY_DOM:-$CPY_ROOT/link/python.dom}
MODSRC=${CPY_MODCAPSTONE_SRC:-$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone}
BR=${CAPSTONE_BUILDROOT_DIR:?}/build
WORK=$CPY_ROOT/run
SHARE=$WORK/share
MUSL_PORT=$CAPSTONE_REPO_ROOT/capstone/ports/musl-capstone

[[ -f "$SCRIPT" ]] || { echo "no script $SCRIPT" >&2; exit 2; }
[[ -f "$DOM" ]] || { echo "no image $DOM; run link-cpython-capstone.py" >&2; exit 2; }
grep -q "beyond the buddy allocator" "$MODSRC/module/capstone.c" \
  || { echo "$MODSRC has no CMA-backed domain blocks; set CPY_MODCAPSTONE_SRC" >&2; exit 2; }

rm -rf "$WORK"; mkdir -p "$SHARE/pyhome/lib" "$WORK/mod"
# Only the loadable segment matters to the guest; the -g image is ~3x larger
# and every byte of it crosses the 9p share. Stripped here, kept whole for
# symbolizing a fault.
"$CAPSTONE_LLVM_BIN/llvm-objcopy" --strip-debug "$DOM" "$SHARE/python.dom"
cp "$SCRIPT" "$SHARE/main.py"

"$CPY_ROOT/build-python/python" "$SCRIPT_DIR/make-stdlib-zip.py" \
  "$CPY_ROOT/src/Python-3.13.7/Lib" "$SHARE/pyhome/lib/python313.zip"

"$BR/host/bin/riscv64-buildroot-linux-gnu-gcc" -O2 \
  -I"$MUSL_PORT/libc-test" -I"$MUSL_PORT/runtime" -I"$MODSRC/userspace/lib" \
  -I"$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/hostcall-stdout-probe" \
  -I"$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu" \
  -o "$SHARE/lt.user" "$MUSL_PORT/libc-test/libc_test_host.c" "$MODSRC/userspace/lib/libcapstone.c"

cp -r "$MODSRC/module" "$WORK/mod/module"; cp -r "$MODSRC/include" "$WORK/mod/include"
make -C "$WORK/mod/module" LINUX_DIR="$BR/build/linux-custom" KERNEL_ARCH=riscv \
  TARGET_CROSS="$BR/host/bin/riscv64-buildroot-linux-gnu-" PWD="$WORK/mod/module" \
  > "$WORK/module-build.log" 2>&1 \
  || { tail -20 "$WORK/module-build.log" >&2; echo "module build failed" >&2; exit 2; }
cp "$WORK/mod/module/capstone.ko" "$SHARE/capstone-cma.ko"
want=$(strings "$BR/target/lib/modules"/*/extra/capstone.ko 2>/dev/null | grep -m1 '^vermagic=' || true)
[[ -n "$want" ]] || want=$(strings "$BR/target/capstone.ko" 2>/dev/null | grep -m1 '^vermagic=' || true)
# grep -m1 can close the pipe before strings is done; under pipefail that
# SIGPIPE (141) aborted the whole run now and then, hence "|| true" here as
# above, and an empty result is an error of its own.
got=$(strings "$SHARE/capstone-cma.ko" | grep -m1 '^vermagic=' || true)
[[ -n "$got" ]] || { echo "no vermagic in the built module $SHARE/capstone-cma.ko" >&2; exit 2; }
[[ -z "$want" || "$want" == "$got" ]] \
  || { echo "module vermagic '$got' does not match the image's '$want'" >&2; exit 2; }

# CPY_MODULE_IN_ROOTFS=1: the rootfs's own /capstone.ko is already the CMA
# module, so the run does not unload and reload a module. Four of six guest
# stalls seen on 2026-09-23 happened at that step.
if [[ "${CPY_MODULE_IN_ROOTFS:-0}" == 1 ]]; then
  SWAP_MODULE="echo CPY-MODULE-ROOTFS;"
else
  SWAP_MODULE="rmmod capstone && insmod /mnt/host/capstone-cma.ko && echo CPY-MODULE-CMA;"
fi
# The image is copied into the guest's tmpfs first rather than loaded from the
# 9p share. The domain's output goes straight to the console: a domain that
# halts wedges the guest, and whatever it printed before is the evidence. It
# passes through sed because run-domain-smoke.py takes "# " for the shell
# prompt, and CPython's output has it ("import _imp # builtin" under -v): the
# runner then typed its exit-code probe into the running program and gave up.
# The same holds for the command itself, which the guest echoes: with a literal
# "hash, space" in it (as the first version of this sed had) the runner matched
# the echo, typed its probe before the domain had printed anything, and gave up
# after its exit-code wait. Hence "#[ ]", and the check below.
GUEST="echo __BOOT_OK__; cp /mnt/host/lt.user /tmp/lt.user && chmod 0755 /tmp/lt.user; \
cp /mnt/host/python.dom /tmp/python.dom && echo CPY-IMAGE-COPIED; \
$SWAP_MODULE \
echo CPY-RUN-BEGIN t=\$(date +%s); \
{ /tmp/lt.user /tmp/python.dom $SECONDS_LIMIT; echo CPY-RUN-END rc=\$? t=\$(date +%s); } 2>&1 | sed 's/#[ ]/#_/g'; \
dmesg | grep -iE 'Domain block|beyond the buddy|Failed to allocate' | tail -3; echo __ALL_DONE__"
if [[ $GUEST == *"# "* ]]; then
  echo "the guest command contains '# ', which run-domain-smoke.py takes for the prompt" >&2
  exit 2
fi

python3 -c "import pexpect" 2>/dev/null \
  || { echo "run-domain-smoke.py needs pexpect in this python3 (activate the venv)" >&2; exit 2; }
# QEMU re-fabricates gp from PCC at every cjalr, with PCC's type: LINEAR once the
# first call has returned. Every code capability derived from gp is then linear,
# and the compiler's `movc s5, a3; cjalr ra, 0(a3)` MOVES it and calls through
# null ("cs.cjalr requires capability in rs1"), which is how the first boot
# halted in Py_InitializeFromConfig. The entry glue delinearises gp on purpose;
# CAPSTONE_GP_NONLIN=1 keeps the re-fabrication from undoing that
# (capstone-qemu target/riscv/op_helper.c; the postgres runners set it too).
export CAPSTONE_GP_NONLIN=${CAPSTONE_GP_NONLIN:-1}
# A boot that ends with neither a result line nor a domain halt stalled in the
# guest (seen on 2026-09-23 with the guest kernel idling, at varying points) and
# says nothing about the interpreter: it is retried, up to CPY_ATTEMPTS times,
# and every attempt is reported.
for attempt in $(seq 1 "${CPY_ATTEMPTS:-3}"); do
  rc=0
  CAPSTONE_QEMU_LOGIN_TIMEOUT=${CAPSTONE_QEMU_LOGIN_TIMEOUT:-240} \
  CAPSTONE_GUEST_COMMAND_TIMEOUT=$((SECONDS_LIMIT + 300)) \
  flock -w 7200 "$HOME/.capstone-locks/qemu.lock" \
    python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
      --share-dir "$SHARE" --log-file "$WORK/qemu.log" --timeout-multiplier 8 \
      --kernel-arg cma=256M --guest-command "$GUEST" \
      ${CPY_QEMU_MONITOR:+--qemu-extra-arg=-monitor --qemu-extra-arg=unix:$CPY_QEMU_MONITOR,server,nowait} \
      --success-marker __BOOT_OK__ --success-marker __ALL_DONE__ > "$WORK/runner.out" 2>&1 || rc=$?
  echo "attempt $attempt: runner rc=$rc (log $WORK/qemu.log)"
  grep -aqE '^LT-RESULT|domain halted' "$WORK/qemu.log" && break
  echo "attempt $attempt: guest stalled before a result; last lines:"
  grep -avE 'remote fence' "$WORK/qemu.log" | tail -3 | sed 's/^/    /'
  cp "$WORK/qemu.log" "$WORK/qemu-stalled-$attempt.log"
done
# The domain's output sits between the markers; the host's verdict is LT-RESULT.
sed -n '/CPY-RUN-BEGIN/,/CPY-RUN-END/p' "$WORK/qemu.log" | grep -v 'remote fence extension' \
  | grep -v '^.*echo CPY-RUN' || true
grep -aE 'CPY-MODULE-CMA|CPY-MODULE-ROOTFS|Domain block|beyond the buddy|Failed to allocate' "$WORK/qemu.log" \
  | grep -v 'grep -iE' || true
result=$(grep -a -m1 '^LT-RESULT' "$WORK/qemu.log" || true)
[[ -n "$result" ]] || { echo "no LT-RESULT line: the domain did not finish (see $WORK/qemu.log)" >&2; exit 1; }
[[ "$result" == *" PASS"* ]]
