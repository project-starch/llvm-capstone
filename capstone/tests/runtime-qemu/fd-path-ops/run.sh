#!/usr/bin/env bash
# dup, FD_CLOEXEC, lstat, symlink, chmod, flock and /dev/tty in a musl domain
# (fd_path_ops.c says what each check is), under QEMU.
#
#   bash run.sh        exit 0 only if the domain built with this tree's runtime
#                      prints "FD-OPS-DONE failures=0" with a PASS status line,
#                      AND the control fails its cloexec, dup, symlink and chmod
#
# The control is the same test linked against the hostcall.c of 162ef1071fba
# (the runtime before these calls; pinned, so it stays a control after this
# lands), which keeps no FD_CLOEXEC and serves none of dup, symlinkat or
# fchmodat. It differs from the test in that one file only -- the host serves
# the new opcodes in both arms -- so a run that passes both ways is not being
# served by the new code at all, and the run says so. The test's exit report
# lists one unserved syscall, dup of stdout, which the last check asks for.
#
# Needs: CAPSTONE_LLVM_BUILD_DIR (clang, ld.lld, llvm-ar), CAPSTONE_BUILDROOT_DIR
# and CAPSTONE_QEMU_BINARY, and a python with pexpect (PYTHON=..., default
# python3). Takes the shared QEMU lock for the run.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../capstone-test-env.sh" >/dev/null
REPO=$CAPSTONE_REPO_ROOT
OUT=${OUT:-$CAPSTONE_TMP_ROOT/fd-path-ops}
PYTHON=${PYTHON:-python3}
CONTROL=162ef1071fba
mkdir -p "$OUT/obj" "$OUT/share"

# musl and its archive, private to this test (prepare rewrites arch/ in place).
export MUSL_CACHE_ROOT=$OUT/musl-src
mkdir -p "$MUSL_CACHE_ROOT"
[[ -f "$MUSL_CACHE_ROOT/musl-1.2.5.tar.gz" || ! -f "$CAPSTONE_TMP_ROOT/musl-src/musl-1.2.5.tar.gz" ]] \
  || cp "$CAPSTONE_TMP_ROOT/musl-src/musl-1.2.5.tar.gz" "$MUSL_CACHE_ROOT/"
MUSL=$(bash "$REPO/capstone/ports/musl-capstone/prepare-musl-capstone.sh" | tail -1)
OUT_DIR=$OUT/musl-build bash "$REPO/capstone/ports/musl-capstone/build-musl-capstone.sh" >/dev/null
ARCHIVE=$OUT/musl-build/libc-capstone.a

# The runtime, as musl-capstone/libc-test/build-libc-test.sh builds it.
MRT=$REPO/capstone/ports/musl-capstone/runtime
INC=(-nostdinc -isystem "$MUSL/arch/capstone64" -isystem "$MUSL/arch/generic"
     -isystem "$MUSL/obj/include" -isystem "$MUSL/include")
CF=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
    -Xclang -target-feature -Xclang +a -ffreestanding -fno-builtin -fno-jump-tables
    -ffunction-sections -fdata-sections -O1 -w -Wno-int-conversion "${INC[@]}")
RF=("${CF[@]}" -std=c99 -D_XOPEN_SOURCE=700
    -I"$MUSL/src/include" -I"$MUSL/src/internal" -I"$MUSL/obj/src/internal")
O=$OUT/obj
for s in start-musl set_thread_area setjmp; do
  "$CAPSTONE_CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
    -ffreestanding -O0 -c "$MRT/$s.S" -o "$O/$s.o"
done
for f in hostcall tls level0; do
  "$CAPSTONE_CLANG" "${RF[@]}" -c "$MRT/$f.c" -o "$O/$f.o"
done
# The control's one difference: $CONTROL's hostcall.c, without the new calls. Its
# one relative include is made absolute, since the copy lives in $O.
git -C "$REPO" show "$CONTROL:capstone/ports/musl-capstone/runtime/hostcall.c" \
  | sed "s#\"../../../tests/runtime-qemu/#\"$REPO/capstone/tests/runtime-qemu/#" > "$O/hostcall-control.c"
grep -q "\"$REPO/capstone/tests/runtime-qemu/hostcall-stdout-probe/" "$O/hostcall-control.c" \
  || { echo "control copy: the include rewrite did not apply" >&2; exit 2; }
if grep -qE 'SYS_dup|SYS_fchmodat|SYS_symlinkat|hc_cloexec' "$O/hostcall-control.c"; then
  echo "$CONTROL's hostcall.c serves what this tests; this control no longer controls anything" >&2
  exit 2
fi
"$CAPSTONE_CLANG" "${RF[@]}" -c "$O/hostcall-control.c" -o "$O/hostcall-control.o"
# The libc overrides, from the one list every musl domain links (runtime/libc_overrides.sh).
source "$MRT/libc_overrides.sh"
build_musl_overrides "$CAPSTONE_CLANG" "$O" "$MUSL" "${RF[@]}"
CLANG=$CAPSTONE_CLANG OBJ_DIR=$O COMPILER_RT=$REPO/compiler-rt/lib/builtins
COMMON_FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
              -ffreestanding -fno-builtin -ffunction-sections -fdata-sections -O1 -w)
source "$REPO/capstone/benchmarks/beebs/build-beebs-softfloat-common.sh"
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -c "$HERE/entry.c" -o "$O/entry.o"
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -O2 -c "$HERE/fd_path_ops.c" -o "$O/fd_path_ops.o"
COMMON=("$O/start-musl.o" "$O/tls.o" "$O/set_thread_area.o" "$O/setjmp.o"
        "${MUSL_OVERRIDE_OBJS[@]}"
        "$O/level0.o" "${softfloat_objs[@]}" "$O/entry.o" "$O/fd_path_ops.o")
rm -f "$OUT/share"/*.dom "$OUT/share"/fdops-*
for arm in test control; do
  hc=$O/hostcall.o; [[ $arm == control ]] && hc=$O/hostcall-control.o
  "$CAPSTONE_LD_LLD" --gc-sections -T "$REPO/capstone/my_first_domain/link.ld" \
    -o "$OUT/share/fo-$arm.dom" "${COMMON[@]}" "$hc" "$ARCHIVE"
done

# The host: libc-test's, built from this tree, so it serves the new opcodes in both arms.
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_DIR=$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib
"$GUEST_CC" -O2 -I"$REPO/capstone/ports/musl-capstone/libc-test" -I"$MRT" -I"$LIBCAPSTONE_DIR" \
  -I"$REPO/capstone/tests/runtime-qemu/hostcall-stdout-probe" -I"$REPO/capstone/tests/runtime-qemu" \
  -o "$OUT/share/lt.user" "$REPO/capstone/ports/musl-capstone/libc-test/libc_test_host.c" \
  "$LIBCAPSTONE_DIR/libcapstone.c"

LOG=$OUT/run-$(date +%Y%m%d-%H%M%S).log
GUEST='echo __BOOT_OK__; cp /mnt/host/lt.user /tmp/lt.user && chmod 0755 /tmp/lt.user; for d in fo-test fo-control; do echo RUN-BEGIN $d; /tmp/lt.user /mnt/host/$d.dom 60; echo RUN-END $d rc=$?; done; echo __ALL_DONE__'
set +e
CAPSTONE_GP_NONLIN=${CAPSTONE_GP_NONLIN:-1} \
capstone_with_qemu_lock "$PYTHON" "$REPO/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$OUT/share" --log-file "$LOG" --timeout-multiplier 8 \
  --guest-command "$GUEST" --success-marker __BOOT_OK__ --success-marker __ALL_DONE__ >/dev/null 2>&1
set -e

arm_log() { sed -n "/RUN-BEGIN fo-$1/,/RUN-END fo-$1/p" "$LOG"; }
verdict=0
if grep -aq "FD-OPS-DONE failures=0" <(arm_log test) &&
   grep -aq "LT-RESULT fo-test.dom status=0 .*PASS" "$LOG"; then
  echo "  test:    PASS"
else
  echo "  test:    FAIL"; verdict=1
fi
arm_log test | grep -aE "^FD-OPS|UNSERVED|halted|LT-RESULT" | sed 's/^/    /'
control_fails=0
for c in cloexec-open dup symlink chmod; do
  grep -aq "FD-OPS FAIL $c " <(arm_log control) && control_fails=$((control_fails + 1))
done
if [[ $control_fails == 4 ]] && grep -aq "FD-OPS-DONE" <(arm_log control); then
  echo "  control: FAILS cloexec-open, dup, symlink and chmod, as it must without the new code"
else
  echo "  control: failed $control_fails of cloexec-open, dup, symlink, chmod -- the test cannot tell the new code from its absence"; verdict=1
fi
arm_log control | grep -aE "^FD-OPS|UNSERVED|halted|LT-RESULT" | sed 's/^/    /'
echo "  compiler $("$CAPSTONE_CLANG" --version | grep -oE '[0-9a-f]{40}' | cut -c1-12), log $LOG"
exit $verdict
