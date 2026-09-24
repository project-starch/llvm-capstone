#!/usr/bin/env bash
# A musl domain that RETURNS from its entry, never calling exit(), under QEMU.
#
#   bash run.sh        exit 0 only if, with this tree's runtime, all three buffered
#                      lines and the atexit handler's line reach the host and the
#                      status is the returned 5, AND the control loses the buffered
#                      lines
#
# The control is the same program linked against the hostcall.c of 40eefa09420c
# (dev before this; pinned, so it stays a control after this lands), whose
# domain_main returns the status straight to the host: musl's stdout is fully
# buffered after its first flush, so only the first line gets out, and no atexit
# handler runs.
#
# Needs: CAPSTONE_LLVM_BUILD_DIR (clang, ld.lld, llvm-ar), CAPSTONE_BUILDROOT_DIR
# and CAPSTONE_QEMU_BINARY, and a python with pexpect (PYTHON=..., default
# python3). Takes the shared QEMU lock for the run.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../capstone-test-env.sh" >/dev/null
REPO=$CAPSTONE_REPO_ROOT
OUT=${OUT:-$CAPSTONE_TMP_ROOT/return-flush}
PYTHON=${PYTHON:-python3}
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
# The libc overrides, from the one list every musl domain links (runtime/libc_overrides.sh).
source "$MRT/libc_overrides.sh"
build_musl_overrides "$CAPSTONE_CLANG" "$O" "$MUSL" "${RF[@]}"
CLANG=$CAPSTONE_CLANG OBJ_DIR=$O COMPILER_RT=$REPO/compiler-rt/lib/builtins
COMMON_FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
              -ffreestanding -fno-builtin -ffunction-sections -fdata-sections -O1 -w)
source "$REPO/capstone/benchmarks/beebs/build-beebs-softfloat-common.sh"
RUNTIME=("$O/start-musl.o" "$O/hostcall.o" "$O/tls.o" "$O/set_thread_area.o" "$O/setjmp.o"
         "${MUSL_OVERRIDE_OBJS[@]}"
         "$O/level0.o" "${softfloat_objs[@]}")

# The control's one difference: 40eefa09420c's hostcall.c. Its one relative include is
# made absolute, since the copy lives in $O.
git -C "$REPO" show 40eefa09420c:capstone/ports/musl-capstone/runtime/hostcall.c \
  | sed "s#\"../../../tests/runtime-qemu/#\"$REPO/capstone/tests/runtime-qemu/#" > "$O/hostcall-control.c"
grep -q 'status = capstone_main();' "$O/hostcall-control.c" \
  || { echo "40eefa09420c's hostcall.c does not return straight to the host; it controls nothing" >&2; exit 2; }
"$CAPSTONE_CLANG" "${RF[@]}" -c "$O/hostcall-control.c" -o "$O/hostcall-control.o"

rm -f "$OUT/share"/*.dom
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -O1 -c "$HERE/return_flush.c" -o "$O/return_flush.o"
CONTROL_RT=("${RUNTIME[@]/#$O\/hostcall.o/$O/hostcall-control.o}")
"$CAPSTONE_LD_LLD" --gc-sections -T "$REPO/capstone/my_first_domain/link.ld" \
  -o "$OUT/share/rf-test.dom" "${RUNTIME[@]}" "$O/return_flush.o" "$ARCHIVE"
"$CAPSTONE_LD_LLD" --gc-sections -T "$REPO/capstone/my_first_domain/link.ld" \
  -o "$OUT/share/rf-control.dom" "${CONTROL_RT[@]}" "$O/return_flush.o" "$ARCHIVE"

# The host: libc-test's, which runs one .dom to DONE and prints its status.
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_DIR=$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib
"$GUEST_CC" -O2 -I"$REPO/capstone/ports/musl-capstone/libc-test" -I"$MRT" -I"$LIBCAPSTONE_DIR" \
  -I"$REPO/capstone/tests/runtime-qemu/hostcall-stdout-probe" -I"$REPO/capstone/tests/runtime-qemu" \
  -o "$OUT/share/lt.user" "$REPO/capstone/ports/musl-capstone/libc-test/libc_test_host.c" \
  "$LIBCAPSTONE_DIR/libcapstone.c"

LOG=$OUT/run-$(date +%Y%m%d-%H%M%S).log
GUEST='echo __BOOT_OK__; cp /mnt/host/lt.user /tmp/lt.user && chmod 0755 /tmp/lt.user; for d in rf-test rf-control; do echo RUN-BEGIN $d; /tmp/lt.user /mnt/host/$d.dom 60; echo RUN-END $d rc=$?; done; echo __ALL_DONE__'
set +e
CAPSTONE_GP_NONLIN=${CAPSTONE_GP_NONLIN:-1} \
capstone_with_qemu_lock "$PYTHON" "$REPO/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$OUT/share" --log-file "$LOG" --timeout-multiplier 8 \
  --guest-command "$GUEST" --success-marker __BOOT_OK__ --success-marker __ALL_DONE__ >/dev/null 2>&1
set -e

arm_log() { sed -n "/RUN-BEGIN $1\$/,/RUN-END $1 /p" "$LOG"; }
lines() { grep -acE "^RETURN-FLUSH (line [123]|atexit handler ran)\$" <<<"$1" || true; }
verdict=0
t=$(arm_log rf-test); c=$(arm_log rf-control)
if [[ $(lines "$t") == 4 ]] && grep -aq "LT-RESULT rf-test.dom status=5 " <<<"$t"; then
  echo "  test:    PASS (3 lines + atexit handler, status 5)"
else
  echo "  test:    FAIL ($(lines "$t") of 4 lines)"; verdict=1
fi
grep -aE "^RETURN-FLUSH|LT-RESULT|halted" <<<"$t" | sed 's/^/    /'
if [[ $(lines "$c") -lt 4 ]] && grep -aq "LT-RESULT rf-control.dom status=5 " <<<"$c"; then
  echo "  control: loses output ($(lines "$c") of 4 lines), as a runtime that returns straight to the host must"
else
  echo "  control: did NOT lose output, or did not finish -- the test cannot tell the change from its absence"; verdict=1
fi
grep -aE "^RETURN-FLUSH|LT-RESULT|halted" <<<"$c" | sed 's/^/    /'
echo "  compiler $("$CAPSTONE_CLANG" --version | grep -oE '[0-9a-f]{40}' | cut -c1-12), log $LOG"
exit $verdict
