#!/usr/bin/env bash
# exit() from main in a musl domain, with and without a __capstone_at_exit hook (C-56).
#
#   bash run.sh        build exit_test.c three times, run all in one boot; exit 0 only
#                      if exit-default ends with status 7 and exit-hook with 42, both
#                      print the line buffered before exit(), neither faults, AND
#                      exit-control halts
#
# exit-control is exit-default linked against the hostcall.c of 40eefa09420c (dev
# before this fix; pinned, so it stays a control after this lands), which only
# declares the hook weak and tests its address: in a domain that address is not
# NULL (C-56), so exit() jumps to the image base and halts. It runs LAST because a
# halted domain can take the guest with it.
#
# Needs: CAPSTONE_LLVM_BUILD_DIR (clang, ld.lld, llvm-ar), CAPSTONE_BUILDROOT_DIR
# and CAPSTONE_QEMU_BINARY (the tree that has them), and a python with pexpect
# (PYTHON=..., default python3). Takes the shared QEMU lock for the run.
# A missing status line is a FAIL, never "no result".
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../capstone-test-env.sh" >/dev/null
REPO=$CAPSTONE_REPO_ROOT
OUT=${OUT:-$CAPSTONE_TMP_ROOT/exit-hook}
PYTHON=${PYTHON:-python3}
AR=$CAPSTONE_LLVM_BIN/llvm-ar
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
for f in hostcall tls level0 string_bounds_safe fputwc_null_safe; do
  "$CAPSTONE_CLANG" "${RF[@]}" -c "$MRT/$f.c" -o "$O/$f.o"
done
"$CAPSTONE_CLANG" "${RF[@]}" -I"$MUSL/src/multibyte" -c "$MRT/mbsrtowcs_bounds_safe.c" \
  -o "$O/mbsrtowcs_bounds_safe.o"
CLANG=$CAPSTONE_CLANG OBJ_DIR=$O COMPILER_RT=$REPO/compiler-rt/lib/builtins
COMMON_FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
              -ffreestanding -fno-builtin -ffunction-sections -fdata-sections -O1 -w)
source "$REPO/capstone/benchmarks/beebs/build-beebs-softfloat-common.sh"
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -c "$HERE/entry.c" -o "$O/entry.o"
RUNTIME=("$O/start-musl.o" "$O/hostcall.o" "$O/tls.o" "$O/set_thread_area.o" "$O/setjmp.o"
         "$O/string_bounds_safe.o" "$O/mbsrtowcs_bounds_safe.o" "$O/fputwc_null_safe.o"
         "$O/level0.o" "${softfloat_objs[@]}" "$O/entry.o")

# The control's one difference: 40eefa09420c's hostcall.c. Its one relative include is
# made absolute, since the copy lives in $O.
git -C "$REPO" show 40eefa09420c:capstone/ports/musl-capstone/runtime/hostcall.c \
  | sed "s#\"../../../tests/runtime-qemu/#\"$REPO/capstone/tests/runtime-qemu/#" > "$O/hostcall-control.c"
grep -q 'if (__capstone_at_exit)' "$O/hostcall-control.c" \
  || { echo "40eefa09420c's hostcall.c does not test the hook's address; it controls nothing" >&2; exit 2; }
"$CAPSTONE_CLANG" "${RF[@]}" -c "$O/hostcall-control.c" -o "$O/hostcall-control.o"

rm -f "$OUT/share"/*.dom
for arm in default hook; do
  def=(); [[ $arm == hook ]] && def=(-DWITH_HOOK)
  "$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -O1 "${def[@]}" -c "$HERE/exit_test.c" -o "$O/test-$arm.o"
  "$CAPSTONE_LD_LLD" --gc-sections -T "$REPO/capstone/my_first_domain/link.ld" \
    -o "$OUT/share/exit-$arm.dom" "${RUNTIME[@]}" "$O/test-$arm.o" "$ARCHIVE"
done
CONTROL_RT=("${RUNTIME[@]/#$O\/hostcall.o/$O/hostcall-control.o}")
"$CAPSTONE_LD_LLD" --gc-sections -T "$REPO/capstone/my_first_domain/link.ld" \
  -o "$OUT/share/exit-control.dom" "${CONTROL_RT[@]}" "$O/test-default.o" "$ARCHIVE"

# exit-control references the hook it does not define, on purpose, and the runner refuses a
# share with an undefined weak symbol (C-56) unless told otherwise. Check that the control is
# the ONLY image with one -- the runtime under test defines the hook, so no other image may
# carry one, and a control without it controls nothing -- then let this share through.
for d in "$OUT/share"/*.dom; do
  w=$("$CAPSTONE_LLVM_BIN/llvm-nm" --undefined-only "$d" | awk '$1 == "w" || $1 == "v" { print $2 }' | tr '\n' ' ')
  case $(basename "$d" .dom):$w in
    "exit-control:__capstone_at_exit ") ;;
    exit-control:*) echo "exit-control does not reference the hook weakly (undefined weak: $w); it controls nothing" >&2; exit 2 ;;
    *:) ;;
    *) echo "$(basename "$d"): undefined weak $w-- the runtime must define the hook" >&2; exit 1 ;;
  esac
done
export CAPSTONE_ALLOW_UNDEFINED_WEAK=1

# The host: libc-test's, which runs one .dom to DONE and prints its status.
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_DIR=$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib
"$GUEST_CC" -O2 -I"$REPO/capstone/ports/musl-capstone/libc-test" -I"$MRT" -I"$LIBCAPSTONE_DIR" \
  -I"$REPO/capstone/tests/runtime-qemu/hostcall-stdout-probe" -I"$REPO/capstone/tests/runtime-qemu" \
  -o "$OUT/share/lt.user" "$REPO/capstone/ports/musl-capstone/libc-test/libc_test_host.c" \
  "$LIBCAPSTONE_DIR/libcapstone.c"

LOG=$OUT/run-$(date +%Y%m%d-%H%M%S).log
GUEST='echo __BOOT_OK__; cp /mnt/host/lt.user /tmp/lt.user && chmod 0755 /tmp/lt.user; for d in exit-default exit-hook exit-control; do echo RUN-BEGIN $d; /tmp/lt.user /mnt/host/$d.dom 60; echo RUN-END $d rc=$?; done; echo __ALL_DONE__'
set +e
capstone_with_qemu_lock "$PYTHON" "$REPO/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$OUT/share" --log-file "$LOG" --timeout-multiplier 8 \
  --guest-command "$GUEST" --success-marker __BOOT_OK__ --success-marker __ALL_DONE__ >/dev/null 2>&1
set -e

verdict=0
for arm in default:7 hook:42; do
  name=exit-${arm%%:*}; want=${arm##*:}
  block=$(sed -n "/RUN-BEGIN $name\$/,/RUN-END $name /p" "$LOG")
  if grep -aq "EXIT-TEST before exit" <<<"$block" &&
     grep -aq "LT-RESULT $name.dom status=$want " <<<"$block" &&
     ! grep -aq "halted by capability fault" <<<"$block"; then
    echo "  $name: PASS (status $want)"
  else
    echo "  $name: FAIL (want status $want)"
    grep -aE "EXIT-TEST|LT-RESULT|halted" <<<"$block" | sed 's/^/    /'
    verdict=1
  fi
done
block=$(sed -n "/RUN-BEGIN exit-control\$/,\$p" "$LOG")
if grep -aq "halted by capability fault" <<<"$block"; then
  echo "  exit-control: halts, as it must when the runtime tests the hook's address"
else
  echo "  exit-control: did NOT halt -- the test cannot tell the fix from its absence"
  grep -aE "EXIT-TEST|LT-RESULT" <<<"$block" | sed 's/^/    /'
  verdict=1
fi
echo "  compiler $("$CAPSTONE_CLANG" --version | grep -oE '[0-9a-f]{40}' | cut -c1-12), log $LOG"
exit $verdict
