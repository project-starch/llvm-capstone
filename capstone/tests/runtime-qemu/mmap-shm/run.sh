#!/usr/bin/env bash
# mmap, munmap and System V shared memory in a musl domain, from level0, under QEMU.
#
#   bash run.sh        exit 0 only if the domain built with this tree's runtime
#                      prints "MMAP-SHM-DONE failures=0" with a PASS status
#                      line, AND the control fails its first mmap
#
# The control is the same domain linked WITHOUT runtime/mmap_shm_level0.o, so
# musl's own mmap() and shmget() reach the syscall layer, which has no mmap and
# no shm: both come back ENOSYS there. It differs from the test in that one
# object only -- hostcall.c and the host are the same in both arms -- so a run
# that passes both ways is not being served by the override at all, and the
# run says so. level0 is compiled with a 512 KiB arena for both arms: the
# mappings need it, the test's "beyond the arena" check counts on it, and an
# undeclared domain's block is sized from its image (2 x code_len), so a 2 MiB
# arena made create_dom fail outright (measured, 2026-09-24).
#
# The control's failure takes one of two forms, both accepted: musl's mmap()
# returns -1 with ENOSYS and the first check says FAIL, or -- C-32 -- the -1
# crosses the call as an integer, the caller's movc turns it into NULL, the
# check reads NULL != MAP_FAILED as a pass and the domain halts reading through
# NULL (seen on clang 595e757cb696: "cincoffset with an UNTAGGED rs1"). Either
# way the control has no mmap, which is what it is there to show.
#
# Needs: CAPSTONE_LLVM_BUILD_DIR (clang, ld.lld, llvm-ar), CAPSTONE_BUILDROOT_DIR
# and CAPSTONE_QEMU_BINARY, and a python with pexpect (PYTHON=..., default
# python3). Takes the shared QEMU lock for the run.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../capstone-test-env.sh" >/dev/null
REPO=$CAPSTONE_REPO_ROOT
OUT=${OUT:-$CAPSTONE_TMP_ROOT/mmap-shm}
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

# The runtime, as musl-capstone/libc-test/build-libc-test.sh builds it, with
# level0's arena at 512 KiB for this domain.
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
for f in hostcall tls; do
  "$CAPSTONE_CLANG" "${RF[@]}" -c "$MRT/$f.c" -o "$O/$f.o"
done
"$CAPSTONE_CLANG" "${RF[@]}" -DCAPSTONE_LEVEL0_ARENA_BYTES='(512*1024)' -c "$MRT/level0.c" -o "$O/level0.o"
# The libc overrides, from the one list every musl domain links (runtime/libc_overrides.sh).
source "$MRT/libc_overrides.sh"
grep -q 'mmap_shm_level0' "$MRT/libc_overrides.sh" \
  || { echo "libc_overrides.sh does not list mmap_shm_level0; the test would link musl's mmap" >&2; exit 2; }
build_musl_overrides "$CAPSTONE_CLANG" "$O" "$MUSL" "${RF[@]}"
# The control's one difference: the override list without mmap_shm_level0.o.
CONTROL_OVERRIDES=()
for o in "${MUSL_OVERRIDE_OBJS[@]}"; do
  [[ $o == */mmap_shm_level0.o ]] || CONTROL_OVERRIDES+=("$o")
done
(( ${#CONTROL_OVERRIDES[@]} == ${#MUSL_OVERRIDE_OBJS[@]} - 1 )) \
  || { echo "the override list has no mmap_shm_level0.o to leave out; the control would not control" >&2; exit 2; }
CLANG=$CAPSTONE_CLANG OBJ_DIR=$O COMPILER_RT=$REPO/compiler-rt/lib/builtins
COMMON_FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
              -ffreestanding -fno-builtin -ffunction-sections -fdata-sections -O1 -w)
source "$REPO/capstone/benchmarks/beebs/build-beebs-softfloat-common.sh"
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -c "$HERE/entry.c" -o "$O/entry.o"
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -O2 -c "$HERE/mmap_shm.c" -o "$O/mmap_shm.o"
BASE=("$O/start-musl.o" "$O/hostcall.o" "$O/tls.o" "$O/set_thread_area.o" "$O/setjmp.o")
TAIL=("$O/level0.o" "${softfloat_objs[@]}" "$O/entry.o" "$O/mmap_shm.o")
rm -f "$OUT/share"/*.dom
"$CAPSTONE_LD_LLD" --gc-sections -T "$REPO/capstone/my_first_domain/link.ld" \
  -o "$OUT/share/ms-test.dom" "${BASE[@]}" "${MUSL_OVERRIDE_OBJS[@]}" "${TAIL[@]}" "$ARCHIVE"
"$CAPSTONE_LD_LLD" --gc-sections -T "$REPO/capstone/my_first_domain/link.ld" \
  -o "$OUT/share/ms-control.dom" "${BASE[@]}" "${CONTROL_OVERRIDES[@]}" "${TAIL[@]}" "$ARCHIVE"

# The host: libc-test's, built from this tree; nothing changed on the wire.
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_DIR=$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib
"$GUEST_CC" -O2 -I"$REPO/capstone/ports/musl-capstone/libc-test" -I"$MRT" -I"$LIBCAPSTONE_DIR" \
  -I"$REPO/capstone/tests/runtime-qemu/hostcall-stdout-probe" -I"$REPO/capstone/tests/runtime-qemu" \
  -o "$OUT/share/lt.user" "$REPO/capstone/ports/musl-capstone/libc-test/libc_test_host.c" \
  "$LIBCAPSTONE_DIR/libcapstone.c"

LOG=$OUT/run-$(date +%Y%m%d-%H%M%S).log
GUEST='echo __BOOT_OK__; cp /mnt/host/lt.user /tmp/lt.user && chmod 0755 /tmp/lt.user; for d in ms-test ms-control; do echo RUN-BEGIN $d; /tmp/lt.user /mnt/host/$d.dom 60; echo RUN-END $d rc=$?; done; echo __ALL_DONE__'
set +e
CAPSTONE_GP_NONLIN=${CAPSTONE_GP_NONLIN:-1} \
capstone_with_qemu_lock "$PYTHON" "$REPO/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$OUT/share" --log-file "$LOG" --timeout-multiplier 8 \
  --guest-command "$GUEST" --success-marker __BOOT_OK__ --success-marker __ALL_DONE__ >/dev/null 2>&1
set -e

arm_log() { sed -n "/RUN-BEGIN ms-$1/,/RUN-END ms-$1/p" "$LOG"; }
verdict=0
if grep -aq "MMAP-SHM-DONE failures=0" <(arm_log test) &&
   grep -aq "LT-RESULT ms-test.dom status=0 .*PASS" "$LOG"; then
  echo "  test:    PASS"
else
  echo "  test:    FAIL"; verdict=1
fi
arm_log test | grep -aE "^MMAP-SHM|halted|LT-RESULT" | sed 's/^/    /' || true
if grep -aq "MMAP-SHM FAIL mmap " <(arm_log control) &&
   grep -aq "MMAP-SHM-DONE" <(arm_log control); then
  echo "  control: FAILS mmap with ENOSYS, as it must without the override"
elif grep -aq "domain halted by capability fault" <(arm_log control) &&
     ! grep -aq "MMAP-SHM-DONE" <(arm_log control); then
  echo "  control: HALTS on musl's MAP_FAILED (C-32: the -1 arrives as NULL through movc); no mmap either way"
else
  echo "  control: did NOT fail mmap -- the test cannot tell the override from its absence"; verdict=1
fi
arm_log control | grep -aE "^MMAP-SHM|halted|LT-RESULT" | sed 's/^/    /' || true
echo "  compiler $("$CAPSTONE_CLANG" --version | grep -oE '[0-9a-f]{40}' | cut -c1-12), log $LOG"
exit $verdict
