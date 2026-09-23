#!/usr/bin/env bash
# Large file I/O and large stdout writes from a musl domain on the 9p share, under QEMU.
#
#   bash run.sh     exit 0 only if, with this tree's host (host_service.h):
#                     - large-read.dom reads and writes a 64 KiB file on /mnt/host whole
#                       and in odd pieces (LARGE-READ-DONE failures=0, a PASS status), and
#                     - big-stdout.dom's 5000-byte line arrives intact in a 9p file the
#                       host's stdout is redirected to,
#                   AND both controls fail: the read control its whole read, the stdout
#                   control its long line.
#
# Two controls, each the same domain image served by a host that differs from this
# tree's in host_service.h only, taken from a fixed commit so they stay controls
# after this lands:
#   read control    e852b3951476^: FILE_READ preads straight into the payload region,
#                   which a 9p zero-copy read cannot pin (EFAULT). Must fail.
#   stdout control  40eefa09420c: FILE_READ/WRITE bounce, WRITE_STDOUT still write()s
#                   straight from the region. Measured 2026-09-23: with the host's stdout
#                   on a 9p file the 5000-byte line never arrives (longest line 47 bytes;
#                   the short lines around it do). It also flushes the host's own stdio
#                   AFTER the write, so the stdout arms differ in two things; the control's
#                   symptom is the line MISSING, the ordering bug's is the line SPLIT
#                   (4096 + 904 around the loader's messages), so the two do not mask
#                   each other.
#
# Needs: CAPSTONE_LLVM_BUILD_DIR (clang, ld.lld, llvm-ar), CAPSTONE_BUILDROOT_DIR and
# CAPSTONE_QEMU_BINARY, and a python with pexpect (PYTHON=..., default python3).
# Takes the shared QEMU lock for the run.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../capstone-test-env.sh" >/dev/null
REPO=$CAPSTONE_REPO_ROOT
OUT=${OUT:-$CAPSTONE_TMP_ROOT/large-io}
PYTHON=${PYTHON:-python3}
READ_CTL=e852b3951476^
STDOUT_CTL=40eefa09420c
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
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -c "$HERE/entry.c" -o "$O/entry.o"
COMMON=("$O/start-musl.o" "$O/tls.o" "$O/set_thread_area.o" "$O/setjmp.o"
        "${MUSL_OVERRIDE_OBJS[@]}"
        "$O/level0.o" "${softfloat_objs[@]}" "$O/entry.o" "$O/hostcall.o")
rm -f "$OUT/share"/*.dom "$OUT/share"/stdout-*.txt "$OUT/share/written.bin"
for p in large_read big_stdout; do
  "$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -O2 -c "$HERE/$p.c" -o "$O/$p.o"
  "$CAPSTONE_LD_LLD" --gc-sections -T "$REPO/capstone/my_first_domain/link.ld" \
    -o "$OUT/share/${p//_/-}.dom" "${COMMON[@]}" "$O/$p.o" "$ARCHIVE"
done
# The file to read, in the share the guest mounts at /mnt/host: byte i is i*7+3.
python3 -c "import sys; sys.stdout.buffer.write(bytes((i*7+3) & 255 for i in range(65536)))" > "$OUT/share/large.bin"

# Three hosts, libc-test's, differing only in host_service.h (a control's copy goes
# first on the include path). Each control is checked to still be the thing it controls.
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_DIR=$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib
for arm in readctl outctl; do
  mkdir -p "$O/$arm-inc"
  rev=$READ_CTL; [[ $arm == outctl ]] && rev=$STDOUT_CTL
  git -C "$REPO" show "$rev:capstone/ports/musl-capstone/runtime/host_service.h" > "$O/$arm-inc/host_service.h"
done
grep -q 'pread(fd, payload' "$O/readctl-inc/host_service.h" \
  || { echo "read control ($READ_CTL) does not pread into the region; it controls nothing" >&2; exit 2; }
grep -q 'write(STDOUT_FILENO, payload' "$O/outctl-inc/host_service.h" \
  || { echo "stdout control ($STDOUT_CTL) does not write from the region; it controls nothing" >&2; exit 2; }
for arm in test readctl outctl; do
  inc=(); out=$OUT/share/lt.user
  [[ $arm != test ]] && { inc=(-I"$O/$arm-inc"); out=$OUT/share/lt-$arm.user; }
  "$GUEST_CC" -O2 "${inc[@]}" -I"$REPO/capstone/ports/musl-capstone/libc-test" -I"$MRT" -I"$LIBCAPSTONE_DIR" \
    -I"$REPO/capstone/tests/runtime-qemu/hostcall-stdout-probe" -I"$REPO/capstone/tests/runtime-qemu" \
    -o "$out" "$REPO/capstone/ports/musl-capstone/libc-test/libc_test_host.c" "$LIBCAPSTONE_DIR/libcapstone.c"
done

# The guest's steps, as a script on the share: only a short line is typed at the prompt.
cat > "$OUT/share/steps.sh" <<'EOF'
cp /mnt/host/lt.user /mnt/host/lt-readctl.user /mnt/host/lt-outctl.user /tmp/
chmod 0755 /tmp/lt.user /tmp/lt-readctl.user /tmp/lt-outctl.user
for h in lt lt-readctl; do
  echo RUN-BEGIN read $h; /tmp/$h.user /mnt/host/large-read.dom 60; echo RUN-END read $h rc=$?
done
for h in lt lt-outctl; do
  echo RUN-BEGIN stdout $h; /tmp/$h.user /mnt/host/big-stdout.dom 60 > /mnt/host/stdout-$h.txt 2>&1
  echo RUN-END stdout $h rc=$?
done
EOF
LOG=$OUT/run-$(date +%Y%m%d-%H%M%S).log
GUEST='echo __BOOT_OK__; sh /mnt/host/steps.sh; echo __ALL_DONE__'
set +e
CAPSTONE_GP_NONLIN=${CAPSTONE_GP_NONLIN:-1} \
capstone_with_qemu_lock "$PYTHON" "$REPO/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$OUT/share" --log-file "$LOG" --timeout-multiplier 8 \
  --guest-command "$GUEST" --success-marker __BOOT_OK__ --success-marker __ALL_DONE__ >/dev/null 2>&1
set -e

arm_log() { sed -n "/RUN-BEGIN $1 $2\$/,/RUN-END $1 $2 /p" "$LOG"; }
# The long line must arrive whole: 5000 bytes, byte i = 'a' + i % 26, then the end marker.
stdout_ok() {
  python3 - "$1" <<'PY'
import sys
want = "".join(chr(97 + i % 26) for i in range(5000))
try:
    lines = open(sys.argv[1], errors="replace").read().splitlines()
except OSError as e:
    print(f"no file ({e.strerror})"); sys.exit(1)
if want in lines and "BIG-STDOUT-END" in lines:
    print("line intact, end marker present"); sys.exit(0)
longest = max((len(l) for l in lines), default=0)
print(f"line NOT intact: {len(lines)} lines, longest {longest} bytes, end marker "
      f"{'present' if 'BIG-STDOUT-END' in lines else 'absent'}"); sys.exit(1)
PY
}
verdict=0
if grep -aq "LARGE-READ-DONE failures=0" <(arm_log read lt) &&
   grep -aq "LT-RESULT large-read.dom status=0 .*PASS" <(arm_log read lt); then
  echo "  read test:      PASS"
else
  echo "  read test:      FAIL"; verdict=1
fi
arm_log read lt | grep -aE "^LARGE-READ|halted|LT-RESULT" | sed 's/^/    /'
if grep -aq "LARGE-READ FAIL read-whole" <(arm_log read lt-readctl) &&
   grep -aq "LARGE-READ-DONE" <(arm_log read lt-readctl); then
  echo "  read control:   FAILS the whole read, as it must without a bounce buffer"
else
  echo "  read control:   did NOT fail the whole read -- the test cannot see the bounce buffer"; verdict=1
fi
if r=$(stdout_ok "$OUT/share/stdout-lt.txt"); then
  echo "  stdout test:    PASS ($r)"
else
  echo "  stdout test:    FAIL ($r)"; verdict=1
fi
# The control must have RUN (file there, end marker in it) and lost the long line;
# a missing file would otherwise read as the control failing.
if r=$(stdout_ok "$OUT/share/stdout-lt-outctl.txt"); then
  echo "  stdout control: line intact -- the test cannot see WRITE_STDOUT's bounce buffer ($r)"; verdict=1
elif [[ $r != *"end marker present"* ]]; then
  echo "  stdout control: did not run to its end marker ($r) -- no measurement"; verdict=1
else
  echo "  stdout control: loses the long line, as it must without a bounce buffer ($r)"
fi
echo "  compiler $("$CAPSTONE_CLANG" --version | grep -oE '[0-9a-f]{40}' | cut -c1-12), log $LOG"
exit $verdict
