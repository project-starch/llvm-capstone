#!/usr/bin/env bash
# M0: FFmpeg (minimal: matroska -> mpeg4) built and linked as a Capstone domain, on the
# musl-capstone / my_first_domain/link.ld ABI, plus the guest-side host.
#
# One image per milestone (FFAPP_STOP_AT = 1..5), so every QEMU run returns a result
# (src/shared/ffapp_decode.h). Only the domain entry object differs between them.
#
# GATES, each of which fails the build rather than warning:
#   * NEGATIVE CONTROL: the M5 image relinked WITHOUT hostcall.o must come back with
#     exactly __capstone_hostcall undefined. Anything else means the libc chain this image
#     depends on is not the one being linked (musl-capstone stdio-probe's control, reused).
#   * BUDGET: the kernel module's own sizing (modcapstone module/capstone.c:152-161) --
#     code_len = the PT_LOAD span through p_memsz (so .bss and the level0 arena count), and
#     with a .capstone_domreq declaration, code_len + 8 KiB + declared data -- rounded to a
#     power-of-two page count, must fit one order-10 allocation (4 MiB). Without the
#     declaration the module would size 2*code_len, which does not fit; hence domreq.S.
#
# Worktree note: a git worktree has no LLVM build and an empty buildroot submodule. Point
# CAPSTONE_LLVM_BUILD_DIR and CAPSTONE_BUILDROOT_DIR at the main clone.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
APP_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$APP_DIR/../../../tests/capstone-test-env.sh"

REPO_ROOT=$CAPSTONE_REPO_ROOT
MUSL_PORT="$REPO_ROOT/capstone/ports/musl-capstone"
WORK=${FFAPP_WORK:-$CAPSTONE_TMP_ROOT/ffmpeg-app}
JOBS=${FFAPP_JOBS:-48}
ARENA=${FFAPP_ARENA_BYTES:-$((1536 * 1024))}      # level0 heap; native peak is 0.71 MB
STACK=${FFAPP_STACK_BYTES:-$((256 * 1024))}       # declared stack (dom_data)
# /tmp, not the 9p share: the host pread()s straight into the shared-region mapping, and a 9p
# read that large goes zero-copy (it pins the destination pages), which a region mapping
# refuses -> EFAULT (2026-09-23; the probe images below keep the /mnt/host twin for the
# matched pair). run-qemu.sh copies the inputs into the guest's /tmp first.
INPUT=${FFAPP_INPUT:-/tmp/input.mkv}
ORDER_CEILING=$((4 * 1024 * 1024))
OUT="$WORK/domain"
RT="$OUT/runtime"
XB="$OUT/ffmpeg-build"
mkdir -p "$RT" "$XB"

CLANG=${CAPSTONE_CLANG:?}
LD_LLD=${CAPSTONE_LD_LLD:?}
[ -x "$CLANG" ] || { echo "no clang at $CLANG; from a worktree export CAPSTONE_LLVM_BUILD_DIR=<main clone>/llvm/cmake-build-debug" >&2; exit 2; }
ARCHIVE="$CAPSTONE_TMP_ROOT/musl-capstone-build/libc-capstone.a"
[ -f "$ARCHIVE" ] || { echo "no $ARCHIVE; run ports/musl-capstone/build-musl-capstone.sh (CAPSTONE_LLVM_AR=llvm-ar-18 if the build has no llvm-ar)" >&2; exit 2; }
MUSL=$(bash "$MUSL_PORT/prepare-musl-capstone.sh" | tail -1)
SRC=$(bash "$SCRIPT_DIR/prepare-source.sh" | tail -1)

TARGET=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
        -Xclang -target-feature -Xclang +a)
INC=(-nostdinc -isystem "$MUSL/arch/capstone64" -isystem "$MUSL/arch/generic"
     -isystem "$MUSL/obj/include" -isystem "$MUSL/include" -isystem "$("$CLANG" -print-resource-dir)/include")
FLAGS=("${TARGET[@]}" -ffreestanding -fno-builtin -fno-jump-tables -ffunction-sections
       -fdata-sections -O1 -Wno-int-conversion -D_GNU_SOURCE "${INC[@]}")

# --- runtime objects: musl-capstone's, as its stdio-probe links them -----------------
RTF=("${TARGET[@]}" -ffreestanding -fno-builtin -fno-jump-tables -ffunction-sections
     -fdata-sections -std=c99 -O1 -w -Wno-int-conversion -D_XOPEN_SOURCE=700
     "${INC[@]}" -I"$MUSL/src/include" -I"$MUSL/src/internal" -I"$MUSL/obj/src/internal")
ASM=("${TARGET[@]}" -ffreestanding -O0)
"$CLANG" "${ASM[@]}" -c "$MUSL_PORT/runtime/start-musl.S"      -o "$RT/start-musl.o"
"$CLANG" "${ASM[@]}" -c "$MUSL_PORT/runtime/set_thread_area.S" -o "$RT/set_thread_area.o"
"$CLANG" "${ASM[@]}" -c "$MUSL_PORT/runtime/setjmp.S"          -o "$RT/setjmp.o"
"$CLANG" "${RTF[@]}" -c "$MUSL_PORT/runtime/hostcall.c"        -o "$RT/hostcall.o"
"$CLANG" "${RTF[@]}" -c "$MUSL_PORT/runtime/tls.c"             -o "$RT/tls.o"
"$CLANG" "${RTF[@]}" -DCAPSTONE_LEVEL0_ARENA_BYTES="$ARENA" \
                     -c "$MUSL_PORT/runtime/level0.c"          -o "$RT/level0.o"
"$CLANG" "${RTF[@]}" -c "$MUSL_PORT/runtime/string_bounds_safe.c" -o "$RT/string_bounds_safe.o"
"$CLANG" "${RTF[@]}" -I"$MUSL/src/multibyte" \
                     -c "$MUSL_PORT/runtime/mbsrtowcs_bounds_safe.c" -o "$RT/mbsrtowcs_bounds_safe.o"
"$CLANG" "${RTF[@]}" -c "$MUSL_PORT/runtime/fputwc_null_safe.c" -o "$RT/fputwc_null_safe.o"
RUNTIME=("$RT/start-musl.o" "$RT/tls.o" "$RT/set_thread_area.o" "$RT/setjmp.o"
         "$RT/string_bounds_safe.o" "$RT/mbsrtowcs_bounds_safe.o" "$RT/fputwc_null_safe.o"
         "$RT/level0.o")

# Soft-float builtins from the shared list (a domain has no FP hardware ABI).
COMPILER_RT="$REPO_ROOT/compiler-rt/lib/builtins"
OBJ_DIR="$RT"
COMMON_FLAGS=("${TARGET[@]}" -ffreestanding -fno-builtin -ffunction-sections -fdata-sections -O1 -w)
source "$REPO_ROOT/capstone/benchmarks/beebs/build-beebs-softfloat-common.sh"

# --- FFmpeg, cross-configured for capstone64 --------------------------------------------
# configure's HAVE_* tests LINK. The real runtime cannot serve them (hostcall.o needs a
# capstone_main that test programs do not have), so the link tests get empty definitions of
# exactly the symbols libc-capstone.a leaves undefined, plus the soft-float set. Then a
# HAVE_x test passes exactly when musl-capstone itself provides x. These stubs are for
# configure only; they are never linked into a domain image.
CFGSTUBS="$OUT/cfgstubs.o"
if [ ! -f "$CFGSTUBS" ]; then
  { "$CAPSTONE_LLVM_BIN/llvm-nm" -u "$ARCHIVE" 2>/dev/null | awk 'NF>=2{print $NF}' | sort -u
    for s in __floatdisf __floatundisf __floatundidf __floatunditf __fixunsdfdi __fixunssfdi \
             __fixunstfdi __fixunsdfsi __fixunssfsi __powidf2 __powisf2 __divdc3 __divsc3 \
             __negdf2 __negsf2 __ashlti3 __lshrti3 __ashrti3 __multi3 __divti3 __udivti3 \
             __modti3 __umodti3 __fixdfti __fixsfti __floattidf __floattisf; do echo "$s"; done
  } | sort -u > "$OUT/cfgstubs.syms"
  "$CAPSTONE_LLVM_BIN/llvm-nm" --defined-only "$ARCHIVE" 2>/dev/null | awk 'NF>=3{print $3}' \
    | sort -u > "$OUT/libc.defined"
  comm -23 "$OUT/cfgstubs.syms" "$OUT/libc.defined" | while read -r s; do
    case $s in
      __init_array_*|__fini_array_*|_DYNAMIC) echo "char $s[1];" ;;
      __vdsosym) echo "void *__vdsosym(const char *a, const char *b) { return 0; }" ;;
      *) echo "void $s(void) {}" ;;
    esac
  done > "$OUT/cfgstubs.c"
  "$CLANG" "${FLAGS[@]}" -w -c "$OUT/cfgstubs.c" -o "$CFGSTUBS"
fi

CONFIG_KEY=$(printf '%s\n' "$SRC" "${FLAGS[*]}" | sha256sum | cut -c1-12)
if [ ! -f "$XB/libavformat/libavformat.a" ] || [ "$(cat "$XB/.config-key" 2>/dev/null)" != "$CONFIG_KEY" ]; then
  rm -rf "$XB"; mkdir -p "$XB"
  ( cd "$XB" && "$SRC/configure" --enable-cross-compile --cc="$CLANG" --ld="$LD_LLD" \
      --arch=riscv64 --target-os=none \
      --extra-cflags="${FLAGS[*]}" --extra-ldflags="-e main --no-warn-mismatch" \
      --extra-libs="$ARCHIVE $CFGSTUBS" \
      --disable-everything --disable-autodetect --disable-doc --disable-network --disable-asm \
      --disable-pthreads --disable-programs --disable-debug --disable-iconv \
      --disable-swresample --disable-swscale --disable-avfilter --disable-avdevice \
      --enable-demuxer=matroska --enable-decoder=mpeg4 --enable-parser=mpeg4video \
      --enable-protocol=file --enable-static --disable-shared > configure.log 2>&1 ) \
    || { echo "FFmpeg configure failed; see $XB/configure.log and $XB/ffbuild/config.log" >&2; exit 1; }
  # av_malloc -> plain malloc: with asm off ALIGN is 16 (libavutil/mem.c:65), exactly
  # level0's alignment, and musl's posix_memalign sits on an allocator this image does
  # not use.
  sed -i 's/^#define HAVE_POSIX_MEMALIGN 1$/#define HAVE_POSIX_MEMALIGN 0/;
          s/^#define HAVE_MEMALIGN 1$/#define HAVE_MEMALIGN 0/' "$XB/config.h"
  grep -qx '#define HAVE_POSIX_MEMALIGN 0' "$XB/config.h" \
    || { echo "config.h override did not take" >&2; exit 1; }
  make -C "$XB" -j"$JOBS" libavutil/libavutil.a libavcodec/libavcodec.a \
       libavformat/libavformat.a > "$XB/build.log" 2>&1 \
    || { echo "FFmpeg domain build failed; see $XB/build.log" >&2; exit 1; }
  echo "$CONFIG_KEY" > "$XB/.config-key"
fi
grep -hE 'warning: .*\[-Wcapstone-pointer-roundtrip\]' "$XB/build.log" \
  | sed -E 's#^(src/)?##; s/: warning:.*//' | sort -u > "$OUT/pointer-roundtrip-sites.txt"
FFLIBS=("$XB/libavformat/libavformat.a" "$XB/libavcodec/libavcodec.a" "$XB/libavutil/libavutil.a")

# --- the program --------------------------------------------------------------------
APPF=("${FLAGS[@]}" -I"$XB" -I"$SRC" -I"$APP_DIR/src/shared")
"$CLANG" "${APPF[@]}" -c "$APP_DIR/src/shared/ffapp_decode.c" -o "$OUT/ffapp_decode.o"
"$CLANG" "${ASM[@]}" -DCAPSTONE_DOMREQ_DATA="$STACK" -DCAPSTONE_DOMREQ_STACK="$STACK" \
  -c "$REPO_ROOT/capstone/tests/runtime-qemu/domreq.S" -o "$OUT/domreq.o"
LDS="$REPO_ROOT/capstone/my_first_domain/link.ld"

budget() {   # prints: code_len total_bytes verdict
  "$CAPSTONE_LLVM_BIN/llvm-readelf" -lW "$1" | python3 -c '
import sys, re
stack, ceiling = int(sys.argv[1]), int(sys.argv[2])
lo = hi = None
for l in sys.stdin:
    m = re.match(r"\s*LOAD\s+0x[0-9a-f]+\s+(0x[0-9a-f]+)\s+0x[0-9a-f]+\s+0x[0-9a-f]+\s+(0x[0-9a-f]+)", l)
    if m:
        va, memsz = int(m.group(1), 16), int(m.group(2), 16)
        lo = va if lo is None else min(lo, va); hi = va + memsz if hi is None else max(hi, va + memsz)
if lo is None:
    sys.exit("no PT_LOAD")
code_len = hi - lo
tot = code_len + 8192 + stack
pages = (tot - 1) // 4096 + 1
p2 = 1
while p2 < pages: p2 *= 2
alloc = p2 * 4096
print(code_len, alloc, "FITS" if alloc <= ceiling else "DOES-NOT-FIT")' "$STACK" "$ORDER_CEILING"
}

for stage in 1 2 3 4 5 6; do   # 6 = M2a, open_input only (bisection stage)
  "$CLANG" "${APPF[@]}" -DFFAPP_STOP_AT="$stage" -DFFAPP_INPUT="\"$INPUT\"" \
    -c "$APP_DIR/src/capstone-domain/ffapp_domain.c" -o "$OUT/ffapp_domain_m$stage.o"
  "$LD_LLD" --gc-sections -T "$LDS" -o "$OUT/ffapp_m$stage.dom" \
    "${RUNTIME[@]}" "$RT/hostcall.o" "${softfloat_objs[@]}" "$OUT/domreq.o" \
    "$OUT/ffapp_domain_m$stage.o" "$OUT/ffapp_decode.o" "${FFLIBS[@]}" "$ARCHIVE"
  read -r code_len alloc verdict < <(budget "$OUT/ffapp_m$stage.dom")
  printf 'M%d image %s  code_len=%d  allocation=%d  %s\n' "$stage" "$OUT/ffapp_m$stage.dom" \
    "$code_len" "$alloc" "$verdict"
  [ "$verdict" = FITS ] || { echo "BUDGET: M$stage needs a $alloc-byte region, over the 4 MiB order ceiling" >&2; exit 1; }
done

# DIAGNOSTIC image (M2a with FFAPP_DIAG): a separate decode object, so the production images
# above are byte-identical with or without it. It prints the first bytes a plain fread gets,
# the registered demuxers, the probe's verdict, and avformat_open_input's error.
"$CLANG" "${APPF[@]}" -DFFAPP_DIAG -c "$APP_DIR/src/shared/ffapp_decode.c" -o "$OUT/ffapp_decode_diag.o"
"$LD_LLD" --gc-sections -T "$LDS" -o "$OUT/ffapp_m6diag.dom" \
  "${RUNTIME[@]}" "$RT/hostcall.o" "${softfloat_objs[@]}" "$OUT/domreq.o" \
  "$OUT/ffapp_domain_m6.o" "$OUT/ffapp_decode_diag.o" "${FFLIBS[@]}" "$ARCHIVE"
# Its matched twin: identical except that it reads the input from the 9p share.
"$CLANG" "${APPF[@]}" -DFFAPP_STOP_AT=6 -DFFAPP_INPUT='"/mnt/host/input.mkv"' \
  -c "$APP_DIR/src/capstone-domain/ffapp_domain.c" -o "$OUT/ffapp_domain_m6_9p.o"
"$LD_LLD" --gc-sections -T "$LDS" -o "$OUT/ffapp_m6diag9p.dom" \
  "${RUNTIME[@]}" "$RT/hostcall.o" "${softfloat_objs[@]}" "$OUT/domreq.o" \
  "$OUT/ffapp_domain_m6_9p.o" "$OUT/ffapp_decode_diag.o" "${FFLIBS[@]}" "$ARCHIVE"
echo "diag images $OUT/ffapp_m6diag.dom ($INPUT) and $OUT/ffapp_m6diag9p.dom (/mnt/host/input.mkv)"

# The M5 POSITIVE CONTROL image: identical except that it decodes the one-byte-flipped
# input. In the same boot as M5 its hashes must DIFFER from the reference, or the domain
# comparison could not have failed and proves nothing (host/compare-md5.py --control).
"$CLANG" "${APPF[@]}" -DFFAPP_STOP_AT=5 -DFFAPP_INPUT="\"${INPUT%.mkv}.flip.mkv\"" \
  -c "$APP_DIR/src/capstone-domain/ffapp_domain.c" -o "$OUT/ffapp_domain_m5flip.o"
"$LD_LLD" --gc-sections -T "$LDS" -o "$OUT/ffapp_m5flip.dom" \
  "${RUNTIME[@]}" "$RT/hostcall.o" "${softfloat_objs[@]}" "$OUT/domreq.o" \
  "$OUT/ffapp_domain_m5flip.o" "$OUT/ffapp_decode.o" "${FFLIBS[@]}" "$ARCHIVE"
echo "control image $OUT/ffapp_m5flip.dom decodes ${INPUT%.mkv}.flip.mkv"

# --- C-50 gate: no integer address formed off sp/s0 and used as a store base ------------
# The compiler miscompiles an integer-valued pointer in a by-value union (ISSUES.md C-50);
# patch 0003 removes the one instance. This scan found exactly that instance in the
# unpatched image (its positive control) and must find none now.
"$CAPSTONE_LLVM_BIN/llvm-objdump" -d --no-show-raw-insn "$OUT/ffapp_m5.dom" > "$OUT/ffapp_m5.dis"
python3 "$SCRIPT_DIR/scan-addi-sp.py" "$OUT/ffapp_m5.dis" \
  || { echo "C-50 GATE: integer sp/s0 address used as a store base (see above)" >&2; exit 1; }

# --- negative control ---------------------------------------------------------------
cat > "$OUT/stub_main.c" <<'STUB'
int capstone_main(void);
void domain_main(unsigned *res, unsigned func) { (void)func; if (res) *res = (unsigned)capstone_main(); }
unsigned long __capstone_unserved_count(void) { return 0; }
long __capstone_unserved_at(unsigned long i) { (void)i; return -1; }
STUB
"$CLANG" "${FLAGS[@]}" -c "$OUT/stub_main.c" -o "$OUT/stub_main.o"
set +e
control=$("$LD_LLD" --gc-sections -T "$LDS" -o "$OUT/nohostcall.dom" \
  "${RUNTIME[@]}" "$OUT/stub_main.o" "${softfloat_objs[@]}" "$OUT/domreq.o" \
  "$OUT/ffapp_domain_m5.o" "$OUT/ffapp_decode.o" "${FFLIBS[@]}" "$ARCHIVE" 2>&1)
set -e
undef=$(printf '%s\n' "$control" | grep -oE 'undefined symbol: [A-Za-z_][A-Za-z0-9_]*' \
        | sed 's/undefined symbol: //' | sort -u)
[ "$undef" = "__capstone_hostcall" ] \
  || { echo "CONTROL FAILED: expected exactly '__capstone_hostcall' undefined, got: ${undef:-<none>}" >&2; exit 1; }
echo "control fired: FFmpeg's libc calls reach __capstone_hostcall and nothing else is missing"

# --- guest-side host ------------------------------------------------------------------
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_DIR="$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib"
if [ -x "$GUEST_CC" ] && [ -f "$LIBCAPSTONE_DIR/libcapstone.c" ]; then
  "$GUEST_CC" -O2 -I"$MUSL_PORT/runtime" -I"$LIBCAPSTONE_DIR" \
    -I"$REPO_ROOT/capstone/tests/runtime-qemu/hostcall-stdout-probe" \
    -I"$REPO_ROOT/capstone/tests/runtime-qemu" \
    -o "$OUT/ffapp.user" "$APP_DIR/src/linux-guest/ffapp_host.c" "$LIBCAPSTONE_DIR/libcapstone.c"
  echo "built   $OUT/ffapp.user"
else
  echo "guest host NOT built: no $GUEST_CC or libcapstone.c (from a worktree, export CAPSTONE_BUILDROOT_DIR=<main clone>/capstone/caplifive-buildroot)" >&2
  exit 1
fi
printf 'pointer round-trip sites flagged by the compiler: %d (list: %s)\n' \
  "$(wc -l < "$OUT/pointer-roundtrip-sites.txt")" "$OUT/pointer-roundtrip-sites.txt"
