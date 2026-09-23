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
# FFAPP_HEAP: which allocator the images link. It is the only thing the arms differ in; the
# FFmpeg libraries are shared (built once, under domain/), so an arm cannot differ by accident
# in anything else.
#   level0  musl-capstone's level0 as every earlier run used it: every heap pointer carries the
#           bounds of the whole arena, and free only marks the block free.        -> domain/
#   shrink  the same allocator with CAPSTONE_LEVEL0_SHRINK: per-object bounds, still no
#           revocation.                                                           -> domain-shrink/
#   sublet  musl-capstone's sublet_heap.c instead of level0: a buddy heap over a LINEAR region the
#           host transfers (a third shared region, parked by hostcall.c under
#           CAPSTONE_PROGRAM_REGIONS), per-object bounds, and every free revokes.  -> domain-sublet/
#           The arm differs in three objects -- the allocator, hostcall.o (the parking) and the
#           guest host (the grant) -- all of them the heap's delivery, none of them FFmpeg.
HEAP=${FFAPP_HEAP:-level0}
HEAP_REGION=${FFAPP_HEAP_REGION_BYTES:-$((4 * 1024 * 1024))}   # sublet arm: the granted pool
HCF=(); HOSTF=(); ENTRYF=()
case $HEAP in
  level0) OUT="$WORK/domain"; HEAPF=() ;;
  shrink) OUT="$WORK/domain-shrink"; HEAPF=(-DCAPSTONE_LEVEL0_SHRINK=1) ;;
  sublet) OUT="$WORK/domain-sublet"; HEAPF=()
          HCF=(-DCAPSTONE_PROGRAM_REGIONS=1); HOSTF=(-DFFAPP_HEAP_REGION_BYTES="${HEAP_REGION}UL")
          ENTRYF=(-DFFAPP_SUBLET_HEAP=1) ;;
  *) echo "FFAPP_HEAP must be level0, shrink or sublet" >&2; exit 2 ;;
esac
# FFAPP_POOL (on the sublet heap only): FFmpeg's OWN pools under the buffer-pool port's
# lifetime hooks, in the whole program. libavutil comes from prepare-source.sh --pool (its own
# build directory), the buffer-pool port's payload allocator and Capstone backend are linked,
# and the guest host shares a fourth region for the payloads (program region 1).
#   0  the payloads are bounded per object and never revoked
#   2  a Sublet lease per pool get, revoked when the buffer returns to its pool
# The two differ in nothing but that mode: the matched pair for the pool fixtures.
POOL=${FFAPP_POOL:-}
POOL_REGION=${FFAPP_POOL_REGION_BYTES:-$((4 * 1024 * 1024))}
POOLF=()
if [ -n "$POOL" ]; then
  [ "$HEAP" = sublet ] || { echo "FFAPP_POOL needs FFAPP_HEAP=sublet" >&2; exit 2; }
  case $POOL in 0|2) ;; *) echo "FFAPP_POOL must be 0 or 2" >&2; exit 2 ;; esac
  OUT="$WORK/domain-sublet-pool$POOL"
  POOLF=(-DFFAPP_POOL_MODE="$POOL" -DFFAPP_POOL_REGION_BYTES="${POOL_REGION}UL"
         -I"$APP_DIR/../buffer-pool/src/shared")
  HOSTF+=(-DFFAPP_POOL_REGION_BYTES="${POOL_REGION}UL")
fi
BASE="$WORK/domain"                  # the shared FFmpeg build and configure stubs
RT="$OUT/runtime"
XB="$BASE/ffmpeg-build"
[ -n "$POOL" ] && XB="$BASE/ffmpeg-build-pool"
mkdir -p "$RT" "$XB"

CLANG=${CAPSTONE_CLANG:?}
LD_LLD=${CAPSTONE_LD_LLD:?}
[ -x "$CLANG" ] || { echo "no clang at $CLANG; from a worktree export CAPSTONE_LLVM_BUILD_DIR=<main clone>/llvm/cmake-build-debug" >&2; exit 2; }
ARCHIVE="$CAPSTONE_TMP_ROOT/musl-capstone-build/libc-capstone.a"
[ -f "$ARCHIVE" ] || { echo "no $ARCHIVE; run ports/musl-capstone/build-musl-capstone.sh (CAPSTONE_LLVM_AR=llvm-ar-18 if the build has no llvm-ar)" >&2; exit 2; }
MUSL=$(bash "$MUSL_PORT/prepare-musl-capstone.sh" | tail -1)
if [ -n "$POOL" ]; then
  SRC=$(bash "$SCRIPT_DIR/prepare-source.sh" --pool | tail -1)
else
  SRC=$(bash "$SCRIPT_DIR/prepare-source.sh" | tail -1)
fi

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
"$CLANG" "${RTF[@]}" "${HCF[@]}" -c "$MUSL_PORT/runtime/hostcall.c" -o "$RT/hostcall.o"
"$CLANG" "${RTF[@]}" -c "$MUSL_PORT/runtime/tls.c"             -o "$RT/tls.o"
if [ "$HEAP" = sublet ]; then
  "$CLANG" "${RTF[@]}" -I"$REPO_ROOT/capstone/sublet" \
                       -c "$MUSL_PORT/runtime/sublet_heap.c"   -o "$RT/heap.o"
else
  "$CLANG" "${RTF[@]}" -DCAPSTONE_LEVEL0_ARENA_BYTES="$ARENA" "${HEAPF[@]}" \
                       -c "$MUSL_PORT/runtime/level0.c"        -o "$RT/level0.o"
fi
"$CLANG" "${RTF[@]}" -c "$MUSL_PORT/runtime/string_bounds_safe.c" -o "$RT/string_bounds_safe.o"
"$CLANG" "${RTF[@]}" -I"$MUSL/src/multibyte" \
                     -c "$MUSL_PORT/runtime/mbsrtowcs_bounds_safe.c" -o "$RT/mbsrtowcs_bounds_safe.o"
"$CLANG" "${RTF[@]}" -c "$MUSL_PORT/runtime/fputwc_null_safe.c" -o "$RT/fputwc_null_safe.o"
RUNTIME=("$RT/start-musl.o" "$RT/tls.o" "$RT/set_thread_area.o" "$RT/setjmp.o"
         "$RT/string_bounds_safe.o" "$RT/mbsrtowcs_bounds_safe.o" "$RT/fputwc_null_safe.o")
if [ "$HEAP" = sublet ]; then RUNTIME+=("$RT/heap.o"); else RUNTIME+=("$RT/level0.o"); fi

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
CFGSTUBS="$BASE/cfgstubs.o"
if [ ! -f "$CFGSTUBS" ]; then
  OUT_SAVED=$OUT; OUT=$BASE
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
  OUT=$OUT_SAVED
fi

CONFIGURE_OPTS=(--disable-everything --disable-autodetect --disable-doc --disable-network --disable-asm
  --disable-pthreads --disable-programs --disable-debug --disable-iconv
  --disable-swresample --disable-swscale --disable-avfilter --disable-avdevice
  --enable-demuxer=matroska --enable-decoder=mpeg4 --enable-parser=mpeg4video
  --enable-protocol=file --enable-static --disable-shared)
CONFIG_EDIT='s/^#define HAVE_POSIX_MEMALIGN 1$/#define HAVE_POSIX_MEMALIGN 0/; s/^#define HAVE_MEMALIGN 1$/#define HAVE_MEMALIGN 0/'
# Everything that decides what the libraries contain goes into the key, so changing any of it
# rebuilds instead of silently reusing stale libraries (audit, 2026-09-23).
CONFIG_KEY=$(printf '%s\n' "$SRC" "${FLAGS[*]}" "${CONFIGURE_OPTS[*]}" "$CONFIG_EDIT" | sha256sum | cut -c1-12)
if [ ! -f "$XB/libavformat/libavformat.a" ] || [ "$(cat "$XB/.config-key" 2>/dev/null)" != "$CONFIG_KEY" ]; then
  rm -rf "$XB"; mkdir -p "$XB"
  ( cd "$XB" && "$SRC/configure" --enable-cross-compile --cc="$CLANG" --ld="$LD_LLD" \
      --arch=riscv64 --target-os=none \
      --extra-cflags="${FLAGS[*]}" --extra-ldflags="-e main --no-warn-mismatch" \
      --extra-libs="$ARCHIVE $CFGSTUBS" "${CONFIGURE_OPTS[@]}" > configure.log 2>&1 ) \
    || { echo "FFmpeg configure failed; see $XB/configure.log and $XB/ffbuild/config.log" >&2; exit 1; }
  # av_malloc -> plain malloc: with asm off ALIGN is 16 (libavutil/mem.c:65), exactly
  # level0's alignment, and musl's posix_memalign sits on an allocator this image does
  # not use.
  sed -i "$CONFIG_EDIT" "$XB/config.h"
  grep -qx '#define HAVE_POSIX_MEMALIGN 0' "$XB/config.h" \
    || { echo "config.h override did not take" >&2; exit 1; }
  make -C "$XB" -j"$JOBS" libavutil/libavutil.a libavcodec/libavcodec.a \
       libavformat/libavformat.a > "$XB/build.log" 2>&1 \
    || { echo "FFmpeg domain build failed; see $XB/build.log" >&2; exit 1; }
  echo "$CONFIG_KEY" > "$XB/.config-key"
fi
# `|| true`: zero warnings is a legitimate outcome, and under pipefail grep's exit 1 would
# otherwise stop the build silently at this line (audit, 2026-09-23).
{ grep -hE 'warning: .*\[-Wcapstone-pointer-roundtrip\]' "$XB/build.log" || true; } \
  | sed -E 's#^(src/)?##; s/: warning:.*//' | sort -u > "$OUT/pointer-roundtrip-sites.txt"
FFLIBS=("$XB/libavformat/libavformat.a" "$XB/libavcodec/libavcodec.a" "$XB/libavutil/libavutil.a")

# The pool arms' payload allocator and Capstone backend: the buffer-pool port's files,
# unmodified. Compiled after FFmpeg, because libavutil/mem.h needs the generated avconfig.h.
if [ -n "$POOL" ]; then
  BPS="$APP_DIR/../buffer-pool/src"
  POOLINC=(-I"$REPO_ROOT/capstone/runtime/include" -I"$BPS/shared" -I"$BPS/capstone-domain"
           -I"$BPS/allocators/sublet" -I"$XB" -I"$SRC")
  for f in shared/pool-allocator.c capstone-domain/payload-capabilities.c allocators/sublet/pool-leases.c; do
    o="$RT/bp-$(basename "${f%.c}").o"
    "$CLANG" "${FLAGS[@]}" "${POOLINC[@]}" -c "$BPS/$f" -o "$o"
    RUNTIME+=("$o")
  done
fi

# --- the program --------------------------------------------------------------------
APPF=("${FLAGS[@]}" "${POOLF[@]}" -I"$XB" -I"$SRC" -I"$APP_DIR/src/shared")
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
  "$CLANG" "${APPF[@]}" "${ENTRYF[@]}" -DFFAPP_STOP_AT="$stage" -DFFAPP_INPUT="\"$INPUT\"" \
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
"$CLANG" "${APPF[@]}" "${ENTRYF[@]}" -DFFAPP_STOP_AT=6 -DFFAPP_INPUT='"/mnt/host/input.mkv"' \
  -c "$APP_DIR/src/capstone-domain/ffapp_domain.c" -o "$OUT/ffapp_domain_m6_9p.o"
"$LD_LLD" --gc-sections -T "$LDS" -o "$OUT/ffapp_m6diag9p.dom" \
  "${RUNTIME[@]}" "$RT/hostcall.o" "${softfloat_objs[@]}" "$OUT/domreq.o" \
  "$OUT/ffapp_domain_m6_9p.o" "$OUT/ffapp_decode_diag.o" "${FFLIBS[@]}" "$ARCHIVE"
echo "diag images $OUT/ffapp_m6diag.dom ($INPUT) and $OUT/ffapp_m6diag9p.dom (/mnt/host/input.mkv)"

# The M5 POSITIVE CONTROL image: identical except that it decodes the one-byte-flipped
# input. In the same boot as M5 its hashes must DIFFER from the reference, or the domain
# comparison could not have failed and proves nothing (host/compare-md5.py --control).
"$CLANG" "${APPF[@]}" "${ENTRYF[@]}" -DFFAPP_STOP_AT=5 -DFFAPP_INPUT="\"${INPUT%.mkv}.flip.mkv\"" \
  -c "$APP_DIR/src/capstone-domain/ffapp_domain.c" -o "$OUT/ffapp_domain_m5flip.o"
"$LD_LLD" --gc-sections -T "$LDS" -o "$OUT/ffapp_m5flip.dom" \
  "${RUNTIME[@]}" "$RT/hostcall.o" "${softfloat_objs[@]}" "$OUT/domreq.o" \
  "$OUT/ffapp_domain_m5flip.o" "$OUT/ffapp_decode.o" "${FFLIBS[@]}" "$ARCHIVE"
echo "control image $OUT/ffapp_m5flip.dom decodes ${INPUT%.mkv}.flip.mkv"

# --- safety fixtures (src/capstone-domain/ffapp_safety.c) ------------------------------
# One image per fixture: a fault ends the emulator, so a faulting fixture reports nothing else.
# Same runtime, allocator and libraries as the milestone images above; only the entry differs.
FIXTURES="1 2 3 4 5 6 7 8 9 10"
[ -n "$POOL" ] && FIXTURES="$FIXTURES $(seq -s ' ' 11 17)"   # the pool fixtures
for fx in $FIXTURES; do
  "$CLANG" "${APPF[@]}" -DFFAPP_FIXTURE="$fx" \
    -c "$APP_DIR/src/capstone-domain/ffapp_safety.c" -o "$OUT/ffapp_safety_$fx.o"
  "$LD_LLD" --gc-sections -T "$LDS" -o "$OUT/ffapp_fx$fx.dom" \
    "${RUNTIME[@]}" "$RT/hostcall.o" "${softfloat_objs[@]}" "$OUT/domreq.o" \
    "$OUT/ffapp_safety_$fx.o" "${FFLIBS[@]}" "$ARCHIVE"
done
echo "safety fixture images ($HEAP heap${POOL:+, pool mode $POOL}): $(ls "$OUT"/ffapp_fx*.dom | wc -l)"

# --- C-50 gate: no integer address formed off sp/s0 and used as a store base ------------
# The compiler miscompiles an integer-valued pointer in a by-value aggregate (ISSUES.md
# C-50); patch 0003 removes the instance that faulted. The scan tracks integer addresses
# derived from a capability sp/s0 into any load/store base. Its positive controls are the
# C-50 reproducer and three instruction layouts that evaded an earlier version (label in
# between, the register used as a store SOURCE first, register-form add); all must hit.
"$CAPSTONE_LLVM_BIN/llvm-objdump" -d --no-show-raw-insn "$OUT/ffapp_m5.dom" > "$OUT/ffapp_m5.dis"
python3 "$SCRIPT_DIR/scan-addi-sp.py" "$OUT/ffapp_m5.dis" \
  || { echo "C-50 GATE: integer sp/s0 address used as a store base (see above)" >&2; exit 1; }

# --- every image, not just M1..M6: budget, and the layout the budget model assumes ------
# The declaration above covers the STACK only. That is right only while the image has no
# .capstone_gp_initdesc (the monitor then copies and carves nothing, and .bss -- the heap
# arena included -- sits inside code_len). If that section ever appears, the stack-only
# declaration is silently too small, which is how MicroPython failed; so it is a gate.
for img in "$OUT"/ffapp_m*.dom "$OUT"/ffapp_fx*.dom; do
  read -r code_len alloc verdict < <(budget "$img")
  [ "$verdict" = FITS ] || { echo "BUDGET: $img needs a $alloc-byte region, over the 4 MiB order ceiling" >&2; exit 1; }
  n=$("$CAPSTONE_LLVM_BIN/llvm-readelf" -SW "$img" | grep -c 'capstone_gp_initdesc' || true)
  [ "$n" = 0 ] || { echo "LAYOUT: $img has .capstone_gp_initdesc; the stack-only domreq no longer covers dom_data" >&2; exit 1; }
done
echo "budget and layout gates: $(ls "$OUT"/ffapp_m*.dom "$OUT"/ffapp_fx*.dom | wc -l) images FIT, none has .capstone_gp_initdesc"

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
# The sublet arm's heap also takes its region from hostcall.o, so exactly one more symbol.
want_undef=__capstone_hostcall
[ "$HEAP" = sublet ] && want_undef=$(printf '__capstone_hostcall\n__capstone_region')
[ "$undef" = "$want_undef" ] \
  || { echo "CONTROL FAILED: expected exactly '$(echo $want_undef)' undefined, got: ${undef:-<none>}" >&2; exit 1; }
echo "control fired: FFmpeg's libc calls reach __capstone_hostcall and nothing else is missing"

# --- guest-side host ------------------------------------------------------------------
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE_DIR="$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib"
if [ -x "$GUEST_CC" ] && [ -f "$LIBCAPSTONE_DIR/libcapstone.c" ]; then
  "$GUEST_CC" -O2 "${HOSTF[@]}" -I"$MUSL_PORT/runtime" -I"$LIBCAPSTONE_DIR" \
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
