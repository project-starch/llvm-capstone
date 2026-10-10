#!/usr/bin/env bash
# Build the pinned decoder and safety fixtures as delegated applications.
# The common SDK owns startup, libc overrides, syscalls, grants, and builtins.
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
# Delegated reads use launcher bounce buffers, including inputs on the 9p share.
# FFAPP_CLIP_SECONDS (build-native.sh): the workload. 1 is the run of record's clip and keeps
# every path; another length compiles its own input path into the images, and they go to their
# own directory (domain...-<n>s).
CLIP=${FFAPP_CLIP_SECONDS:-1}
SFX=; [ "$CLIP" = 1 ] || SFX="-${CLIP}s"
INPUT=${FFAPP_INPUT:-/tmp/input$SFX.mkv}
# Includes the common SDK data/stack reservation; CMA must cover the image.
ORDER_CEILING=$(( ${FFAPP_ORDER_CEILING_MB:-256} * 1024 * 1024 ))
# FFAPP_HEAP: which allocator the images link. It is the only thing the arms differ in; the
# FFmpeg libraries are shared (built once, under domain/), so an arm cannot differ by accident
# in anything else.
#   level0  musl-capstone's level0 as every earlier run used it: every heap pointer carries the
#           bounds of the whole arena, and free only marks the block free. That now takes
#           -DCAPSTONE_LEVEL0_OBJECT_BOUNDS=0, since the allocator bounds each object by
#           default.                                                              -> domain/
#   shrink  the same allocator with its default per-object bounds, still no
#           revocation.                                                           -> domain-shrink/
#   sublet  musl-capstone's sublet_heap.c instead of level0: a buddy heap over a LINEAR region the
#           launcher transfers (a program region, parked by hostcall.c's
#           __capstone_region), per-object bounds, and every free revokes.         -> domain-sublet/
#           The arm differs in three objects -- the allocator, hostcall.o (the parking) and the
#           guest host (the grant) -- all of them the heap's delivery, none of them FFmpeg.
HEAP=${FFAPP_HEAP:-level0}
HEAP_REGION=${FFAPP_HEAP_REGION_BYTES:-$((8 * 1024 * 1024))}   # sublet arm: the granted pool
ENTRYF=()
case $HEAP in
  level0) OUT="$WORK/domain"; HEAPF=(-DCAPSTONE_LEVEL0_OBJECT_BOUNDS=0) ;;
  shrink) OUT="$WORK/domain-shrink"; HEAPF=() ;;
  sublet) OUT="$WORK/domain-sublet"; HEAPF=()
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
# The Sublet port of FFmpeg's own pools (modes sublet and stock, ports/ffmpeg/sublet) was removed
# on 2026-10-11: the pools' protection is now the buffer-pool port's patch 0003 (CDERIVE/CREVOKE).
POOL=${FFAPP_POOL:-}
POOL_REGION=${FFAPP_POOL_REGION_BYTES:-$((4 * 1024 * 1024))}
POOLF=(); FFEXTRA=()
if [ -n "$POOL" ]; then
  [ "$HEAP" = sublet ] || { echo "FFAPP_POOL needs FFAPP_HEAP=sublet" >&2; exit 2; }
  case $POOL in 0|2) ;; *) echo "FFAPP_POOL must be 0 or 2" >&2; exit 2 ;; esac
  OUT="$WORK/domain-sublet-pool$POOL"
  POOLF=(-DFFAPP_POOL_MODE="$POOL" -DFFAPP_POOL_REGION_BYTES="${POOL_REGION}UL"
         -I"$APP_DIR/../buffer-pool/src/shared")
fi
OUT="$OUT$SFX"
BASE="$WORK/domain"                  # the shared FFmpeg build and configure stubs
RT="$OUT/runtime"
XB="$BASE/ffmpeg-build"
[ -n "$POOL" ] && XB="$BASE/ffmpeg-build-pool"
mkdir -p "$RT" "$XB"

CLANG=${CAPSTONE_CLANG:?}
LD_LLD=${CAPSTONE_LD_LLD:?}
[ -x "$CLANG" ] || { echo "no clang at $CLANG; from a worktree export CAPSTONE_LLVM_BUILD_DIR=<main clone>/llvm/cmake-build-debug" >&2; exit 2; }
export MUSL_CACHE_ROOT=$WORK/musl-src
mkdir -p "$MUSL_CACHE_ROOT"
if [[ ! -f "$MUSL_CACHE_ROOT/musl-1.2.5.tar.gz" && -f "$CAPSTONE_TMP_ROOT/musl-src/musl-1.2.5.tar.gz" ]]; then
  cp "$CAPSTONE_TMP_ROOT/musl-src/musl-1.2.5.tar.gz" "$MUSL_CACHE_ROOT/"
fi
MUSL=$(bash "$MUSL_PORT/prepare-musl-capstone.sh" | tail -1)
ARCHIVE=$WORK/musl-build/libc-capstone.a
COMPILER_HASH=$(sha256sum "$CLANG" | cut -d' ' -f1)
COMPILER_HASH="$COMPILER_HASH:${CAPSTONE_APPLICATION_PROFILE:-physical}"
if [[ ! -f "$ARCHIVE" || $(cat "$WORK/musl-build/compiler.sha256" 2>/dev/null) != "$COMPILER_HASH" ]]; then
  OUT_DIR=$WORK/musl-build bash "$MUSL_PORT/build-musl-capstone.sh"
  printf '%s\n' "$COMPILER_HASH" > "$WORK/musl-build/compiler.sha256"
fi
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
if [[ ${CAPSTONE_APPLICATION_PROFILE:-physical} == virtual ]]; then
  FLAGS+=(-mllvm -capstone-gp-free -mllvm -capstone-image-gp)
fi

# --- shared application SDK ------------------------------------------------
SDK=$RT/sdk
SDK_HEAP=$HEAP
[[ $HEAP == shrink ]] && SDK_HEAP=level0
SDK_FLAGS="-O1 -DCAPSTONE_SUBLET_HEAP_STATS -DCAPSTONE_LEVEL0_STATS ${HEAPF[*]}"
GRANT=0
[[ $HEAP == sublet ]] && GRANT=$HEAP_REGION
if [[ $POOL == 0 || $POOL == 2 ]]; then GRANT=$((HEAP_REGION + POOL_REGION)); fi
bash "$REPO_ROOT/capstone/ports/common/application/build-sdk.sh" "$SDK" "$MUSL" "$ARCHIVE" \
  -DCAPSTONE_APPLICATION_HEAP="$SDK_HEAP" -DCAPSTONE_APPLICATION_HEAP_LOG=22 \
  -DCAPSTONE_APPLICATION_ARENA_BYTES="$ARENA" -DCAPSTONE_APPLICATION_STACK_BYTES="$STACK" \
  -DCAPSTONE_APPLICATION_GRANT_BYTES="$GRANT" -DCMAKE_C_FLAGS_RELEASE="$SDK_FLAGS"
export CAPSTONE_SDK=$SDK
RUNTIME=()
if [[ $POOL == 0 || $POOL == 2 ]]; then
  "$SDK/capstone-cc" -O1 -DEXP_HEAP_AND_POOL -DPORT_HEAP_REGION_BYTES="${HEAP_REGION}UL" \
    -DPORT_INNER_REGION_BYTES="${POOL_REGION}UL" -I"$REPO_ROOT/capstone/runtime/include" \
    -c "$REPO_ROOT/capstone/ports/common/application/regions.c" -o "$RT/regions.o"
  RUNTIME+=("$RT/regions.o" -Wl,--wrap=__capstone_region)
fi

CONFIGURE_OPTS=(--disable-everything --disable-autodetect --disable-doc --disable-network --disable-asm --disable-inline-asm
  --optflags="${FFAPP_OPT_FLAGS:--O1 -fno-omit-frame-pointer}"
  --disable-pthreads --disable-programs --disable-debug --disable-iconv
  --disable-swresample --disable-swscale --disable-avfilter --disable-avdevice
  --enable-demuxer=matroska --enable-decoder=mpeg4 --enable-parser=mpeg4video
  --enable-protocol=file --enable-static --disable-shared)
# FFAPP_EXTRA_CONFIGURE: further configure options, appended so that they WIN -- FFmpeg's
# configure applies options in order, so "--enable-avfilter" here re-enables what the list
# above disabled. It feeds CONFIG_KEY below, so changing it rebuilds the libraries instead of
# reusing stale ones. Use a separate FFAPP_WORK when probing: the rebuild `rm -rf`s the
# library tree, so probing in the default work directory destroys the baseline build.
if [ -n "${FFAPP_EXTRA_CONFIGURE:-}" ]; then
  read -r -a _ffapp_extra <<< "$FFAPP_EXTRA_CONFIGURE"
  CONFIGURE_OPTS+=("${_ffapp_extra[@]}")
fi
CONFIG_EDIT='s/^#define HAVE_POSIX_MEMALIGN 1$/#define HAVE_POSIX_MEMALIGN 0/; s/^#define HAVE_MEMALIGN 1$/#define HAVE_MEMALIGN 0/'
# Everything that decides what the libraries contain goes into the key, so changing any of it
# rebuilds instead of silently reusing stale libraries (audit, 2026-09-23).
# That includes the compiler. It is a shared-libraries build, so its codegen lives in the
# libLLVM*/libclang* objects clang loads, not in the clang binary; the key takes the size and
# mtime of each (a rebuilt compiler changes them). Before 2026-09-24 it did not, and libraries
# compiled before C-50..C-58 would have been reused on a fixed compiler.
toolchain_id() {
  local bin; bin=$(readlink -f "$CLANG")
  printf '%s\n' "${CAPSTONE_APPLICATION_PROFILE:-physical}"
  # `|| true`: a statically linked clang loads no LLVM libraries, and its codegen is in the binary.
  { echo "$bin"; ldd "$bin" | awk '/=> \//{print $3}' | grep -E 'libLLVM|libclang' || true; } |
    xargs stat -L -c '%n %s %Y'
}
TOOLCHAIN_ID=$(toolchain_id | sha256sum | cut -c1-12)
# --- libvidstab, only when configure is asked for it (Track B, vidstab) -----------------
# vidstabtransform is FFmpeg's wrapper around libvidstab, an external library, so running that
# defect as FFmpeg's real code needs libvidstab in the image: the pinned release
# (deps/libvidstab.json), every source its CMake build compiles, with this image's flags, and
# OpenMP, SSE2 and ORC off, as its CMake leaves them on a target that has none of them.
# configure finds it the way it does on any system, through a pkg-config file, and its
# identity joins the configure key. A configure that does not ask for it is unchanged. The
# header path goes through the compiler's flags, not the .pc file's Cflags: configure hands a
# package's Cflags to its link test too, and the linker here is ld.lld itself, which refuses -I.
VSLIB=(); VIDSTAB_ID=; CFGENV=()
case " ${CONFIGURE_OPTS[*]} " in *" --enable-libvidstab "*)
  read -r VSURL VSSHA VSVER < <(python3 -c '
import json,sys; u=json.load(open(sys.argv[1])); print(u["url"], u["sha256"], u["version"])' "$APP_DIR/deps/libvidstab.json")
  VSTAR="$WORK/vid.stab-$VSVER.tar.gz"
  if [ ! -f "$VSTAR" ]; then
    curl -sSfL --retry 5 --retry-all-errors --retry-delay 3 -o "$VSTAR.part" "$VSURL"
    mv "$VSTAR.part" "$VSTAR"
  fi
  echo "$VSSHA  $VSTAR" | sha256sum -c --quiet - \
    || { echo "libvidstab: $VSTAR does not match deps/libvidstab.json; refusing to build from it" >&2; exit 1; }
  VIDSTAB_ID=$(printf '%s\n' "$VSSHA" "${FLAGS[*]}" "$TOOLCHAIN_ID" | sha256sum | cut -c1-12)
  VSP="$BASE/libvidstab-$VIDSTAB_ID"
  if [ ! -f "$VSP/lib/libvidstab.a" ]; then
    rm -rf "$VSP"; mkdir -p "$VSP/src" "$VSP/obj" "$VSP/lib/pkgconfig" "$VSP/include/vid.stab"
    tar xzf "$VSTAR" -C "$VSP/src" --strip-components=1
    for c in frameinfo transformtype libvidstab transform transformfixedpoint motiondetect \
             motiondetect_opt serialize localmotion2transform boxblur vsvector orc/motiondetectorc; do
      "$CLANG" "${FLAGS[@]}" -std=gnu99 -DDISABLE_ORC -c "$VSP/src/src/$c.c" -o "$VSP/obj/${c##*/}.o" \
        2>> "$VSP/build.log" || { echo "libvidstab: $c.c did not compile; see $VSP/build.log" >&2; exit 1; }
    done
    "${CAPSTONE_LLVM_AR:-$CAPSTONE_LLVM_BIN/llvm-ar}" rcs "$VSP/lib/libvidstab.a.part" "$VSP"/obj/*.o
    cp "$VSP"/src/src/*.h "$VSP/include/vid.stab/"
    mv "$VSP/lib/libvidstab.a.part" "$VSP/lib/libvidstab.a"
  fi
  # Written every time, not cached with the library: it is configure's input, and a stale one
  # survived a change to it once (the first had Cflags -I, which ld.lld refuses).
  printf '%s\n' "prefix=$VSP" 'libdir=${prefix}/lib' 'includedir=${prefix}/include' '' \
    'Name: vidstab' 'Description: vid.stab, built for the capstone domain' "Version: $VSVER" \
    'Libs: -L${libdir} -lvidstab' 'Cflags:' > "$VSP/lib/pkgconfig/vidstab.pc"
  VSLIB=("$VSP/lib/libvidstab.a"); FFEXTRA+=(-I"$VSP/include")
  CFGENV=(env PKG_CONFIG_LIBDIR="$VSP/lib/pkgconfig" PKG_CONFIG_PATH=) ;;
esac
SDK_ID=$(sha256sum "$SDK/capstone-cc" "$SDK/libapplication-runtime.a" | sha256sum | cut -c1-12)
CONFIG_KEY=$(printf '%s\n' "$SRC" "${FLAGS[*]}" "${FFEXTRA[*]}" "${CONFIGURE_OPTS[*]}" "$CONFIG_EDIT" "$TOOLCHAIN_ID" "$SDK_ID" ${VIDSTAB_ID:+"$VIDSTAB_ID"} | sha256sum | cut -c1-12)
# The enabled libraries, read from configure's own config.mak, in static link order (avutil
# last, since everything depends on it). avfilter and swresample appear only when configure
# turned them on, so the default minimal build is unchanged.
ff_libdirs() {
  local m="$XB/ffbuild/config.mak" out=()
  grep -qx 'CONFIG_AVFILTER=yes'    "$m" 2>/dev/null && out+=(libavfilter)
  out+=(libavformat libavcodec)
  grep -qx 'CONFIG_SWRESAMPLE=yes'  "$m" 2>/dev/null && out+=(libswresample)
  grep -qx 'CONFIG_SWSCALE=yes'     "$m" 2>/dev/null && out+=(libswscale)
  out+=(libavutil)
  printf '%s\n' "${out[*]}"
}
if [ ! -f "$XB/libavformat/libavformat.a" ] || [ "$(cat "$XB/.config-key" 2>/dev/null)" != "$CONFIG_KEY" ]; then
  rm -rf "$XB"; mkdir -p "$XB"
  ( cd "$XB" && "${CFGENV[@]}" "$SRC/configure" --enable-cross-compile --cc="$SDK/capstone-cc" --ld="$SDK/capstone-cc" \
      --arch=riscv64 --target-os=none \
      --extra-cflags="${FLAGS[*]}${FFEXTRA[*]:+ ${FFEXTRA[*]}}" --extra-ldflags="" \
      --extra-libs="" "${CONFIGURE_OPTS[@]}" > configure.log 2>&1 ) \
    || { echo "FFmpeg configure failed; see $XB/configure.log and $XB/ffbuild/config.log" >&2; exit 1; }
  # av_malloc -> plain malloc: with asm off ALIGN is 16 (libavutil/mem.c:65), exactly
  # level0's alignment, and musl's posix_memalign sits on an allocator this image does
  # not use.
  sed -i "$CONFIG_EDIT" "$XB/config.h"
  grep -qx '#define HAVE_POSIX_MEMALIGN 0' "$XB/config.h" \
    || { echo "config.h override did not take" >&2; exit 1; }
  # Which libraries to build is CONFIGURE's answer, not a hardcoded list: with libavfilter
  # enabled a hardcoded list configures it and never builds it, so the image silently lacks
  # every filter while the build reports success (found by the 2026-09-25 size probe).
  ff_libdirs > "$OUT/.fflibdirs"
  # shellcheck disable=SC2046
  make -C "$XB" -j"$JOBS" $(sed 's#\([^ ]*\)#\1/\1.a#g' "$OUT/.fflibdirs") \
       > "$XB/build.log" 2>&1 \
    || { echo "FFmpeg domain build failed; see $XB/build.log" >&2; exit 1; }
  echo "$CONFIG_KEY" > "$XB/.config-key"
fi
# `|| true`: zero warnings is a legitimate outcome, and under pipefail grep's exit 1 would
# otherwise stop the build silently at this line (audit, 2026-09-23).
{ grep -hE 'warning: .*\[-Wcapstone-pointer-roundtrip\]' "$XB/build.log" || true; } \
  | sed -E 's#^(src/)?##; s/: warning:.*//' | sort -u > "$OUT/pointer-roundtrip-sites.txt"
FFLIBS=(); for _d in $(ff_libdirs); do FFLIBS+=("$XB/$_d/$_d.a"); done
FFLIBS+=("${VSLIB[@]}")   # after libavfilter, which calls it; libvidstab itself needs only libc
for _l in "${FFLIBS[@]}"; do [ -f "$_l" ] || { echo "configure enabled $(basename "$_l") but it was not built" >&2; exit 1; }; done

# The pool arms' payload allocator and Capstone backend: the buffer-pool port's files,
# unmodified. Compiled after FFmpeg, because libavutil/mem.h needs the generated avconfig.h.
if [ "$POOL" = 0 ] || [ "$POOL" = 2 ]; then
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
print(code_len, alloc, "FITS" if alloc <= ceiling else "DOES-NOT-FIT")' 33554688 "$ORDER_CEILING"
}

for stage in 1 2 3 4 5 6; do   # 6 = M2a, open_input only (bisection stage)
  "$CLANG" "${APPF[@]}" "${ENTRYF[@]}" -DFFAPP_STOP_AT="$stage" -DFFAPP_INPUT="\"$INPUT\"" \
    -c "$APP_DIR/src/capstone-domain/ffapp_domain.c" -o "$OUT/ffapp_domain_m$stage.o"
  "$SDK/capstone-cc" -o "$OUT/ffapp_m$stage.dom" \
    "${RUNTIME[@]}" \
    "$OUT/ffapp_domain_m$stage.o" "$OUT/ffapp_decode.o" "${FFLIBS[@]}" "$ARCHIVE"
  read -r code_len alloc verdict < <(budget "$OUT/ffapp_m$stage.dom")
  printf 'M%d image %s  code_len=%d  allocation=%d  %s\n' "$stage" "$OUT/ffapp_m$stage.dom" \
    "$code_len" "$alloc" "$verdict"
  [ "$verdict" = FITS ] || { echo "BUDGET: M$stage needs a $alloc-byte region, over the configured allocation ceiling" >&2; exit 1; }
done

# DIAGNOSTIC image (M2a with FFAPP_DIAG): a separate decode object, so the production images
# above are byte-identical with or without it. It prints the first bytes a plain fread gets,
# the registered demuxers, the probe's verdict, and avformat_open_input's error.
"$CLANG" "${APPF[@]}" -DFFAPP_DIAG -c "$APP_DIR/src/shared/ffapp_decode.c" -o "$OUT/ffapp_decode_diag.o"
"$SDK/capstone-cc" -o "$OUT/ffapp_m6diag.dom" \
  "${RUNTIME[@]}" \
  "$OUT/ffapp_domain_m6.o" "$OUT/ffapp_decode_diag.o" "${FFLIBS[@]}" "$ARCHIVE"
# Its matched twin: identical except that it reads the input from the 9p share.
"$CLANG" "${APPF[@]}" "${ENTRYF[@]}" -DFFAPP_STOP_AT=6 -DFFAPP_INPUT='"/mnt/host/input.mkv"' \
  -c "$APP_DIR/src/capstone-domain/ffapp_domain.c" -o "$OUT/ffapp_domain_m6_9p.o"
"$SDK/capstone-cc" -o "$OUT/ffapp_m6diag9p.dom" \
  "${RUNTIME[@]}" \
  "$OUT/ffapp_domain_m6_9p.o" "$OUT/ffapp_decode_diag.o" "${FFLIBS[@]}" "$ARCHIVE"
echo "diag images $OUT/ffapp_m6diag.dom ($INPUT) and $OUT/ffapp_m6diag9p.dom (/mnt/host/input.mkv)"

# The M5 POSITIVE CONTROL image: identical except that it decodes the one-byte-flipped
# input. In the same boot as M5 its hashes must DIFFER from the reference, or the domain
# comparison could not have failed and proves nothing (host/compare-md5.py --control).
"$CLANG" "${APPF[@]}" "${ENTRYF[@]}" -DFFAPP_STOP_AT=5 -DFFAPP_INPUT="\"${INPUT%.mkv}.flip.mkv\"" \
  -c "$APP_DIR/src/capstone-domain/ffapp_domain.c" -o "$OUT/ffapp_domain_m5flip.o"
"$SDK/capstone-cc" -o "$OUT/ffapp_m5flip.dom" \
  "${RUNTIME[@]}" \
  "$OUT/ffapp_domain_m5flip.o" "$OUT/ffapp_decode.o" "${FFLIBS[@]}" "$ARCHIVE"
echo "control image $OUT/ffapp_m5flip.dom decodes ${INPUT%.mkv}.flip.mkv"

# --- safety fixtures (src/capstone-domain/ffapp_safety.c) ------------------------------
# One image per fixture: a fault ends the emulator, so a faulting fixture reports nothing else.
# Same runtime, allocator and libraries as the milestone images above; only the entry differs.
FIXTURES="1 2 3 4 5 6 7 8 9 10 16 24 25"   # 16 on the heap arms: the stock control for the pool arms;
                                           # 24 and 25 are two upstream defects live at the pin
[ -n "$POOL" ] && FIXTURES="$(seq -s ' ' 1 17) 20 21"          # the pool fixtures, and the pool-end counts
# FFAPP_EXTRA_FIXTURES: more fixture ids on any arm, e.g. a diagnostic on the level0 heap
FIXTURES="$FIXTURES ${FFAPP_EXTRA_FIXTURES:-}"
# Track B, af_join (fixtures 18 and 19): only when configure built the join filter
# (FFAPP_EXTRA_CONFIGURE="--enable-avfilter --enable-filter=join --enable-decoder=pcm_s16le_planar").
# 18 links libavfilter as built. 19 links af_join.o with upstream's fix 461fb22053 reverted -- one
# token, the dedup loop's bound -- ahead of libavfilter.a, so the archive's copy is never pulled.
# The as-shipped file is compiled here too, with the same command, and must come out
# BYTE-IDENTICAL to the archive's member: that proves the command is the library's own, so the
# reverted object differs from the linked library by that one token and nothing else.
FIXLINK=()
if [ -n "$POOL" ] && [ -f "$XB/libavfilter/af_join.o" ]; then
  FIXTURES="$FIXTURES 18 19 30 31 32 33"   # 30-33: configure-only diagnostics for 18's graph
  AJ=$OUT/afjoin; rm -rf "$AJ"; mkdir -p "$AJ"
  # make's own command for the archive member, as make would run it now (V=1 prints it whole; -s
  # would silence the recipe echo), less its dependency-file flags, which would overwrite make's
  # record for af_join.o with the reruns' inputs.
  AJCMD=$(cd "$XB" && make -n -B V=1 libavfilter/af_join.o 2>/dev/null | grep -F ' -c -o libavfilter/af_join.o ' | tail -1 || true)
  [ -n "$AJCMD" ] || { echo "AF_JOIN GATE: make printed no compile command for libavfilter/af_join.o" >&2; exit 1; }
  AJCMD=$(printf '%s' "$AJCMD" | sed 's/ -MMD -MF [^ ]* -MT [^ ]*//')
  case $AJCMD in *" -MF "*|*" -MMD"*) echo "AF_JOIN GATE: dependency flags left in: $AJCMD" >&2; exit 1 ;; esac
  AJSRC=$(printf '%s\n' $AJCMD | grep -E 'af_join\.c$' | tail -1)
  [ -n "$AJSRC" ] || { echo "AF_JOIN GATE: no af_join.c in make's command: $AJCMD" >&2; exit 1; }
  cp "$SRC/libavfilter/af_join.c" "$AJ/af_join.c"
  python3 - "$AJ/af_join.c" "$AJ/rev/af_join.c" <<'PY'
import os, sys
s = open(sys.argv[1]).read()
old = "        if (j == nb_buffers)\n            s->buffers[nb_buffers++] = buf;\n"
if s.count(old) != 1: sys.exit("af_join: the fixed dedup bound is not there exactly once")
os.makedirs(os.path.dirname(sys.argv[2]), exist_ok=True)
open(sys.argv[2], "w").write(s.replace(old, "        if (j == i)\n            s->buffers[nb_buffers++] = buf;\n"))
PY
  # The shipped object, by make's command with only -o changed; then BOTH copies by the same route
  # the revert must take -- a file outside the tree, its includes found with -I, and __FILE__ mapped
  # to make's spelling -- so the route itself is shown to reproduce the member.
  ( cd "$XB" && eval "${AJCMD/ -o libavfilter\/af_join.o / -o $AJ/af_join.o }" )
  cmp -s "$AJ/af_join.o" "$XB/libavfilter/af_join.o" \
    || { echo "AF_JOIN GATE: make's own command, rerun, does not reproduce libavfilter's af_join.o" >&2; exit 1; }
  ROUTE="-I$SRC/libavfilter -fmacro-prefix-map=$(dirname "$AJ/af_join.c")/=$(dirname "$AJSRC")/"
  ( cd "$XB" && eval "$(printf '%s' "$AJCMD" | sed "s| -o libavfilter/af_join.o | -o $AJ/af_join_shipped.o |; s| $AJSRC\$| $AJ/af_join.c|") $ROUTE" )
  ( cd "$XB" && eval "$(printf '%s' "$AJCMD" | sed "s| -o libavfilter/af_join.o | -o $AJ/af_join_reverted.o |; s| $AJSRC\$| $AJ/rev/af_join.c|") -I$SRC/libavfilter -fmacro-prefix-map=$AJ/rev/=$(dirname "$AJSRC")/" )
  cmp -s "$AJ/af_join_shipped.o" "$XB/libavfilter/af_join.o" \
    || { echo "AF_JOIN GATE: the out-of-tree route does not reproduce libavfilter's af_join.o" >&2; exit 1; }
  diff "$AJ/af_join.c" "$AJ/rev/af_join.c" > "$AJ/revert.diff" || true   # 1 = they differ, as they must
  [ "$(grep -c '^[<>]' "$AJ/revert.diff")" = 2 ] || { echo "AF_JOIN GATE: the revert is not one line" >&2; exit 1; }
  echo "af_join: as-shipped object reproduced byte-identically; fixture 19 links the one-token revert"
fi
# Track B, vidstab (fixtures 22 and 23): only when configure built the vidstabtransform filter
# (FFAPP_EXTRA_CONFIGURE="--enable-avfilter --enable-gpl --enable-libvidstab
#  --enable-filter=vidstabtransform --enable-decoder=yuv4"; libvidstab is built above).
# 22 links libavfilter as built. 23 links vf_vidstabtransform.o with upstream's fix 316531e61c
# reverted: trackb/vidstab-316531e61c.diff, the fix's own diff, applied in reverse at fuzz 0.
# af_join's two gates apply unchanged, and one more: applying the fix to the reverted file must
# give back the shipped file byte for byte, so the revert is exactly the fix's reverse.
if [ -n "$POOL" ] && [ -f "$XB/libavfilter/vf_vidstabtransform.o" ]; then
  FIXTURES="$FIXTURES 22 23"
  VS=$OUT/vidstab; rm -rf "$VS"; mkdir -p "$VS/rev/libavfilter" "$VS/rt/libavfilter"
  VSCMD=$(cd "$XB" && make -n -B V=1 libavfilter/vf_vidstabtransform.o 2>/dev/null | grep -F ' -c -o libavfilter/vf_vidstabtransform.o ' | tail -1 || true)
  [ -n "$VSCMD" ] || { echo "VIDSTAB GATE: make printed no compile command for libavfilter/vf_vidstabtransform.o" >&2; exit 1; }
  VSCMD=$(printf '%s' "$VSCMD" | sed 's/ -MMD -MF [^ ]* -MT [^ ]*//')
  case $VSCMD in *" -MF "*|*" -MMD"*) echo "VIDSTAB GATE: dependency flags left in: $VSCMD" >&2; exit 1 ;; esac
  VSSRC=$(printf '%s\n' $VSCMD | grep -E 'vf_vidstabtransform\.c$' | tail -1)
  [ -n "$VSSRC" ] || { echo "VIDSTAB GATE: no vf_vidstabtransform.c in make's command: $VSCMD" >&2; exit 1; }
  cp "$SRC/libavfilter/vf_vidstabtransform.c" "$VS/vf_vidstabtransform.c"
  cp "$SRC/libavfilter/vf_vidstabtransform.c" "$VS/rev/libavfilter/"
  ( cd "$VS/rev" && patch -s -p1 -R --fuzz=0 --no-backup-if-mismatch < "$APP_DIR/trackb/vidstab-316531e61c.diff" ) \
    || { echo "VIDSTAB GATE: upstream's fix 316531e61c does not reverse at fuzz 0" >&2; exit 1; }
  cp "$VS/rev/libavfilter/vf_vidstabtransform.c" "$VS/rt/libavfilter/"
  ( cd "$VS/rt" && patch -s -p1 --fuzz=0 --no-backup-if-mismatch < "$APP_DIR/trackb/vidstab-316531e61c.diff" ) \
    && cmp -s "$VS/rt/libavfilter/vf_vidstabtransform.c" "$SRC/libavfilter/vf_vidstabtransform.c" \
    || { echo "VIDSTAB GATE: applying the fix to the revert does not give back the shipped file" >&2; exit 1; }
  ( cd "$XB" && eval "${VSCMD/ -o libavfilter\/vf_vidstabtransform.o / -o $VS/vs_make.o }" )
  cmp -s "$VS/vs_make.o" "$XB/libavfilter/vf_vidstabtransform.o" \
    || { echo "VIDSTAB GATE: make's own command, rerun, does not reproduce libavfilter's vf_vidstabtransform.o" >&2; exit 1; }
  ( cd "$XB" && eval "$(printf '%s' "$VSCMD" | sed "s| -o libavfilter/vf_vidstabtransform.o | -o $VS/vs_shipped.o |; s| $VSSRC\$| $VS/vf_vidstabtransform.c|") -I$SRC/libavfilter -fmacro-prefix-map=$VS/=$(dirname "$VSSRC")/" )
  ( cd "$XB" && eval "$(printf '%s' "$VSCMD" | sed "s| -o libavfilter/vf_vidstabtransform.o | -o $VS/vs_reverted.o |; s| $VSSRC\$| $VS/rev/libavfilter/vf_vidstabtransform.c|") -I$SRC/libavfilter -fmacro-prefix-map=$VS/rev/libavfilter/=$(dirname "$VSSRC")/" )
  cmp -s "$VS/vs_shipped.o" "$XB/libavfilter/vf_vidstabtransform.o" \
    || { echo "VIDSTAB GATE: the out-of-tree route does not reproduce libavfilter's vf_vidstabtransform.o" >&2; exit 1; }
  diff "$VS/vf_vidstabtransform.c" "$VS/rev/libavfilter/vf_vidstabtransform.c" > "$VS/revert.diff" || true
  echo "vidstab: as-shipped object reproduced byte-identically; fixture 23 links upstream's fix reversed ($(grep -c '^[<>]' "$VS/revert.diff") changed lines)"
fi
# Each Track B pair links its filter's object AHEAD of the libraries on BOTH sides, the shipped
# one for 18 and 22, so the two images of a pair differ in that object's bytes and the fixture id
# alone. Linking it ahead for the reverted side only moved every later symbol too (audit,
# 2026-09-29: 10,062 symbol lines differed between the first 18 and 19 images).
for fx in $FIXTURES; do
  FIXLINK=()
  case $fx in
    18) FIXLINK=("$AJ/af_join_shipped.o") ;;  19) FIXLINK=("$AJ/af_join_reverted.o") ;;
    22) FIXLINK=("$VS/vs_shipped.o") ;;       23) FIXLINK=("$VS/vs_reverted.o") ;;
  esac
  # FFAPP_LINK_AHEAD_<n>: objects linked ahead of the libraries for fixture <n> only (diagnostics)
  _ahead=FFAPP_LINK_AHEAD_$fx; [ -n "${!_ahead:-}" ] && FIXLINK+=(${!_ahead})
  "$CLANG" "${APPF[@]}" -DFFAPP_FIXTURE="$fx" \
    -c "$APP_DIR/src/capstone-domain/ffapp_safety.c" -o "$OUT/ffapp_safety_$fx.o"
  "$SDK/capstone-cc" -o "$OUT/ffapp_fx$fx.dom" \
    "${RUNTIME[@]}" \
    "$OUT/ffapp_safety_$fx.o" "${FIXLINK[@]}" "${FFLIBS[@]}" "$ARCHIVE"
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

# Every fixture uses the same descriptor and delegated runtime as the decoder.
for img in "$OUT"/ffapp_m*.dom "$OUT"/ffapp_fx*.dom; do
  read -r code_len alloc verdict < <(budget "$img")
  [[ $verdict == FITS ]] || { echo "image exceeds the declared allocation budget: $img" >&2; exit 1; }
done
printf 'delegated application and fixtures: %s\n' "$OUT"
