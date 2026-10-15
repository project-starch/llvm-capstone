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
# FFAPP_HEAP selects physical-profile heaps only; virtual always links mallocng.
# It is the only thing the physical arms differ in; the
# FFmpeg libraries are shared (built once, under domain/), so an arm cannot differ by accident
# in anything else.
#   level0  musl-capstone's level0 as every earlier run used it: every heap pointer carries the
#           bounds of the whole arena, and free only marks the block free. That now takes
#           -DCAPSTONE_LEVEL0_OBJECT_BOUNDS=0, since the allocator bounds each object by
#           default.                                                              -> domain/
#   shrink  the same allocator with its default per-object bounds, still no
#           revocation.                                                           -> domain-shrink/
# The sublet arm (musl-capstone's sublet_heap.c) and the pool modes on it (FFAPP_POOL=0|2) were
# removed on 2026-10-11 with the physical Sublet heap; their recorded results stay in results/.
if [[ ${CAPSTONE_APPLICATION_PROFILE:-virtual} == virtual ]]; then
  [[ ! ${FFAPP_HEAP+x} ]] || { echo "FFAPP_HEAP is physical-only; virtual uses mallocng" >&2; exit 2; }
  HEAP=mallocng OUT="$WORK/domain" HEAPF=()
else
  HEAP=${FFAPP_HEAP:-level0}
  case $HEAP in
    level0) OUT="$WORK/domain"; HEAPF=(-DCAPSTONE_LEVEL0_OBJECT_BOUNDS=0) ;;
    shrink) OUT="$WORK/domain-shrink"; HEAPF=() ;;
    *) echo "FFAPP_HEAP must be level0 or shrink" >&2; exit 2 ;;
  esac
fi
FFEXTRA=()
# FFAPP_SUBLET=1 (virtual profile only): FFmpeg's own pools protected by the buffer-pool port's
# patch 0003, every AVBufferPool and AVRefStructPool entry handed out as a Sublet child lifetime
# and revoked when it returns to its pool. The libraries are built from prepare-source.sh
# --sublet, and capability.h's include path joins the configure key, so they rebuild.
SUBLET=${FFAPP_SUBLET:-0}
case $SUBLET in 0|1) ;; *) echo "FFAPP_SUBLET must be 0 or 1" >&2; exit 2 ;; esac
if [[ $SUBLET == 1 ]]; then
  [[ ${CAPSTONE_APPLICATION_PROFILE:-virtual} == virtual ]] \
    || { echo "FFAPP_SUBLET=1 needs CAPSTONE_APPLICATION_PROFILE=virtual" >&2; exit 2; }
  FFEXTRA+=(-I"$REPO_ROOT/capstone/runtime/include")
fi
OUT="$OUT$SFX"
BASE="$WORK/domain"                  # the shared FFmpeg build and configure stubs
RT="$OUT/runtime"
XB="$BASE/ffmpeg-build"
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
# The key also covers musl's compile flags (survey-musl-capstone.py: the virtual
# profile's F/D code generation), as the hard-float SDK needs.
LIBC_KEY=$({
  sha256sum "$CLANG"
  printf '%s\n' "${CAPSTONE_APPLICATION_PROFILE:-virtual}"
  cat "$MUSL_PORT"/{build-musl-capstone,prepare-musl-capstone}.sh "$MUSL_PORT/survey-musl-capstone.py" \
      "$MUSL_PORT"/musl-patches/*.patch
  find "$MUSL_PORT/arch-capstone64" -type f -print0 | sort -z | xargs -0 cat
} | sha256sum | cut -d' ' -f1)
if [[ ! -f "$ARCHIVE" || $(cat "$WORK/musl-build/source.sha256" 2>/dev/null) != "$LIBC_KEY" ]]; then
  OUT_DIR=$WORK/musl-build bash "$MUSL_PORT/build-musl-capstone.sh"
  printf '%s\n' "$LIBC_KEY" > "$WORK/musl-build/source.sha256"
fi
if [[ $SUBLET == 1 ]]; then
  SRC=$(bash "$SCRIPT_DIR/prepare-source.sh" --sublet | tail -1)
else
  SRC=$(bash "$SCRIPT_DIR/prepare-source.sh" | tail -1)
fi

TARGET=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
        -Xclang -target-feature -Xclang +a)
INC=(-nostdinc -isystem "$MUSL/arch/capstone64" -isystem "$MUSL/arch/generic"
     -isystem "$MUSL/obj/include" -isystem "$MUSL/include" -isystem "$("$CLANG" -print-resource-dir)/include")
FLAGS=("${TARGET[@]}" -ffreestanding -fno-builtin -fno-jump-tables -ffunction-sections
       -fdata-sections -O1 -Wno-int-conversion -D_GNU_SOURCE "${INC[@]}")
if [[ ${CAPSTONE_APPLICATION_PROFILE:-virtual} == virtual ]]; then
  FLAGS+=(-mllvm -capstone-gp-free -mllvm -capstone-image-gp
          -Xclang -target-feature -Xclang +f -Xclang -target-feature -Xclang +d
          -Xclang -target-abi -Xclang lp64)
fi

# --- shared application SDK ------------------------------------------------
SDK=$RT/sdk
SDK_FLAGS=-O1
[[ $HEAP == mallocng ]] || SDK_FLAGS+=" -DCAPSTONE_LEVEL0_STATS ${HEAPF[*]}"
bash "$REPO_ROOT/capstone/ports/common/application/build-sdk.sh" "$SDK" "$MUSL" "$ARCHIVE" \
  -DCAPSTONE_APPLICATION_ARENA_BYTES="$ARENA" -DCAPSTONE_APPLICATION_STACK_BYTES="$STACK" \
  -DCMAKE_C_FLAGS_RELEASE="$SDK_FLAGS"
export CAPSTONE_SDK=$SDK

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
  printf '%s\n' "${CAPSTONE_APPLICATION_PROFILE:-virtual}"
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

# --- the program --------------------------------------------------------------------
APPF=("${FLAGS[@]}" -I"$XB" -I"$SRC" -I"$APP_DIR/src/shared")
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
  "$CLANG" "${APPF[@]}" -DFFAPP_STOP_AT="$stage" -DFFAPP_INPUT="\"$INPUT\"" \
    -c "$APP_DIR/src/capstone-domain/ffapp_domain.c" -o "$OUT/ffapp_domain_m$stage.o"
  "$SDK/capstone-cc" -o "$OUT/ffapp_m$stage.dom" \
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
  "$OUT/ffapp_domain_m6.o" "$OUT/ffapp_decode_diag.o" "${FFLIBS[@]}" "$ARCHIVE"
# Its matched twin: identical except that it reads the input from the 9p share.
"$CLANG" "${APPF[@]}" -DFFAPP_STOP_AT=6 -DFFAPP_INPUT='"/mnt/host/input.mkv"' \
  -c "$APP_DIR/src/capstone-domain/ffapp_domain.c" -o "$OUT/ffapp_domain_m6_9p.o"
"$SDK/capstone-cc" -o "$OUT/ffapp_m6diag9p.dom" \
  "$OUT/ffapp_domain_m6_9p.o" "$OUT/ffapp_decode_diag.o" "${FFLIBS[@]}" "$ARCHIVE"
echo "diag images $OUT/ffapp_m6diag.dom ($INPUT) and $OUT/ffapp_m6diag9p.dom (/mnt/host/input.mkv)"

# The M5 POSITIVE CONTROL image: identical except that it decodes the one-byte-flipped
# input. In the same boot as M5 its hashes must DIFFER from the reference, or the domain
# comparison could not have failed and proves nothing (host/compare-md5.py --control).
"$CLANG" "${APPF[@]}" -DFFAPP_STOP_AT=5 -DFFAPP_INPUT="\"${INPUT%.mkv}.flip.mkv\"" \
  -c "$APP_DIR/src/capstone-domain/ffapp_domain.c" -o "$OUT/ffapp_domain_m5flip.o"
"$SDK/capstone-cc" -o "$OUT/ffapp_m5flip.dom" \
  "$OUT/ffapp_domain_m5flip.o" "$OUT/ffapp_decode.o" "${FFLIBS[@]}" "$ARCHIVE"
echo "control image $OUT/ffapp_m5flip.dom decodes ${INPUT%.mkv}.flip.mkv"

# --- safety fixtures (src/capstone-domain/ffapp_safety.c) ------------------------------
# One image per fixture: a fault ends the emulator, so a faulting fixture reports nothing else.
# Same runtime, allocator and libraries as the milestone images above; only the entry differs.
FIXTURES="1 2 3 4 5 6 7 8 9 10 16 24 25"   # 16: stock refstruct's in-band header;
                                           # 24 and 25 are two upstream defects live at the pin
# FFAPP_EXTRA_FIXTURES: more fixture ids on any arm, e.g. a diagnostic on the level0 heap
FIXTURES="$FIXTURES ${FFAPP_EXTRA_FIXTURES:-}"
for fx in $FIXTURES; do
  "$CLANG" "${APPF[@]}" -DFFAPP_FIXTURE="$fx" \
    -c "$APP_DIR/src/capstone-domain/ffapp_safety.c" -o "$OUT/ffapp_safety_$fx.o"
  "$SDK/capstone-cc" -o "$OUT/ffapp_fx$fx.dom" \
    "$OUT/ffapp_safety_$fx.o" "${FFLIBS[@]}" "$ARCHIVE"
done
echo "safety fixture images ($HEAP heap): $(ls "$OUT"/ffapp_fx*.dom | wc -l)"

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
