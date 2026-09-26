#!/usr/bin/env bash
# mruby as a Capstone musl domain: musl and the port runtime from a given
# llvm-capstone tree, mruby at a pinned commit with this port's patches, and
# rake's cross build for the capstone target, linked as a domain. The same
# configuration is built natively as the reference.
#
#   CAPSTONE_LLVM_BUILD_DIR=<llvm build> [RUNTIME_REPO=<llvm-capstone tree>] \
#     bash build-mruby-domain.sh
#
# Outputs, under $MRBD_ROOT (default $CAPSTONE_TMP_ROOT/mruby-domain):
#   src/mruby/build/capstone/bin/mruby    the interpreter, a domain image
#   src/mruby/build/capstone/bin/mrbtest  mruby's test suite, with MRBD_TESTS=1
#   src/mruby/build/native/bin/{mruby,mrbtest}   the same, natively
#
# MRBD_PIN picks the mruby: head (default, 2026-09-17, every known defect fixed)
# or 4.0.0-rc2 (2026-03-12, the Sublet evaluation's pin: temporal defects on
# mruby's own allocators, fixed later). Each pin has its own patches/<pin>/.
# Knobs (see build_config.rb): MRBD_BOXING, MRBD_DISPATCH, MRBD_OPT, MRBD_TESTS,
# MRBD_DEFINES. MRBD_FROM=runtime|mruby starts at that stage. MRUBY_MIRROR=<a
# local mruby clone> clones from there instead of GitHub (its Prism submodule too).
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../../tests/capstone-test-env.sh" >/dev/null
ROOT=${MRBD_ROOT:-$CAPSTONE_TMP_ROOT/mruby-domain}
RT=${RUNTIME_REPO:-$CAPSTONE_REPO_ROOT}
MUSL_PORT=$RT/capstone/ports/musl-capstone
MRT=$MUSL_PORT/runtime
ARENA=${MRBD_ARENA_BYTES:-$((64 * 1024 * 1024))}
JOBS=${JOBS:-16}
FROM=${MRBD_FROM:-all}
BOXING=${MRBD_BOXING:-no}
PIN=${MRBD_PIN:-head}

# The pinned trees. head: mruby of 2026-09-17 and the Prism its .gitmodules
# names. 4.0.0-rc2: the tag's commit, whose parser is still parse.y (no Prism).
MRUBY_URL=https://github.com/mruby/mruby.git
case "$PIN" in
  head)      MRUBY_COMMIT=ad98f216eb472202c8e5deece5ea13655d9f7969
             PRISM_COMMIT=c0e37816e97e23e92524a4070e1b99a4025bc63f ;;
  4.0.0-rc2) MRUBY_COMMIT=9d523e2f74f2e63ca02840937523de61398a617d
             PRISM_COMMIT= ;;
  *) echo "MRBD_PIN=$PIN? (head, 4.0.0-rc2)" >&2; exit 2 ;;
esac
PATCHES=$SCRIPT_DIR/patches/$PIN

[[ -f "$MRT/hostcall.c" ]] || { echo "no runtime at $MRT (RUNTIME_REPO=$RT)" >&2; exit 2; }
mkdir -p "$ROOT/runtime" "$ROOT/src"
log() { echo "[build-mruby] $*"; }
stage() {
  case "$FROM" in all) return 0;; runtime) [[ $1 != musl ]];; mruby) [[ $1 == mruby ]];;
    *) echo "MRBD_FROM=$FROM?" >&2; exit 2;; esac
}

# ---- musl and its archive, private to this build --------------------------
export MUSL_CACHE_ROOT=$ROOT/musl-src
mkdir -p "$MUSL_CACHE_ROOT"
[[ -f "$MUSL_CACHE_ROOT/musl-1.2.5.tar.gz" || ! -f "$CAPSTONE_TMP_ROOT/musl-src/musl-1.2.5.tar.gz" ]] \
  || cp "$CAPSTONE_TMP_ROOT/musl-src/musl-1.2.5.tar.gz" "$MUSL_CACHE_ROOT/"
MUSL=$(bash "$MUSL_PORT/prepare-musl-capstone.sh" | tail -1)
if stage musl; then
  OUT_DIR=$ROOT/musl-build bash "$MUSL_PORT/build-musl-capstone.sh" >/dev/null
fi
ARCHIVE=$ROOT/musl-build/libc-capstone.a
[[ -f "$ARCHIVE" ]] || { echo "no $ARCHIVE" >&2; exit 2; }
log "musl $MUSL, archive $ARCHIVE"

# ---- the runtime, as the PostgreSQL port builds it --------------------------
INC=(-nostdinc -isystem "$MUSL/arch/capstone64" -isystem "$MUSL/arch/generic"
     -isystem "$MUSL/obj/include" -isystem "$MUSL/include")
CF=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
    -Xclang -target-feature -Xclang +a -ffreestanding -fno-builtin -fno-jump-tables
    -ffunction-sections -fdata-sections -O1 -w -Wno-int-conversion "${INC[@]}")
RF=("${CF[@]}" -std=c99 -D_XOPEN_SOURCE=700
    -I"$MUSL/src/include" -I"$MUSL/src/internal" -I"$MUSL/obj/src/internal")
O=$ROOT/runtime
if stage runtime; then
  rm -f "$O"/*.o
  for s in start-musl set_thread_area setjmp; do
    "$CAPSTONE_CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
      -ffreestanding -O0 -c "$MRT/$s.S" -o "$O/$s.o"
  done
  for f in hostcall tls; do
    "$CAPSTONE_CLANG" "${RF[@]}" -c "$MRT/$f.c" -o "$O/$f.o"
  done
  "$CAPSTONE_CLANG" "${RF[@]}" -DCAPSTONE_LEVEL0_ARENA_BYTES="($ARENA)" -c "$MRT/level0.c" -o "$O/level0.o"
  source "$MRT/libc_overrides.sh"
  build_musl_overrides "$CAPSTONE_CLANG" "$O" "$MUSL" "${RF[@]}"
  CLANG=$CAPSTONE_CLANG OBJ_DIR=$O COMPILER_RT=$RT/compiler-rt/lib/builtins
  COMMON_FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
                -ffreestanding -fno-builtin -ffunction-sections -fdata-sections -O1 -w)
  source "$RT/capstone/benchmarks/beebs/build-beebs-softfloat-common.sh"
  "$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -O1 -c "$SCRIPT_DIR/toolchain/domain_entry.c" -o "$O/domain_entry.o"
  log "runtime: $(ls "$O"/*.o | wc -l) objects from $MRT, level0 arena $ARENA bytes"
fi

# ---- the compiler and linker rake is given ----------------------------------
export MRBD_MUSL=$MUSL MRBD_RUNTIME_DIR=$O MRBD_LIBC_ARCHIVE=$ARCHIVE
export MRBD_LINKER_SCRIPT=$RT/capstone/my_first_domain/link.ld
export PATH=$SCRIPT_DIR/toolchain:$PATH
export LLVM_AR=$CAPSTONE_LLVM_BIN/llvm-ar

# ---- mruby, pinned and patched ----------------------------------------------
M=$ROOT/src/mruby
if [[ ! -d "$M/.git" ]]; then
  git clone -q "${MRUBY_MIRROR:-$MRUBY_URL}" "$M"
  git -C "$M" checkout -q "$MRUBY_COMMIT"
  if [[ -n $PRISM_COMMIT ]]; then
    if [[ -n ${MRUBY_MIRROR:-} ]]; then
      git -C "$M" config submodule.mrbgems/mruby-compiler/lib/prism.url "$MRUBY_MIRROR/mrbgems/mruby-compiler/lib/prism"
    fi
    # A mirror is a local path, which git refuses as a submodule source by default.
    git -C "$M" ${MRUBY_MIRROR:+-c protocol.file.allow=always} submodule update -q --init mrbgems/mruby-compiler/lib/prism
    [[ $(git -C "$M/mrbgems/mruby-compiler/lib/prism" rev-parse HEAD) == "$PRISM_COMMIT" ]] \
      || { echo "Prism is not at $PRISM_COMMIT" >&2; exit 2; }
  fi
  if [[ $BOXING == word ]] && ! ls "$PATCHES"/0006-*.patch >/dev/null 2>&1; then
    echo "MRBD_BOXING=word needs a 0006 patch for $PIN, and $PATCHES has none" >&2; exit 2
  fi
  for p in "$PATCHES"/*.patch; do
    case $(basename "$p") in 0006-*) [[ $BOXING == word ]] || continue ;; esac
    (cd "$M" && patch -p1 -s < "$p") || { echo "patch $p did not apply" >&2; exit 2; }
    log "applied $(basename "$p")"
  done
  echo "$PIN $BOXING" > "$M/.mrbd-boxing"
fi
[[ $(cat "$M/.mrbd-boxing") == "$PIN $BOXING" ]] \
  || { echo "the tree at $M was patched for MRBD_PIN MRBD_BOXING = $(cat "$M/.mrbd-boxing"); remove it to rebuild" >&2; exit 2; }

if stage mruby; then
  export MRBD_SURVEY_LOG=$ROOT/objects.tsv
  : > "$MRBD_SURVEY_LOG"
  targets=(all)
  [[ ${MRBD_TESTS:-0} == 1 ]] && targets+=(test:build:lib)
  (cd "$M" && MRUBY_CONFIG="$SCRIPT_DIR/build_config.rb" rake -j"$JOBS" "${targets[@]}") \
    > "$ROOT/rake.log" 2>&1 || { tail -5 "$ROOT/rake.log" >&2; echo "rake failed (see $ROOT/rake.log)" >&2; exit 2; }
  failed=$(awk -F'\t' '$2 != 0' "$MRBD_SURVEY_LOG" | wc -l)
  log "capstone objects compiled: $(wc -l < "$MRBD_SURVEY_LOG"), failed: $failed"
fi
ls -la "$M/build/capstone/bin/" "$M/build/native/bin/" 2>/dev/null | awk '/mruby|mrbtest/ {print "[build-mruby] " $NF}'
