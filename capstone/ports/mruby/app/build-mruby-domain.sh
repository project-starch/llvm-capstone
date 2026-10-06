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
# MRBD_HEAP picks the domain's malloc, which mruby's mrb_malloc sits on:
# level0 (default; every pointer carries the arena's bounds, free only marks) or
# sublet (runtime/sublet_heap.c: a buddy heap over a region the host grants, one
# bounded alias per block, every free revokes). The sublet arm differs in three
# runtime objects -- the heap, hostcall.o (which parks the grant) and the entry
# (which reports the heap's counters) -- and in the host (the grant); mruby is
# the same. sublet-gc is sublet plus every GC object slot issued and revoked on
# its own (patches/4.0.0-rc2/0008, MRB_CAPSTONE_GC_SUBLET): the GC carves its
# pages from a 32 MiB pool split from the descriptor's combined grant.
# MRBD_HEAP_LOG sets its pool, 2^26 = 64 MiB by default; the host must
# grant twice that (requested by the descriptor), since a CMA
# region is only 1 MiB-aligned and the pool is a self-aligned block inside it.
# Knobs (see build_config.rb): MRBD_BOXING, MRBD_DISPATCH, MRBD_OPT, MRBD_TESTS,
# MRBD_DEFINES. MRBD_FROM=runtime|mruby starts at that stage. MRUBY_MIRROR=<a
# local mruby clone> clones from there instead of GitHub (its Prism submodule too).
# MRBD_SDK=<existing application SDK> reuses its libc and runtime profile.
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
HEAP=${MRBD_HEAP:-level0}
HEAP_LOG=${MRBD_HEAP_LOG:-26}
case "$HEAP" in level0|sublet|sublet-gc) ;; *) echo "MRBD_HEAP=$HEAP? (level0, sublet, sublet-gc)" >&2; exit 2 ;; esac
if [[ $HEAP == sublet-gc ]]; then
  [[ -f "$SCRIPT_DIR/patches/$PIN/0008-gc-slots-under-sublet.patch" ]] \
    || { echo "MRBD_HEAP=sublet-gc needs patches/$PIN/0008-gc-slots-under-sublet.patch" >&2; exit 2; }
  export MRBD_GC_SUBLET_INCLUDE=$RT/capstone/sublet
fi

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
if [[ -n ${MRBD_SDK:-} ]]; then
  O=$(cd -- "$MRBD_SDK" && pwd)
  [[ -f "$O/sdk.json" && -x "$O/capstone-cc" ]] \
    || { echo "MRBD_SDK is not an application SDK" >&2; exit 2; }
else
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

# ---- shared delegated application SDK --------------------------------------
O=$ROOT/runtime
SDK_HEAP=$HEAP
[[ $HEAP == sublet-gc ]] && SDK_HEAP=sublet
if stage runtime; then
  EXTRA=()
  if [[ $HEAP == sublet-gc ]]; then
    EXTRA=(-DCAPSTONE_APPLICATION_GRANT_BYTES="$(( (2 << HEAP_LOG) + (32 << 20) ))")
  fi
  bash "$RT/capstone/ports/common/application/build-sdk.sh" "$O" "$MUSL" "$ARCHIVE" \
    -DCAPSTONE_APPLICATION_HEAP="$SDK_HEAP" -DCAPSTONE_APPLICATION_HEAP_LOG="$HEAP_LOG" \
    -DCAPSTONE_APPLICATION_ARENA_BYTES="$ARENA" "${EXTRA[@]}"
  printf '%s\n' "$HEAP" > "$O/.heap"
fi
[[ $(cat "$O/.heap" 2>/dev/null) == "$HEAP" && -x "$O/capstone-cc" ]] \
  || { echo "rebuild the application SDK (MRBD_FROM=runtime)" >&2; exit 2; }
fi
export CAPSTONE_SDK=$O
export PATH=$O:$CAPSTONE_LLVM_BIN:$PATH
export LLVM_AR=$CAPSTONE_LLVM_BIN/llvm-ar
"$O/capstone-cc" --check-toolchain
"$O/capstone-cc" -O1 -c "$SCRIPT_DIR/spawn-shell.c" -o "$O/spawn-shell.o"
export MRBD_SPAWN_OBJECT=$O/spawn-shell.o
# The GC adapter takes region 1; the common application descriptor grants one pool.
if [[ $HEAP == sublet-gc ]]; then
  "$O/capstone-cc" -O1 -I"$RT/capstone/runtime/include" -DEXP_HEAP_AND_POOL \
    -DPORT_HEAP_REGION_BYTES="$((2 << HEAP_LOG))UL" -DPORT_INNER_REGION_BYTES=33554432UL \
    -c "$RT/capstone/ports/common/application/regions.c" -o "$O/regions.o"
  export MRBD_REGION_OBJECT=$O/regions.o
fi

# ---- mruby, pinned and patched ----------------------------------------------
M=$ROOT/src/mruby
PATCH_HASH=$(sha256sum "$PATCHES"/*.patch | sha256sum | cut -d' ' -f1)
if [[ -d "$M/.git" && $(cat "$M/.mrbd-patchset" 2>/dev/null) != "$PATCH_HASH" ]]; then
  echo "patch set changed; use a fresh MRBD_ROOT" >&2; exit 2
fi
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
  printf '%s\n' "$PATCH_HASH" > "$M/.mrbd-patchset"
fi
[[ $(cat "$M/.mrbd-boxing") == "$PIN $BOXING" ]] \
  || { echo "the tree at $M was patched for MRBD_PIN MRBD_BOXING = $(cat "$M/.mrbd-boxing"); remove it to rebuild" >&2; exit 2; }

if stage mruby; then
  COMPILER_HASH=$(sha256sum "$CAPSTONE_CLANG" | cut -d' ' -f1)
  if [[ $(cat "$ROOT/.compiler-sha256" 2>/dev/null) != "$COMPILER_HASH" ]]; then
    rm -rf "$M/build/capstone"
  fi
  export CAPSTONE_COMPILE_LOG=$ROOT/objects.tsv
  : > "$CAPSTONE_COMPILE_LOG"
  # Rake does not track the SDK archive as an upstream dependency.
  rm -f "$M/build/capstone/bin/mruby" "$M/build/capstone/bin/mrbtest"
  targets=(all)
  [[ ${MRBD_TESTS:-0} == 1 ]] && targets+=(test:build:lib)
  (cd "$M" && MRUBY_CONFIG="$SCRIPT_DIR/build_config.rb" rake -j"$JOBS" "${targets[@]}") \
    > "$ROOT/rake.log" 2>&1 || { tail -5 "$ROOT/rake.log" >&2; echo "rake failed (see $ROOT/rake.log)" >&2; exit 2; }
  failed=$(awk -F'\t' '$2 != 0' "$CAPSTONE_COMPILE_LOG" | wc -l)
  printf '%s\n' "$COMPILER_HASH" > "$ROOT/.compiler-sha256"
  log "capstone objects compiled: $(wc -l < "$CAPSTONE_COMPILE_LOG"), failed: $failed"
fi
ls -la "$M/build/capstone/bin/" "$M/build/native/bin/" 2>/dev/null | awk '/mruby|mrbtest/ {print "[build-mruby] " $NF}'
