# Sourced by dependency recipes: one musl archive and the delegated application SDK.
# Adapted from ports/wireshark/app/deps/env.sh (same controls), for memcached's libevent.
MC_DEPS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$MC_DEPS_DIR/../../../../tests/capstone-test-env.sh"
MC_WORK=${MC_WORK:-$CAPSTONE_TMP_ROOT/memcached-app}
export MC_DEPS_SRC=$MC_WORK/deps-src MC_DEPS_BUILD=$MC_WORK/deps-build MC_DEPS_PREFIX=$MC_WORK/deps-cap
mkdir -p "$MC_DEPS_SRC" "$MC_DEPS_BUILD" "$MC_DEPS_PREFIX/lib" "$MC_DEPS_PREFIX/include"
_mc_musl_port="$CAPSTONE_REPO_ROOT/capstone/ports/musl-capstone"

export MUSL_CACHE_ROOT=$MC_WORK/musl-src
mkdir -p "$MUSL_CACHE_ROOT"
export MC_MUSL; MC_MUSL=$(bash "$_mc_musl_port/prepare-musl-capstone.sh" | tail -1)
export MC_LINKER_SCRIPT="$CAPSTONE_REPO_ROOT/capstone/my_first_domain/link.ld"

# The key: the compiler (size and mtime of clang and the LLVM libraries it loads; a shared-libs
# build keeps codegen in libLLVM*.so), every source the runtime is built from, and this file.
_mc_sources=$(capstone_source_listing capstone/runtime capstone/ports/common/application) || return 1
_mc_key=$( {
  printf '%s\n' "${CAPSTONE_APPLICATION_PROFILE:-physical}"
  _b=$(readlink -f "$CAPSTONE_CLANG")
  { echo "$_b"; ldd "$_b" | awk '/=> \//{print $3}' | grep -E 'libLLVM|libclang' || true; } | xargs stat -L -c '%n %s %Y'
  cat "$_mc_musl_port"/runtime/*.c "$_mc_musl_port"/runtime/*.S "$_mc_musl_port/runtime/libc_overrides.list" \
      "$MC_DEPS_DIR/capstone-cc" "$MC_DEPS_DIR/env.sh"
  # capstone_source_listing, not a bare git ls-files: it falls back to find outside a git
  # repository and FAILS rather than printing nothing, so the key can never be computed
  # with no runtime source in it. Resolved before the digest, because a failure inside
  # this pipeline would be invisible.
  printf '%s\n' "$_mc_sources" | while IFS= read -r path; do cat "$CAPSTONE_REPO_ROOT/$path"; done
  cat "$CAPSTONE_REPO_ROOT/capstone/ports/common/application/build-sdk.sh"
  cat "$_mc_musl_port"/musl-patches/*.patch
  find "$_mc_musl_port/arch-capstone64" -type f -print0 | sort -z | xargs -0 cat
} | sha256sum | cut -c1-16)

# One libc and one runtime per key, never rebuilt in place: a build that sourced this file keeps
# the directories it started with while another prepares a new key. (Rebuilding in place raced
# once: a link ran against a runtime directory that was half written, 2026-09-24.)
export MC_LIBC_ARCHIVE=$MC_WORK/musl-capstone-build-$_mc_key/libc-capstone.a
export MC_RUNTIME_DIR=$MC_WORK/deps-runtime-$_mc_key
export CAPSTONE_SDK=$MC_RUNTIME_DIR
exec {_mc_lock}> "$MC_WORK/.env.lock"; flock "$_mc_lock"
if [[ ! -f "$MC_RUNTIME_DIR/.key" || ! -f "$MC_LIBC_ARCHIVE" ]]; then
  echo "deps/env.sh: preparing libc and runtime (key $_mc_key)" >&2
  OUT_DIR=$MC_WORK/musl-capstone-build-$_mc_key bash "$_mc_musl_port/build-musl-capstone.sh" >&2 || return 1
  rm -rf "$MC_RUNTIME_DIR"; mkdir -p "$MC_RUNTIME_DIR"
  bash "$CAPSTONE_REPO_ROOT/capstone/ports/common/application/build-sdk.sh" \
    "$MC_RUNTIME_DIR" "$MC_MUSL" "$MC_LIBC_ARCHIVE" >&2 || return 1

  # capstone-cc, both directions.
  _mc_c=$(mktemp -d)
  printf '#include <stdio.h>\nint main(void){return puts("x");}\n' > "$_mc_c/ok.c"
  printf 'int no_such_function_in_any_libc(void);\nint main(void){return no_such_function_in_any_libc();}\n' > "$_mc_c/bad.c"
  "$MC_DEPS_DIR/capstone-cc" -o "$_mc_c/ok" "$_mc_c/ok.c" 2> "$_mc_c/ok.err" ||
    { echo "deps/env.sh: CONTROL FAILED: a program calling puts() does not link:" >&2; cat "$_mc_c/ok.err" >&2; return 1; }
  if "$MC_DEPS_DIR/capstone-cc" -o "$_mc_c/bad" "$_mc_c/bad.c" 2>/dev/null; then
    echo "deps/env.sh: CONTROL FAILED: an undefined function links; configure would report everything present" >&2; return 1
  fi
  if "$MC_DEPS_DIR/capstone-cc" -o "$_mc_c/badl" "$_mc_c/ok.c" -lno_such_library_anywhere 2>/dev/null; then
    echo "deps/env.sh: CONTROL FAILED: -l of a missing library links" >&2; return 1
  fi
  rm -rf "$_mc_c"
  echo "$_mc_key" > "$MC_RUNTIME_DIR/.key"
  echo "deps/env.sh: controls pass (links puts; refuses an undefined function and a missing -l)" >&2
fi
flock -u "$_mc_lock"; exec {_mc_lock}>&-

export CC="$MC_DEPS_DIR/capstone-cc"
export AR="$CAPSTONE_LLVM_BIN/llvm-ar"
export RANLIB="$MC_DEPS_DIR/capstone-ranlib"
