# Sourced by dependency recipes: one musl archive and the delegated application SDK.
TS_DEPS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$TS_DEPS_DIR/../../../../tests/capstone-test-env.sh"
TS_WORK=${TS_WORK:-$CAPSTONE_TMP_ROOT/tshark-app}
export TS_DEPS_SRC=$TS_WORK/deps-src TS_DEPS_BUILD=$TS_WORK/deps-build TS_DEPS_PREFIX=$TS_WORK/deps-cap
mkdir -p "$TS_DEPS_SRC" "$TS_DEPS_BUILD" "$TS_DEPS_PREFIX/lib" "$TS_DEPS_PREFIX/include"
_ts_musl_port="$CAPSTONE_REPO_ROOT/capstone/ports/musl-capstone"

export MUSL_CACHE_ROOT=$TS_WORK/musl-src
mkdir -p "$MUSL_CACHE_ROOT"
export TS_MUSL; TS_MUSL=$(bash "$_ts_musl_port/prepare-musl-capstone.sh" | tail -1)
export TS_LINKER_SCRIPT="$CAPSTONE_REPO_ROOT/capstone/my_first_domain/link.ld"

# The key: the compiler (size and mtime of clang and the LLVM libraries it loads; a shared-libs
# build keeps codegen in libLLVM*.so), every source the runtime is built from, and this file.
_ts_key=$( {
  printf '%s\n' "${CAPSTONE_APPLICATION_PROFILE:-physical}"
  _b=$(readlink -f "$CAPSTONE_CLANG")
  { echo "$_b"; ldd "$_b" | awk '/=> \//{print $3}' | grep -E 'libLLVM|libclang' || true; } | xargs stat -L -c '%n %s %Y'
  cat "$_ts_musl_port"/runtime/*.c "$_ts_musl_port"/runtime/*.S "$_ts_musl_port/runtime/libc_overrides.list" \
      "$TS_DEPS_DIR/capstone-cc" "$TS_DEPS_DIR/env.sh"
  # git ls-files, not rg. A script does not see an interactive rg: on apollo `rg` is "command not
  # found", the listing is empty, and the key silently left out all 163 runtime and SDK sources,
  # so a change there reused the old runtime (2026-10-01).
  (cd "$CAPSTONE_REPO_ROOT" && git ls-files capstone/runtime capstone/ports/common/application) |
    sort | while IFS= read -r path; do cat "$CAPSTONE_REPO_ROOT/$path"; done
  cat "$CAPSTONE_REPO_ROOT/capstone/ports/common/application/build-sdk.sh"
} | sha256sum | cut -c1-16)

# One libc and one runtime per key, never rebuilt in place: a build that sourced this file keeps
# the directories it started with while another prepares a new key. (Rebuilding in place raced
# once: a link ran against a runtime directory that was half written, 2026-09-24.)
export TS_LIBC_ARCHIVE=$TS_WORK/musl-capstone-build-$_ts_key/libc-capstone.a
export TS_RUNTIME_DIR=$TS_WORK/deps-runtime-$_ts_key
export CAPSTONE_SDK=$TS_RUNTIME_DIR
exec {_ts_lock}> "$TS_WORK/.env.lock"; flock "$_ts_lock"
if [[ ! -f "$TS_RUNTIME_DIR/.key" || ! -f "$TS_LIBC_ARCHIVE" ]]; then
  echo "deps/env.sh: preparing libc and runtime (key $_ts_key)" >&2
  OUT_DIR=$TS_WORK/musl-capstone-build-$_ts_key bash "$_ts_musl_port/build-musl-capstone.sh" >&2 || return 1
  rm -rf "$TS_RUNTIME_DIR"; mkdir -p "$TS_RUNTIME_DIR"
  bash "$CAPSTONE_REPO_ROOT/capstone/ports/common/application/build-sdk.sh" \
    "$TS_RUNTIME_DIR" "$TS_MUSL" "$TS_LIBC_ARCHIVE" >&2 || return 1

  # capstone-cc, both directions.
  _ts_c=$(mktemp -d)
  printf '#include <stdio.h>\nint main(void){return puts("x");}\n' > "$_ts_c/ok.c"
  printf 'int no_such_function_in_any_libc(void);\nint main(void){return no_such_function_in_any_libc();}\n' > "$_ts_c/bad.c"
  "$TS_DEPS_DIR/capstone-cc" -o "$_ts_c/ok" "$_ts_c/ok.c" 2> "$_ts_c/ok.err" ||
    { echo "deps/env.sh: CONTROL FAILED: a program calling puts() does not link:" >&2; cat "$_ts_c/ok.err" >&2; return 1; }
  if "$TS_DEPS_DIR/capstone-cc" -o "$_ts_c/bad" "$_ts_c/bad.c" 2>/dev/null; then
    echo "deps/env.sh: CONTROL FAILED: an undefined function links; configure would report everything present" >&2; return 1
  fi
  if "$TS_DEPS_DIR/capstone-cc" -o "$_ts_c/badl" "$_ts_c/ok.c" -lno_such_library_anywhere 2>/dev/null; then
    echo "deps/env.sh: CONTROL FAILED: -l of a missing library links" >&2; return 1
  fi
  rm -rf "$_ts_c"
  echo "$_ts_key" > "$TS_RUNTIME_DIR/.key"
  echo "deps/env.sh: controls pass (links puts; refuses an undefined function and a missing -l)" >&2
fi
flock -u "$_ts_lock"; exec {_ts_lock}>&-

export CC="$TS_DEPS_DIR/capstone-cc"
export AR="$CAPSTONE_LLVM_BIN/llvm-ar"
export RANLIB="$TS_DEPS_DIR/capstone-ranlib"
