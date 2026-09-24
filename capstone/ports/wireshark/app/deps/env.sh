# Sourced, not run: the cross environment for the tshark port's third-party libraries.
#
#   source capstone/ports/wireshark/app/deps/env.sh
#
# Sets CC (deps/capstone-cc), AR, RANLIB and the TS_* variables capstone-cc reads, and prepares,
# once per toolchain and runtime source:
#   - this lane's own libc-capstone.a (built with the current compiler, not a shared one that may
#     predate it), under $TS_WORK/musl-capstone-build;
#   - the domain runtime objects every link takes: start-musl, hostcall, tls, the level0 heap,
#     the libc overrides (runtime/libc_overrides.sh, the one list), atomic_libcalls, soft-float,
#     compiler-rt's 128-bit integer division and float conversions, and the capstone_main -> main
#     adapter (deps/domain_entry.c).
# Then checks capstone-cc in both directions before any library is built with it:
#   - a program calling puts() links;
#   - a program calling a function no libc has does NOT link;
#   - -l of a library that is neither musl's nor built here does NOT link.
# A wrapper that fails the second or third would make configure report everything present.
#
# From a worktree, point CAPSTONE_LLVM_BUILD_DIR and CAPSTONE_BUILDROOT_DIR at the main clone
# first, as every port script here requires.

TS_DEPS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$TS_DEPS_DIR/../../../../tests/capstone-test-env.sh"
TS_WORK=${TS_WORK:-$CAPSTONE_TMP_ROOT/tshark-app}
export TS_DEPS_SRC=$TS_WORK/deps-src TS_DEPS_BUILD=$TS_WORK/deps-build TS_DEPS_PREFIX=$TS_WORK/deps-cap
mkdir -p "$TS_DEPS_SRC" "$TS_DEPS_BUILD" "$TS_DEPS_PREFIX/lib" "$TS_DEPS_PREFIX/include"
_ts_musl_port="$CAPSTONE_REPO_ROOT/capstone/ports/musl-capstone"

export TS_MUSL; TS_MUSL=$(bash "$_ts_musl_port/prepare-musl-capstone.sh" | tail -1)
export TS_LINKER_SCRIPT="$CAPSTONE_REPO_ROOT/capstone/my_first_domain/link.ld"

# The key: the compiler (size and mtime of clang and the LLVM libraries it loads; a shared-libs
# build keeps codegen in libLLVM*.so), every source the runtime is built from, and this file.
_ts_key=$( {
  _b=$(readlink -f "$CAPSTONE_CLANG")
  { echo "$_b"; ldd "$_b" | awk '/=> \//{print $3}' | grep -E 'libLLVM|libclang'; } | xargs stat -L -c '%n %s %Y'
  cat "$_ts_musl_port"/runtime/*.c "$_ts_musl_port"/runtime/*.S "$_ts_musl_port/runtime/libc_overrides.sh" \
      "$TS_DEPS_DIR/domain_entry.c" "$TS_DEPS_DIR/capstone-cc" "$TS_DEPS_DIR/env.sh"
} | sha256sum | cut -c1-16)

# One libc and one runtime per key, never rebuilt in place: a build that sourced this file keeps
# the directories it started with while another prepares a new key. (Rebuilding in place raced
# once: a link ran against a runtime directory that was half written, 2026-09-24.)
export TS_LIBC_ARCHIVE=$TS_WORK/musl-capstone-build-$_ts_key/libc-capstone.a
export TS_RUNTIME_DIR=$TS_WORK/deps-runtime-$_ts_key
exec {_ts_lock}> "$TS_WORK/.env.lock"; flock "$_ts_lock"
if [[ ! -f "$TS_RUNTIME_DIR/.key" || ! -f "$TS_LIBC_ARCHIVE" ]]; then
  echo "deps/env.sh: preparing libc and runtime (key $_ts_key)" >&2
  OUT_DIR=$TS_WORK/musl-capstone-build-$_ts_key bash "$_ts_musl_port/build-musl-capstone.sh" >&2 || return 1
  rm -rf "$TS_RUNTIME_DIR"; mkdir -p "$TS_RUNTIME_DIR"
  _ts_tf=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m -Xclang -target-feature -Xclang +a)
  _ts_cf=("${_ts_tf[@]}" -ffreestanding -fno-builtin -fno-jump-tables -ffunction-sections -fdata-sections
          -std=c99 -O1 -w -Wno-int-conversion -D_XOPEN_SOURCE=700 -nostdinc
          -isystem "$TS_MUSL/arch/capstone64" -isystem "$TS_MUSL/arch/generic"
          -isystem "$TS_MUSL/obj/include" -isystem "$TS_MUSL/include"
          -I"$TS_MUSL/src/include" -I"$TS_MUSL/src/internal" -I"$TS_MUSL/obj/src/internal")
  _ts_rt=$TS_RUNTIME_DIR
  for _s in start-musl set_thread_area setjmp; do
    "$CAPSTONE_CLANG" "${_ts_tf[@]}" -ffreestanding -O0 -c "$_ts_musl_port/runtime/$_s.S" -o "$_ts_rt/$_s.o" || return 1
  done
  # atomic_libcalls: the generic __atomic_* libcalls that 16-byte (capability) atomics compile to
  # (C-54); libc-test links it by hand, and it is not on the override list.
  for _s in hostcall tls level0 atomic_libcalls; do
    "$CAPSTONE_CLANG" "${_ts_cf[@]}" -c "$_ts_musl_port/runtime/$_s.c" -o "$_ts_rt/$_s.o" || return 1
  done
  "$CAPSTONE_CLANG" "${_ts_cf[@]}" -c "$TS_DEPS_DIR/domain_entry.c" -o "$_ts_rt/domain_entry.o" || return 1
  source "$_ts_musl_port/runtime/libc_overrides.sh"
  build_musl_overrides "$CAPSTONE_CLANG" "$_ts_rt" "$TS_MUSL" "${_ts_cf[@]}" || return 1
  ( CLANG=$CAPSTONE_CLANG OBJ_DIR=$_ts_rt/sf COMPILER_RT="$CAPSTONE_REPO_ROOT/compiler-rt/lib/builtins"
    COMMON_FLAGS=("${_ts_tf[@]}" -ffreestanding -fno-builtin -ffunction-sections -fdata-sections -O1 -w)
    mkdir -p "$OBJ_DIR"
    source "$CAPSTONE_REPO_ROOT/capstone/benchmarks/beebs/build-beebs-softfloat-common.sh"
    for _o in "${softfloat_objs[@]}"; do cp "$_o" "$_ts_rt/sf_$(basename "$_o")"; done ) || return 1
  rm -rf "$_ts_rt/sf"
  # 128-bit integer division, which the soft-float list does not carry: libgcrypt's generic C mpi
  # divides in 128 bits, and a link without these fails on __udivti3/__umodti3.
  # And four 128-bit float conversions the soft-float list lacks (GLib's tests reach them).
  for _b in udivmodti4 udivti3 umodti3 divmodti4 divti3 modti3 floatditf floatunditf fixtfdi fixunstfdi; do
    "$CAPSTONE_CLANG" "${_ts_tf[@]}" -ffreestanding -fno-builtin -ffunction-sections -fdata-sections -O1 -w \
      -c "$CAPSTONE_REPO_ROOT/compiler-rt/lib/builtins/$_b.c" -o "$_ts_rt/crt_$_b.o" || return 1
  done

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
