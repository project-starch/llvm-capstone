#!/usr/bin/env bash
# Prepare a CPython 3.13.7 tree configured for a musl capstone64 domain.
#
#   1. the verified source archive, unpacked outside the repository
#   2. a native CPython 3.13.7 (configure requires one for a cross build: it
#      regenerates and freezes modules)
#   3. musl as libc-capstone.a, the port runtime, and compiler-rt's builtins,
#      all built by the compiler under test
#   4. a link check that must pass for a libc symbol and FAIL for a missing one,
#      because configure's HAVE_* answers are only as good as that link
#   5. CPython's own configure, cross, against all of the above
#
# Prints the configured build directory as its last line. Everything it writes
# lives under $CPY_ROOT (default $CAPSTONE_TMP_ROOT/cpython-interpreter).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PORTS_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
source "$PORTS_DIR/../tests/capstone-test-env.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
CPY_ROOT=${CPY_ROOT:-$CAPSTONE_TMP_ROOT/cpython-interpreter}
CPY_VERSION=3.13.7
CPY_SHA256=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' "$SCRIPT_DIR/upstream.json")
CPY_URL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["url"])' "$SCRIPT_DIR/upstream.json")
CPY_ARCHIVE=${CPY_ARCHIVE:-$CPY_ROOT/download/Python-$CPY_VERSION.tgz}
CPY_SRC=$CPY_ROOT/src/Python-$CPY_VERSION
BUILD_DIR=$CPY_ROOT/build
LLVM_AR=${CAPSTONE_LLVM_AR:-$CAPSTONE_LLVM_BIN/llvm-ar}
log() { printf '[prepare] %s\n' "$*" >&2; }

for tool in "$CAPSTONE_CLANG" "$CAPSTONE_LD_LLD" "$LLVM_AR"; do
  [[ -x "$tool" ]] || { echo "missing tool: $tool (set CAPSTONE_LLVM_BUILD_DIR)" >&2; exit 2; }
done
mkdir -p "$CPY_ROOT"/{download,src,runtime,builtins}

# ---- 1. source ----------------------------------------------------------
[[ -f "$CPY_ARCHIVE" ]] || curl -fL "$CPY_URL" -o "$CPY_ARCHIVE"
got=$(sha256sum "$CPY_ARCHIVE" | cut -d' ' -f1)
[[ "$got" == "$CPY_SHA256" ]] || { echo "CPython archive SHA-256 mismatch: $got" >&2; exit 2; }
# Always from the archive: a tree left behind by an earlier run may carry
# regenerated files, and the survey must not depend on history.
rm -rf "$CPY_SRC"
tar -xzf "$CPY_ARCHIVE" -C "$CPY_ROOT/src"
# patches/ in order. CPY_PATCHES=none measures upstream as released, which is
# the number every patch has to be argued against.
APPLIED=()
if [[ "${CPY_PATCHES:-all}" != none ]]; then
  for p in "$SCRIPT_DIR"/patches/cpython-$CPY_VERSION-*.patch; do
    [[ -e "$p" ]] || continue
    patch -d "$CPY_SRC" --batch --forward --fuzz=0 -p1 < "$p" >/dev/null \
      || { echo "patch did not apply: $p" >&2; exit 2; }
    APPLIED+=("$(basename "$p")")
  done
fi
log "patches applied: ${#APPLIED[@]} ${APPLIED[*]:-}"

# ---- 2. native build python ---------------------------------------------
BUILD_PYTHON=${CPY_BUILD_PYTHON:-$CPY_ROOT/build-python/python}
if [[ ! -x "$BUILD_PYTHON" ]]; then
  log "building native CPython $CPY_VERSION"
  mkdir -p "$CPY_ROOT/build-python"
  (cd "$CPY_ROOT/build-python" && "$CPY_SRC/configure" >configure.log 2>&1 \
     && make -j"${CPY_JOBS:-8}" >make.log 2>&1)
fi
bv=$("$BUILD_PYTHON" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
[[ "$bv" == "$CPY_VERSION" ]] || { echo "build python is $bv, need $CPY_VERSION" >&2; exit 2; }

# ---- 3. musl, runtime, builtins -----------------------------------------
# A private musl tree and archive. prepare-musl-capstone.sh rewrites
# arch/capstone64 in place, and the default cache is shared with other work.
export MUSL_CACHE_ROOT=$CPY_ROOT/musl-src
mkdir -p "$MUSL_CACHE_ROOT"
shared_tarball=$CAPSTONE_TMP_ROOT/musl-src/musl-1.2.5.tar.gz
[[ -f "$MUSL_CACHE_ROOT/musl-1.2.5.tar.gz" || ! -f "$shared_tarball" ]] \
  || cp "$shared_tarball" "$MUSL_CACHE_ROOT/"
MUSL_DIR=$(bash "$PORTS_DIR/musl-capstone/prepare-musl-capstone.sh" | tail -1)
log "building libc-capstone.a with $CAPSTONE_CLANG"
OUT_DIR=$CPY_ROOT/musl-build bash "$PORTS_DIR/musl-capstone/build-musl-capstone.sh" >&2
LIBC_ARCHIVE=$CPY_ROOT/musl-build/libc-capstone.a

# The runtime exactly as musl-capstone/libc-test/build-libc-test.sh builds it.
RT=$CPY_ROOT/runtime
rm -f "$RT"/*.o
MRT=$PORTS_DIR/musl-capstone/runtime
INC=(-nostdinc -isystem "$MUSL_DIR/arch/capstone64" -isystem "$MUSL_DIR/arch/generic"
     -isystem "$MUSL_DIR/obj/include" -isystem "$MUSL_DIR/include"
     -I"$MUSL_DIR/src/include" -I"$MUSL_DIR/src/internal" -I"$MUSL_DIR/obj/src/internal")
CF=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
    -Xclang -target-feature -Xclang +a -ffreestanding -fno-builtin -fno-jump-tables
    -ffunction-sections -fdata-sections -std=c99 -O1 -w -Wno-int-conversion
    -D_XOPEN_SOURCE=700 "${INC[@]}")
ASF=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m -ffreestanding -O0)
for s in start-musl set_thread_area setjmp; do
  "$CAPSTONE_CLANG" "${ASF[@]}" -c "$MRT/$s.S" -o "$RT/$s.o"
done
for f in hostcall tls level0 string_bounds_safe fputwc_null_safe; do
  "$CAPSTONE_CLANG" "${CF[@]}" -c "$MRT/$f.c" -o "$RT/$f.o"
done
"$CAPSTONE_CLANG" "${CF[@]}" -I"$MUSL_DIR/src/multibyte" \
  -c "$MRT/mbsrtowcs_bounds_safe.c" -o "$RT/mbsrtowcs_bounds_safe.o"
"$CAPSTONE_CLANG" "${CF[@]}" -c "$SCRIPT_DIR/toolchain/domain_entry.c" -o "$RT/domain_entry.o"

# compiler-rt's generic builtins, as an ARCHIVE so the linker takes only what is
# referenced -- which is what a toolchain's libclang_rt.builtins.a is. The
# benchmarks' hand-picked soft-float list would make configure report a libc
# function missing whenever the function needs a builtin the list lacks.
COMPILER_RT=$REPO_ROOT/compiler-rt/lib/builtins
BF=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
    -ffreestanding -fno-builtin -ffunction-sections -fdata-sections -O1 -w -I"$COMPILER_RT")
rm -f "$CPY_ROOT/builtins"/*.o "$CPY_ROOT/builtins/failed.txt"
built=0; failed=0
for f in "$COMPILER_RT"/*.c; do
  b=$(basename "$f" .c)
  if "$CAPSTONE_CLANG" "${BF[@]}" -c "$f" -o "$CPY_ROOT/builtins/$b.o" 2>"$CPY_ROOT/builtins/$b.err"; then
    built=$((built+1)); rm -f "$CPY_ROOT/builtins/$b.err"
  else
    failed=$((failed+1)); rm -f "$CPY_ROOT/builtins/$b.o"
    printf '%s\t%s\n' "$b" "$(grep -m1 'error' "$CPY_ROOT/builtins/$b.err")" >> "$CPY_ROOT/builtins/failed.txt"
  fi
done
(( built > 0 )) || { echo "no compiler-rt builtin compiled" >&2; exit 2; }
rm -f "$CPY_ROOT/libclang_rt.builtins.a"
"$LLVM_AR" rcs "$CPY_ROOT/libclang_rt.builtins.a" "$CPY_ROOT/builtins"/*.o
log "compiler-rt builtins: $built archived, $failed did not compile (builtins/failed.txt)"

# One archive for the wrapper: musl first, builtins after, as a link would order them.
COMBINED=$CPY_ROOT/libc-and-builtins.a
rm -f "$COMBINED"
printf 'CREATE %s\nADDLIB %s\nADDLIB %s\nSAVE\nEND\n' \
  "$COMBINED" "$LIBC_ARCHIVE" "$CPY_ROOT/libclang_rt.builtins.a" | "$LLVM_AR" -M

export CPY_MUSL=$MUSL_DIR CPY_RUNTIME_DIR=$RT CPY_LIBC_ARCHIVE=$COMBINED
export CPY_LINKER_SCRIPT=$REPO_ROOT/capstone/my_first_domain/link.ld
CC=$SCRIPT_DIR/toolchain/capstone-cc

# ---- 4. the link check must be able to say no ----------------------------
LC=$CPY_ROOT/linkcheck; rm -rf "$LC"; mkdir -p "$LC"
printf '#include <string.h>\nint main(void){return (int)strlen("x");}\n' > "$LC/has.c"
printf 'char capstone_no_such_function(void);\nint main(void){return capstone_no_such_function();}\n' > "$LC/hasnot.c"
"$CC" -O1 "$LC/has.c" -o "$LC/has.dom" >"$LC/has.log" 2>&1 \
  || { cat "$LC/has.log" >&2; echo "link check: a program calling strlen does not link" >&2; exit 2; }
if "$CC" -O1 "$LC/hasnot.c" -o "$LC/hasnot.dom" >"$LC/hasnot.log" 2>&1; then
  echo "link check: an undefined function LINKED; configure's HAVE_* would all be yes" >&2; exit 2
fi
grep -q 'undefined symbol: capstone_no_such_function' "$LC/hasnot.log" \
  || { cat "$LC/hasnot.log" >&2; echo "link check: failed, but not for the undefined symbol" >&2; exit 2; }
if "$CC" -O1 "$LC/has.c" -lz -o "$LC/hasz.dom" >"$LC/hasz.log" 2>&1; then
  echo "link check: -lz accepted; configure would find a zlib this target does not have" >&2; exit 2
fi
log "link check: strlen links, an undefined function and -lz are refused"

# ---- 5. configure -------------------------------------------------------
rm -rf "$BUILD_DIR"; mkdir -p "$BUILD_DIR"
# Answers configure cannot get by running a program on the target. Each is a
# fact about the domain, not a way to make configure pass.
cat > "$BUILD_DIR/config.site" <<'EOF'
# No device files exist in a domain: the file service opens host paths only.
ac_cv_file__dev_ptmx=no
ac_cv_file__dev_ptc=no
# getaddrinfo is never run (no socket opcode); "not buggy" only stops configure
# from refusing to continue without a run test.
ac_cv_buggy_getaddrinfo=no
# Left to itself this check CRASHES the compiler (C-51: its conftest uses
# _Py_atomic_or_uint8), configure reads the crash as "libatomic needed", and
# LIBS gains a -latomic no capstone64 library provides. With +a every width the
# check uses is lowered inline, so the answer is no.
ac_cv_libatomic_needed=no
EOF
log "configuring CPython for riscv64-unknown-linux-musl via $CC"
# --without-computed-gotos: a table of &&label values is emitted WITHOUT
#   capability-init records and loads untagged; the first dispatch faults on
#   instruction fetch. It compiles cleanly, so a compile survey cannot see it.
#   docs/history/05-08-2026_06-00-00_gp-captable-lua-bringup.md, "KNOWN COMPILER GAP".
# MODULE_BUILDTYPE=static: no dlopen in a domain, so every module is built in.
# --with-pkg-config=no: the host's pkg-config would hand over host library flags.
# RANLIB is `llvm-ar s`: not every LLVM build here has the llvm-ranlib link.
# -D_Py_THREAD_LOCAL_AS_GLOBAL: a domain has one hart and no clone, so each
#   thread-local has one instance; patches/...-0006 makes it a global, because
#   capstone64 cannot lower TLS (ISSUES.md C-47). Only valid while nothing can
#   start a thread.
(cd "$BUILD_DIR" && \
  CONFIG_SITE="$BUILD_DIR/config.site" MODULE_BUILDTYPE=static \
  CPPFLAGS="-D_Py_THREAD_LOCAL_AS_GLOBAL" \
  CC="$CC" AR="$LLVM_AR" RANLIB="$LLVM_AR s" READELF=: \
  "$CPY_SRC/configure" \
    --host=riscv64-unknown-linux-musl \
    --build="$("$CPY_SRC/config.guess")" \
    --with-build-python="$BUILD_PYTHON" \
    --disable-shared --disable-ipv6 --disable-test-modules \
    --without-ensurepip --without-computed-gotos \
    --with-pkg-config=no \
    > configure.log 2>&1) \
  || { tail -30 "$BUILD_DIR/configure.log" >&2; echo "configure failed: $BUILD_DIR/configure.log" >&2; exit 2; }
[[ -s "$BUILD_DIR/pyconfig.h" && -s "$BUILD_DIR/Makefile" ]] \
  || { echo "configure produced no pyconfig.h/Makefile" >&2; exit 2; }
# The settings above reach configure only through its environment and argv;
# a slip in how they are passed leaves configure running with its defaults and
# nothing failing. Read them back from what configure wrote.
# MODULE_BUILDTYPE is substituted into Modules/Setup.stdlib, not the Makefile:
# its first build-type marker must be *static*. (The later *shared* block holds
# only test modules, which --disable-test-modules leaves commented out.)
[[ "$(grep -m1 -E '^\*(static|shared)\*$' "$BUILD_DIR/Modules/Setup.stdlib")" == '*static*' ]] \
  || { echo "configure did not take MODULE_BUILDTYPE=static" >&2; exit 2; }
if grep -qE '^[a-z_]' <(sed -n '/^\*shared\*$/,$p' "$BUILD_DIR/Modules/Setup.stdlib"); then
  echo "Setup.stdlib builds a module as shared; a domain cannot load it" >&2; exit 2
fi
grep -q "loading site script $BUILD_DIR/config.site" "$BUILD_DIR/configure.log" \
  || { echo "configure did not read $BUILD_DIR/config.site" >&2; exit 2; }
grep -qE '^CONFIGURE_CPPFLAGS=.*-D_Py_THREAD_LOCAL_AS_GLOBAL' "$BUILD_DIR/Makefile" \
  || { echo "configure did not carry -D_Py_THREAD_LOCAL_AS_GLOBAL into the Makefile" >&2; exit 2; }
# A configure check whose conftest crashes the compiler reads as "feature
# absent", silently. Every such check must be one whose answer is right anyway.
python3 - "$BUILD_DIR/config.log" <<'PY' || exit 2
import re, sys
EXPECTED = {
    # x86 and m68k FPU control words: foreign to this target, "no" is correct.
    # The crash itself is C-53 (an inline-asm "m" input operand).
    "whether we can use gcc inline assembler to get and set x87 control word",
    "whether we can use gcc inline assembler to get and set mc68881 fpcr",
}
check, crashed = None, set()
for line in open(sys.argv[1], errors="replace"):
    m = re.match(r"configure:\d+: checking (.*)", line)
    if m:
        check = m.group(1).strip()
    elif re.search(r"PLEASE submit a bug report|Assertion `.*' failed|error in backend|exit code 139", line):
        crashed.add(check)
if not crashed <= EXPECTED:
    print("configure checks that crashed the compiler, answer unreviewed:", file=sys.stderr)
    for c in sorted(crashed - EXPECTED):
        print("  " + c, file=sys.stderr)
    sys.exit(2)
print(f"[prepare] configure checks that crashed the compiler: {len(crashed)}, all reviewed", file=sys.stderr)
PY
if grep -qE '^LIBS=.*-latomic' "$BUILD_DIR/Makefile"; then
  echo "configure put -latomic in LIBS; no capstone64 libatomic exists" >&2; exit 2
fi
if grep -q '^#define USE_COMPUTED_GOTOS 1' "$BUILD_DIR/pyconfig.h"; then
  echo "configure enabled computed gotos" >&2; exit 2
fi

# The environment the survey needs to drive make with the same wrapper.
cat > "$BUILD_DIR/capstone-env.sh" <<EOF
export CAPSTONE_CLANG='$CAPSTONE_CLANG' CAPSTONE_LD_LLD='$CAPSTONE_LD_LLD'
export CPY_MUSL='$MUSL_DIR' CPY_RUNTIME_DIR='$RT' CPY_LIBC_ARCHIVE='$COMBINED'
export CPY_LINKER_SCRIPT='$CPY_LINKER_SCRIPT'
EOF
printf '%s\n' "${APPLIED[@]}" > "$BUILD_DIR/applied-patches.txt"
printf '%s\n' "$BUILD_DIR"
