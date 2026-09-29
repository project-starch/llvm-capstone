#!/usr/bin/env bash
# Build CPython 3.13.7 with spatial or PoisonCap pymalloc on CheriBSD purecap.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PORT=$HERE/..
source "$HERE/../../../../tests/capstone-test-env.sh" >/dev/null
: "${CHERI_SDK:?set CHERI_SDK to the CheriBSD SDK}"
: "${CHERI_SYSROOT:?set CHERI_SYSROOT to the purecap rootfs}"
ROOT=${CPY_CHERI_ROOT:-$CAPSTONE_TMP_ROOT/cpython-cheribsd}
MODE=${CPY_CHERI_MODE:-spatial}
[[ $MODE == spatial || $MODE == poisoncap ]] \
  || { echo "CPY_CHERI_MODE must be spatial or poisoncap" >&2; exit 2; }
if [[ ${CPY_CHERI_GAP_OBSERVER:-0} == 1 && $MODE != poisoncap ]]; then
  echo "CPY_CHERI_GAP_OBSERVER requires CPY_CHERI_MODE=poisoncap" >&2
  exit 2
fi
[[ ! -e $ROOT ]] || { echo "build root already exists: $ROOT" >&2; exit 2; }
JOBS=${JOBS:-12}
ARCHIVE=${CPY_ARCHIVE:-$CAPSTONE_TMP_ROOT/cpython-upstream/Python-3.13.7.tgz}
mkdir -p "$ROOT/src" "$ROOT/build" "$(dirname "$ARCHIVE")"
if [[ ! -f $ARCHIVE ]]; then
  url=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["url"])' "$PORT/upstream.json")
  curl -fL "$url" -o "$ARCHIVE"
fi
expected=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' "$PORT/upstream.json")
[[ $(sha256sum "$ARCHIVE" | cut -d' ' -f1) == "$expected" ]] \
  || { echo "CPython source archive SHA-256 mismatch" >&2; exit 2; }
tar -xzf "$ARCHIVE" -C "$ROOT/src"
SRC=$ROOT/src/Python-3.13.7
PATCHES=()
for patch_file in "$PORT"/patches/cpython-3.13.7-*.patch; do
  case "$patch_file" in
    *-0006-*) continue ;;
    *-0009-*) [[ $MODE == poisoncap ]] && continue ;;
    *-0014-*) [[ $MODE == spatial ]] && continue ;;
  esac
  patch -d "$SRC" -p1 --batch --forward --fuzz=0 -s < "$patch_file"
  PATCHES+=("$patch_file")
done
patch -d "$SRC" -p1 --batch --forward --fuzz=0 -s \
  < "$HERE/patches/cpython-3.13.7-cheribsd-purecap.patch"
PATCHES+=("$HERE/patches/cpython-3.13.7-cheribsd-purecap.patch")
if [[ $MODE == poisoncap ]]; then
  patch -d "$SRC" -p1 --batch --forward --fuzz=0 -s \
    < "$HERE/patches/cpython-3.13.7-cheribsd-deferred-free.patch"
  PATCHES+=("$HERE/patches/cpython-3.13.7-cheribsd-deferred-free.patch")
fi

BUILD_PYTHON=${CPY_BUILD_PYTHON:-}
if [[ -z $BUILD_PYTHON ]]; then
  mkdir -p "$ROOT/native"
  (
    cd "$ROOT/native"
    "$SRC/configure" --disable-test-modules --without-ensurepip > configure.log 2>&1
    make -j"$JOBS" python > make.log 2>&1
  )
  BUILD_PYTHON=$ROOT/native/python
fi
[[ -x $BUILD_PYTHON ]] || { echo "missing native build Python: $BUILD_PYTHON" >&2; exit 2; }
[[ $("$BUILD_PYTHON" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])') == 3.13.7 ]] \
  || { echo "build Python must be 3.13.7" >&2; exit 2; }

CC="$CHERI_SDK/bin/clang --target=riscv64-unknown-freebsd13 --sysroot=$CHERI_SYSROOT -march=rv64imafdcxcheri -mabi=l64pc128d -mno-relax -fuse-ld=lld -B$CHERI_SDK/bin"
(
  cd "$ROOT/build"
  CC="$CC" CFLAGS='-O1 -Wno-error' ac_cv_file__dev_ptmx=no \
    ac_cv_file__dev_ptc=no ac_cv_buggy_getaddrinfo=no \
    "$SRC/configure" --host=riscv64-unknown-freebsd13 \
    --build=x86_64-pc-linux-gnu --with-build-python="$BUILD_PYTHON" \
    --disable-test-modules --without-ensurepip --without-readline \
    --disable-ipv6 > configure.log 2>&1
  if [[ $MODE == poisoncap ]]; then
    PYM=$PORT/../pymalloc/src
    COMMON=(--target=riscv64-unknown-freebsd13 --sysroot="$CHERI_SYSROOT"
            -march=rv64imafdcxcheri -mabi=l64pc128d -mno-relax
            -O1 -fPIC -DPYMALLOC_POISONCAP=1 -I"$PYM/shared")
    if [[ ${CPY_CHERI_GAP_OBSERVER:-0} == 1 ]]; then
      COMMON+=(-DPYMALLOC_GAP_OBSERVER=1)
    fi
    "$CHERI_SDK/bin/clang" "${COMMON[@]}" -DPYMALLOC_APP_MEMORY=1 \
      -c "$PYM/cheribsd/poisoncap-lifetimes.c" -o "$ROOT/poisoncap-lifetimes.o"
    "$CHERI_SDK/bin/clang" "${COMMON[@]}" \
      -c "$PYM/shared/backing.c" -o "$ROOT/pym-backing.o"
    "$CHERI_SDK/bin/clang" "${COMMON[@]}" \
      -c "$HERE/poisoncap-glue.c" -o "$ROOT/poisoncap-glue.o"
    printf '\nLIBS += %s %s %s\n' "$ROOT/poisoncap-lifetimes.o" \
      "$ROOT/pym-backing.o" "$ROOT/poisoncap-glue.o" >> Makefile
  fi
  make -j"$JOBS" python > make.log 2>&1 \
    || { tail -n 60 make.log >&2; exit 1; }
  if [[ $MODE == poisoncap ]]; then
    "$CHERI_SDK/bin/clang" "${COMMON[@]}" \
      -c "$PORT/../../../experiments/applications/cheribsd-memory.c" \
      -o "$ROOT/cheribsd-memory.o"
    printf '\nLIBS += %s\nPY_CORE_LDFLAGS += -Wl,--wrap=main,--wrap=write\n' \
      "$ROOT/cheribsd-memory.o" >> Makefile
    touch Programs/python.o
    make -j"$JOBS" python > phase-link.log 2>&1 \
      || { tail -n 60 phase-link.log >&2; exit 1; }
  fi
)
"$CHERI_SDK/bin/llvm-strip" --strip-debug -o "$ROOT/python" "$ROOT/build/python"
"$BUILD_PYTHON" "$PORT/make-stdlib-zip.py" "$SRC/Lib" \
  "$ROOT/pyhome/lib/python313.zip" > "$ROOT/stdlib.log"
python3 - "$ROOT" "$ARCHIVE" "$HERE" "$PORT" "$CHERI_SDK" "$CHERI_SYSROOT" "$MODE" \
          "$BUILD_PYTHON" "${PATCHES[@]}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import sys

root, archive, here, port, sdk, sysroot = map(Path, sys.argv[1:7])
mode = sys.argv[7]
build_python = Path(sys.argv[8])
patches = list(map(Path, sys.argv[9:]))
def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

config = (root / 'build/pyconfig.h').read_text()
def setting(name):
    match = re.search(rf'^#define {name} (\d+)\b', config, re.MULTILINE)
    if not match:
        raise ValueError(f'{name} missing from pyconfig.h')
    return int(match.group(1))

if (setting('SIZEOF_VOID_P'), setting('SIZEOF_UINTPTR_T'),
        setting('WITH_PYMALLOC')) != (16, 16, 1):
    raise ValueError('CheriBSD purecap pymalloc build has the wrong configuration')
document = {
    'schema': 1,
    'application': 'cpython-3.13.7',
    'arm': 'cheribsd-pymalloc-' + mode,
    'source_archive_sha256': sha256(archive),
    'binary_sha256': sha256(root / 'python'),
    'stdlib_zip_sha256': sha256(root / 'pyhome/lib/python313.zip'),
    'pyconfig_sha256': sha256(root / 'build/pyconfig.h'),
    'compiler_sha256': sha256(sdk / 'bin/clang'),
    'build_python_sha256': sha256(build_python),
    'sysroot': str(sysroot.resolve()),
    'compiler_target': 'riscv64-unknown-freebsd13, l64pc128d',
    'application_cflags': '-O1 -Wno-error',
    'pointer_bytes': 16,
    'uintptr_bytes': 16,
    'pymalloc': True,
    'reuse_gap_observer': os.environ.get('CPY_CHERI_GAP_OBSERVER') == '1',
    'inputs_sha256': {str(path): sha256(path) for path in
        [here / 'build.sh', port / 'make-stdlib-zip.py', port / 'upstream.json',
         *patches, *([here / 'poisoncap-glue.c',
                      port / '../pymalloc/src/cheribsd/poisoncap-lifetimes.c',
                      port / '../pymalloc/src/shared/backing.c',
                      port / '../pymalloc/src/shared/port.h',
                      port / '../../../experiments/applications/cheribsd-memory.c']
                     if mode == 'poisoncap' else []),
         *([here / '../../../../experiments/study/reuse-gap-observer.h']
           if os.environ.get('CPY_CHERI_GAP_OBSERVER') == '1' else [])]},
}
(root / 'manifest.json').write_text(json.dumps(document, indent=2, sort_keys=True) + '\n')
PY
echo "$ROOT/python"
