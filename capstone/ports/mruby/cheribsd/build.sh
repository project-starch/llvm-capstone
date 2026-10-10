#!/usr/bin/env bash
# mruby 4.0.0-rc2 cross-built for CheriBSD purecap, the fourth arm of the
# release corpus: the same defects under a deployed temporal defence instead of
# ours. CHERI_SDK and CHERI_SYSROOT must be a matched installed pair; see
# ../../common/host/cheribsd/README.md.
#
#   CHERI_SDK=... CHERI_SYSROOT=... bash build.sh [MRUBY_CHERI_ROOT]
#
# The result is $ROOT/src/build/cheribsd/bin/mruby, a STATIC purecap binary.
# Three things this build needs that the domain build does not:
#   - patches/4.0.0-rc2/0001, because RProc's declared alignment is below a
#     capability's and the CHERI compiler rejects that (mruby does not build
#     for purecap without it, at the pin or at master);
#   - static linking: dynamically linked it dies in the loader with
#     "Traditional TLS not supported" before main;
#   - the same configuration the domain port needs on a 16-byte-pointer target
#     (MRB_NO_BOXING, POOL_ALIGNMENT=16, MRB_NO_DIRECT_THREADING). Without it
#     the interpreter builds and runs --version, then takes SIGBUS on the first
#     Ruby expression it evaluates.
# It also takes ../app/patches/4.0.0-rc2/0001-0003 (the capability-ABI fixes,
# which a purecap target needs for the same reason the domain does) and 0010
# (envadjust; under revocation the old VM stack pointer is dead, so without it
# recursion dies at ~40 frames).
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../tests/capstone-test-env.sh" >/dev/null
: "${CHERI_SDK:?set CHERI_SDK to an installed CHERI SDK}"
: "${CHERI_SYSROOT:?set CHERI_SYSROOT to the matching purecap rootfs}"
ROOT=${1:-${MRUBY_CHERI_ROOT:-$CAPSTONE_TMP_ROOT/mruby-cheribsd}}
PIN=4.0.0-rc2
COMMIT=9d523e2f74f2e63ca02840937523de61398a617d
[[ ! -e $ROOT ]] || { echo "build root already exists: $ROOT" >&2; exit 2; }
mkdir -p "$ROOT"
MIRROR=${MRUBY_MIRROR:-}
if [[ -n $MIRROR ]]; then
  git clone -q "$MIRROR" "$ROOT/src"
else
  git clone -q https://github.com/mruby/mruby.git "$ROOT/src"
fi
git -C "$ROOT/src" checkout -q "$COMMIT"
for p in "$HERE"/../app/patches/$PIN/000{1,2,3}-*.patch \
         "$HERE"/../app/patches/$PIN/0010-*.patch \
         "$HERE"/patches/$PIN/*.patch; do
  (cd "$ROOT/src" && patch -p1 -s --fuzz=0 < "$p") \
    || { echo "patch $p did not apply" >&2; exit 2; }
  echo "applied $(basename "$p")"
done
( cd "$ROOT/src" && MRUBY_CONFIG="$HERE/build_config.rb" rake -j"${JOBS:-8}" )
BIN=$ROOT/src/build/cheribsd/bin/mruby
[[ -x $BIN ]] || { echo "no interpreter at $BIN" >&2; exit 2; }
file "$BIN" | grep -q 'statically linked' \
  || { echo "the interpreter is not static; the guest loader will refuse it" >&2; exit 2; }
echo "built $BIN"
echo "sha256 $(sha256sum "$BIN" | cut -d' ' -f1)"
