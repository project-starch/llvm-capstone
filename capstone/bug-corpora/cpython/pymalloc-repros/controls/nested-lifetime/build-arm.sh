#!/usr/bin/env bash
# Build one interpreter arm for the nested-lifetime controls, with _pymctl.c built in.
#
#   build-arm.sh protected|release NEW_OUTPUT
#
# The same steps as `CPY_TEST_CAPI=1 [CPY_SUBLET=1] build-virtual.sh cpython OUT`
# (capstone/ports/common/application/build-virtual.sh, its cpython branch), with
# one step added between configure and the compile survey: _pymctl.c is copied
# into the unpacked source's Modules/ and appended to Setup.local's *static*
# block, exactly as prepare-cpython-capstone.sh adds the test C-API modules. Both
# arms get the identical module; protected differs only by CPY_SUBLET=1 (patch
# 0014). Nothing under capstone/ports is modified.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ $# == 2 ]] || { echo 'usage: build-arm.sh protected|release NEW_OUTPUT' >&2; exit 2; }
ARM=$1
OUT=$(realpath -m "$2")
[[ ! -e "$OUT" ]] || { echo "use a fresh output directory: $OUT" >&2; exit 2; }
case "$ARM" in
  protected) export CPY_SUBLET=1; NESTED=cpython ;;
  release)   export CPY_SUBLET=0; NESTED=none ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac
source "$HERE/../../../../../tests/capstone-test-env.sh" >/dev/null
export CAPSTONE_APPLICATION_PROFILE=virtual CPY_TEST_CAPI=1
PORTS=$CAPSTONE_REPO_ROOT/capstone/ports
mkdir -p "$OUT/source"
export CPY_ROOT=$OUT/source
bash "$PORTS/cpython/app/prepare-cpython-capstone.sh"
source "$CPY_ROOT/build/capstone-env.sh"

# The module, built in.
cp "$HERE/_pymctl.c" "$CPY_ROOT/src/Python-3.13.7/Modules/_pymctl.c"
printf '\n*static*\n_pymctl _pymctl.c\n' >> "$CPY_ROOT/build/Modules/Setup.local"
(cd "$CPY_ROOT/build" && make Makefile > make-makefile-pymctl.log 2>&1) \
  || { tail -12 "$CPY_ROOT/build/make-makefile-pymctl.log" >&2; echo "make Makefile failed" >&2; exit 2; }
grep -q 'Modules/_pymctl.o' "$CPY_ROOT/build/Makefile" \
  || { echo "_pymctl is not in the Makefile" >&2; exit 2; }
grep -q 'PyInit__pymctl' "$CPY_ROOT/build/Modules/config.c" \
  || { echo "_pymctl is not in Modules/config.c" >&2; exit 2; }

python3 "$PORTS/cpython/app/survey-cpython-capstone.py" "$CPY_ROOT/build" --jobs "${JOBS:-12}"
python3 "$PORTS/cpython/app/link-cpython-capstone.py" "$CPY_ROOT/build" \
  --native "$CPY_ROOT/build-python" --out "$CPY_ROOT/link"
[[ -f "$CPY_ROOT/build/Modules/_pymctl.o" ]] || { echo "_pymctl.o was not compiled" >&2; exit 2; }
python3 "$PORTS/common/application/build.py" --app cpython --root "$OUT/source" \
  --profile virtual --nested "$NESTED" --toolchain "$CAPSTONE_LLVM_BUILD_DIR" \
  --input-revision HEAD --stack-bytes 8388608 --out "$OUT/image"
# Not a pipe into grep -q: under pipefail nm's SIGPIPE would fail the check.
symbols=$("$CAPSTONE_LLVM_BUILD_DIR/bin/llvm-nm" "$OUT/image/cpython.dom")
grep -q ' pymctl_stale_read$' <<<"$symbols" \
  || { echo "the image carries no pymctl_stale_read label" >&2; exit 2; }
sha256sum "$OUT/image/cpython.dom"
