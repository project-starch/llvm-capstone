#!/usr/bin/env bash
# Build one arm of virtual PHP 5.0.0 with the layout trace of
# docs/plans/2026-11-15-php-port-layout-fix.md, without editing the port:
#
#   build-layout-trace.sh stock|protected ROOT
#
# Environment as for build.sh (CAPSTONE_REPO_ROOT, CAPSTONE_LLVM_BUILD_DIR, CAPSTONE_LLVM_BIN,
# CAPSTONE_TMP_ROOT).
#
# 1. build.sh's step 1 (the common entry's own build: SDK, source tree by hash, uninstrumented
#    image) into ROOT.
# 2. zend-alloc-layout-trace.patch (fuzz 0) and the two trace headers into the prepared tree's
#    Zend/, a check that the nested layer's patch still applies to the traced zend_alloc.c at
#    fuzz 0 and without offset, then the port's realphp build again over the same SDK with
#    EXTRA_CF=-DZEND_CAP_LAYOUT_TRACE. The image is ROOT/trace.dom; it must define the trace's
#    counters, and the uninstrumented image must not.
# 3. layout-static.c with the arm's realphp flags: ROOT/layout-static.dom prints the header's
#    sizeof and offsets.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ $# == 2 ]] || { echo "usage: build-layout-trace.sh stock|protected ROOT" >&2; exit 2; }
ARM=$1 ROOT=$(realpath -m "$2")
case $ARM in stock) ALLOC=stock ;; protected) ALLOC=nested ;; *) echo "arm: stock or protected" >&2; exit 2 ;; esac
R=${CAPSTONE_REPO_ROOT:?}
NESTED=$R/capstone/ports/php/zend-alloc/sublet
NESTED_PATCH=$NESTED/patches/5.0.0/0001-zend-alloc-sublet-hooks.patch
NM=${CAPSTONE_LLVM_BUILD_DIR:?}/bin/llvm-nm

PHP_REUSE_STOP=1 bash "$HERE/build.sh" "$ARM" "$ROOT"
# nm into a file, then grep: grep -q stops reading early, and pipefail would read nm's SIGPIPE as a failure.
"$NM" "$ROOT/uninstrumented.dom" > "$ROOT/uninstrumented.dom.nm"
grep -q ' [tT] zend_layout_malloc$' "$ROOT/uninstrumented.dom.nm" \
  && { echo "control: the uninstrumented image already defines the trace" >&2; exit 1; }

s=$(date +%s)
P=$(ls -d "$ROOT"/source/php-5.0.0)
cp "$HERE/zend-layout-trace.h" "$HERE/zend-layout-trace-def.h" "$P/Zend/"
patch -d "$P" -p1 --batch --forward --fuzz=0 < "$HERE/zend-alloc-layout-trace.patch" > "$ROOT/trace-patch.log"
rm -rf "$ROOT/nested-check" && mkdir -p "$ROOT/nested-check/Zend"
cp "$P/Zend/zend_alloc.c" "$ROOT/nested-check/Zend/"
patch -d "$ROOT/nested-check" -p1 --batch --fuzz=0 < "$NESTED_PATCH" > "$ROOT/nested-check.log" \
  || { echo "the nested layer's patch does not apply to the traced zend_alloc.c at fuzz 0" >&2; exit 1; }
! grep -q 'offset\|fuzz' "$ROOT/nested-check.log" \
  || { echo "the nested layer's patch applies to the traced zend_alloc.c only with an offset" >&2; exit 1; }
PHP_SRC=$P PHP_VIRTUAL_OUT=$ROOT PHP_REALPHP_ALLOC=$ALLOC EXTRA_CF=-DZEND_CAP_LAYOUT_TRACE \
  CAPSTONE_APPLICATION_PROFILE=virtual MUSL_CACHE_ROOT=$ROOT/musl-src \
  bash "$R/capstone/ports/php/app/build-virtual.sh" realphp > "$ROOT/build-trace.log" 2>&1
cp "$ROOT/realphp/php-real-$ALLOC.dom" "$ROOT/trace.dom"
"$NM" "$ROOT/trace.dom" > "$ROOT/trace.dom.nm"
grep -q ' [tT] zend_layout_malloc$' "$ROOT/trace.dom.nm" \
  || { echo "the traced image lacks the trace" >&2; exit 1; }
echo "TRACE $ARM $(( $(date +%s) - s ))s trace.dom sha256=$(sha256sum "$ROOT/trace.dom" | cut -d' ' -f1)"

T=$ROOT/realphp/$ALLOC/tree
CF=(-O0 -std=gnu89 -fno-common -D__off64_t=off_t -Wno-implicit-int -Wno-implicit-function-declaration
    -I"$T" -I"$T/Zend" -I"$T/main" -I"$T/TSRM" -I"$T/ext" -I"$T/ext/standard" -I"$T/regex" -I"$T/sapi/cli")
[[ $ALLOC == nested ]] && CF+=(-DZEND_CAP_SUBLET -I"$NESTED" -I"$R/capstone" -I"$R/capstone/runtime/include")
"$ROOT/sdk/capstone-cc" "${CF[@]}" "$HERE/layout-static.c" -o "$ROOT/layout-static.dom"
echo "STATIC $ARM layout-static.dom"
