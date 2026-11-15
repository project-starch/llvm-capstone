#!/usr/bin/env bash
# Build one arm of virtual PHP 5.0.0 with the reuse-distance hooks
# (docs/plans/2026-11-15-php-reuse-distance.md), without editing the port:
#
#   build.sh stock|protected ROOT
#
# Environment as for ports/common/application/build-virtual.sh (CAPSTONE_REPO_ROOT,
# CAPSTONE_LLVM_BUILD_DIR, CAPSTONE_LLVM_BIN, CAPSTONE_TMP_ROOT).
#
# 1. The common entry's own build, unchanged: build-virtual.sh php with
#    CAPSTONE_VIRTUAL_LIFETIMES=0 (stock: PHP's own zend_alloc.c over virtual mallocng) or 1
#    (protected: zend_alloc.c with the nested layer, ports/php/zend-alloc/sublet). It builds the
#    SDK, prepares the source tree by hash with the tree patches and links the image, kept as
#    ROOT/uninstrumented.dom.
# 2. The hook patch (zend-alloc-reuse-gap.patch, fuzz 0) and the three headers into the
#    prepared tree's Zend/, then a check that the nested layer's patch still applies to the
#    hooked zend_alloc.c at fuzz 0, then the port's realphp build again over the same SDK with
#    EXTRA_CF=-DEXP_REUSE_GAP. The protected image is built without a default arm, so every run
#    names its PHP_CAP_MODE. The image is ROOT/php.dom.
# 3. Checks that fail the build: the instrumented image defines the five hook functions and the
#    uninstrumented image defines none of them.
#
# PHP_REUSE_FROM=2 starts at step 2 in a ROOT that step 1 already filled (its uninstrumented.dom
# present, no php.dom yet): the uninstrumented build can run before the plan is committed.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ $# == 2 ]] || { echo "usage: build.sh stock|protected ROOT" >&2; exit 2; }
ARM=$1 ROOT=$(realpath -m "$2")
case $ARM in stock) LIFE=0 ALLOC=stock ;; protected) LIFE=1 ALLOC=nested ;; *) echo "arm: stock or protected" >&2; exit 2 ;; esac
R=${CAPSTONE_REPO_ROOT:?}
STUDY=$R/capstone/experiments/study
NESTED_PATCH=$R/capstone/ports/php/zend-alloc/sublet/patches/5.0.0/0001-zend-alloc-sublet-hooks.patch
NM=${CAPSTONE_LLVM_BUILD_DIR:?}/bin/llvm-nm
HOOKS="php_reuse_gap_issue php_reuse_gap_release php_reuse_gap_drop php_reuse_gap_shutdown_free php_reuse_gap_realloc"
FROM=${PHP_REUSE_FROM:-1}
case $FROM in
  1) [[ ! -e $ROOT ]] || { echo "use a fresh ROOT: $ROOT" >&2; exit 2; } ;;
  2) [[ -f $ROOT/uninstrumented.dom && ! -e $ROOT/php.dom ]] \
       || { echo "PHP_REUSE_FROM=2 needs a ROOT with step 1's uninstrumented.dom and no php.dom" >&2; exit 2; } ;;
  *) echo "PHP_REUSE_FROM=$FROM? (1, 2)" >&2; exit 2 ;;
esac
STOP=${PHP_REUSE_STOP:-2}   # 1: stop after step 1

# defines_hooks IMAGE: how many of the five hook functions the image defines.
defines_hooks() {
  "$NM" "$1" > "$1.nm"
  local n=0 h
  for h in $HOOKS; do
    grep -q " [tT] $h\$" "$1.nm" && n=$((n + 1))
  done
  echo $n
}

if [[ $FROM == 1 ]]; then
s=$(date +%s)
CAPSTONE_VIRTUAL_LIFETIMES=$LIFE bash "$R/capstone/ports/common/application/build-virtual.sh" php "$ROOT" \
  > "$ROOT.build-1.log" 2>&1
mv "$ROOT.build-1.log" "$ROOT/build-1.log"
cp "$ROOT/realphp/php-real-$ALLOC.dom" "$ROOT/uninstrumented.dom"
[[ $(defines_hooks "$ROOT/uninstrumented.dom") == 0 ]] \
  || { echo "control: the uninstrumented image already defines a hook" >&2; exit 1; }
echo "STEP1 $ARM $(( $(date +%s) - s ))s uninstrumented sha256=$(sha256sum "$ROOT/uninstrumented.dom" | cut -d' ' -f1)"
fi
[[ $STOP == 2 ]] || exit 0

s=$(date +%s)
P=$(ls -d "$ROOT"/source/php-5.0.0)
cp "$STUDY/reuse-gap-observer.h" "$HERE/reuse-gap-hook.h" "$HERE/php-reuse-gap.h" "$P/Zend/"
patch -d "$P" -p1 --batch --forward --fuzz=0 < "$HERE/zend-alloc-reuse-gap.patch" > "$ROOT/hook-patch.log"
rm -rf "$ROOT/nested-check" && mkdir -p "$ROOT/nested-check/Zend"
cp "$P/Zend/zend_alloc.c" "$ROOT/nested-check/Zend/"
patch -d "$ROOT/nested-check" -p1 --batch --fuzz=0 < "$NESTED_PATCH" > "$ROOT/nested-check.log" \
  || { echo "the nested layer's patch does not apply to the hooked zend_alloc.c at fuzz 0" >&2; exit 1; }
PHP_SRC=$P PHP_VIRTUAL_OUT=$ROOT PHP_REALPHP_ALLOC=$ALLOC EXTRA_CF=-DEXP_REUSE_GAP \
  CAPSTONE_APPLICATION_PROFILE=virtual MUSL_CACHE_ROOT=$ROOT/musl-src \
  bash "$R/capstone/ports/php/app/build-virtual.sh" realphp > "$ROOT/build-2.log" 2>&1
cp "$ROOT/realphp/php-real-$ALLOC.dom" "$ROOT/php.dom"
[[ $(defines_hooks "$ROOT/php.dom") == 5 ]] \
  || { echo "the instrumented image lacks a hook function" >&2; exit 1; }
echo "STEP2 $ARM $(( $(date +%s) - s ))s php.dom sha256=$(sha256sum "$ROOT/php.dom" | cut -d' ' -f1)"
