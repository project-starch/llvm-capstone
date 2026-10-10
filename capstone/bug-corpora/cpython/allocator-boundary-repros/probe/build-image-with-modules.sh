#!/bin/bash
# build4.sh <arm> <variant-suffix>
#
# Same as build3 but the variant name and the heap are parameters, so two
# module images can coexist instead of one overwriting the other.
#
# Why a second sublet image at all: sublet-mod at the default 48 MiB heap runs
# out of application heap DURING IMPORT -- cases 15, 19, 23 and 31 all died in
# importlib, three with "error return without exception set" and one with
# MemoryError. That arm revokes on every free, so it was already near its limit
# and two more builtin modules put it over. Two of those four, 19 and 31, are
# controls that WERE detections on the base image, so the module image loses
# more rows than it recovers there. The control set is the only reason that was
# visible: running just 15 and 23 would have shown two CAPACITY rows and looked
# like "still cannot fix it".
#
# Only CPY_HEAP_BYTES is raised. The failure was a Python-level allocation, so
# it is the application arena that is short, not pymalloc's tables -- leaving
# PYM_ARENA_BYTES and friends at their defaults keeps the confound as small as
# it can be, and the same controls will say whether even this much moves a
# previously measured row.
set -u
# Everything below has to be named by whoever runs this. Naming one author's
# miniconda and home directory here is what made the earlier copy of this script
# work on exactly one machine, and a wrong toolchain does not fail loudly: a
# missing cmake once surfaced only as "0 binaries built" in an unread log.
for tool in cmake ninja; do
  command -v "$tool" >/dev/null || { echo "$tool is not on PATH" >&2; exit 2; }
done
python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' || {
  echo "python3 is $(python3 -V 2>&1); the port's scripts need 3.11 or later" >&2; exit 2; }
: "${CAPSTONE_LLVM_BUILD_DIR:?set CAPSTONE_LLVM_BUILD_DIR to the LLVM build directory}"
[ -x "$CAPSTONE_LLVM_BUILD_DIR/bin/clang" ] || {
  echo "no clang under $CAPSTONE_LLVM_BUILD_DIR/bin" >&2; exit 2; }
export CAPSTONE_LLVM_BUILD_DIR
: "${CPY_NATIVE_BUILD:?set CPY_NATIVE_BUILD to a native CPython 3.13.7 build directory}"
NATIVE=$CPY_NATIVE_BUILD
[ -x "$NATIVE/python" ] || { echo "no python in $NATIVE" >&2; exit 2; }
export CPY_BUILD_PYTHON=$NATIVE/python
# Locate the repo from this script, not from one author's home directory.
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
W=${REPO:-$(cd "$HERE/../../../../.." && pwd)}
[ -d "$W/capstone/ports/cpython/app" ] || {
  echo "cannot find the repo from $HERE; set REPO" >&2; exit 2; }
APP=$W/capstone/ports/cpython/app
KIT=/tmp/capstone/cpython-arms
STORE=$HOME/arms/cpython/images
mkdir -p "$KIT/images" "$STORE"

arm=${1:?usage: build4.sh spatial|sublet <variant>}
VAR=${2:?usage: build4.sh spatial|sublet <variant>}
case $arm in
  spatial) SUB=0 ;;
  sublet)  SUB=1 ;;
  *) echo "unknown arm $arm" >&2; exit 2 ;;
esac
ROOT=$KIT/tree4-$arm-$VAR
LOG=$KIT/build4-$arm-$VAR.log
: > "$LOG"
HEAPNOTE=${CPY_HEAP_BYTES:+heap=$((CPY_HEAP_BYTES>>20))MiB }
echo "=== $arm-$VAR: CPY_SUBLET=$SUB ${HEAPNOTE:-default capacity}+ the two test-capi modules"
t0=$(date +%s)

CPY_ROOT=$ROOT CPY_SUBLET=$SUB bash "$APP/prepare-cpython-capstone.sh" >>"$LOG" 2>&1
rc=$?; BUILD=$(tail -1 "$LOG")
[ $rc -eq 0 ] && [ -d "${BUILD:-}" ] || { echo "  prepare failed (rc=$rc):"; tail -12 "$LOG" | sed 's/^/    /'; exit 2; }
echo "  prepare ok $(( $(date +%s) - t0 ))s"

LIMITED="_testlimitedcapi _testlimitedcapi.c"
for f in abstract bytearray bytes complex dict eval float heaptype_relative import \
         list long object pyos set sys tuple unicode vectorcall_limited file; do
  LIMITED="$LIMITED _testlimitedcapi/$f.c"
done
{
  echo ""
  echo "# Added 2026-10-09 for cases 15 and 23. In Setup.local's *static* block"
  echo "# because prepare refuses a module in Setup.stdlib's *shared* block: a"
  echo "# domain has no dlopen, and --disable-test-modules is what puts them there."
  echo "*static*"
  echo "_testinternalcapi _testinternalcapi.c _testinternalcapi/test_lock.c _testinternalcapi/pytime.c _testinternalcapi/set.c _testinternalcapi/test_critical_sections.c"
  echo "$LIMITED"
} >> "$BUILD/Modules/Setup.local"

(cd "$BUILD" && make Makefile >>"$LOG" 2>&1) || { echo "  make Makefile failed"; tail -12 "$LOG" | sed 's/^/    /'; exit 2; }
for m in _testinternalcapi _testlimitedcapi; do
  grep -q "$m" "$BUILD/Makefile" || { echo "  REFUSING: $m not in the Makefile"; exit 2; }
done
echo "  Makefile carries both modules"

ENV=$BUILD/capstone-env.sh
[ -f "$ENV" ] || { echo "  prepare left no $ENV"; exit 2; }
. "$ENV"
[ -n "${CAPSTONE_CLANG:-}" ] || { echo "  $ENV did not set CAPSTONE_CLANG"; exit 2; }

CPY_ROOT=$ROOT CPY_SUBLET=$SUB python3 "$APP/survey-cpython-capstone.py" "$BUILD" --jobs 12 >>"$LOG" 2>&1
[ $? -eq 0 ] || { echo "  survey failed:"; tail -25 "$LOG" | sed 's/^/    /'; exit 2; }
echo "  survey ok  $(( $(date +%s) - t0 ))s"

CPY_ROOT=$ROOT CPY_SUBLET=$SUB python3 "$APP/link-cpython-capstone.py" "$BUILD" \
  --native "$NATIVE" --out "$ROOT/link" >>"$LOG" 2>&1
IMG=$ROOT/link/python.dom
[ -f "$IMG" ] || { echo "  link failed:"; tail -25 "$LOG" | sed 's/^/    /'; exit 2; }
sha=$(sha256sum "$IMG" | cut -d' ' -f1)
for d in "$STORE" "$KIT/images"; do cp "$IMG" "$d/python-$arm-$VAR.dom"; done
grep -v "^$arm-$VAR	" "$KIT/images/inputs.tsv" > "$KIT/images/inputs.tsv.new" 2>/dev/null || true
printf '%s\t%s\t%s\t%s\n' "$arm-$VAR" \
  "${HEAPNOTE}+ _testinternalcapi + _testlimitedcapi (PYM_* at defaults)" "$sha" "$(date -u +%FT%TZ)" \
  >> "$KIT/images/inputs.tsv.new"
mv "$KIT/images/inputs.tsv.new" "$KIT/images/inputs.tsv"
cp "$KIT/images/inputs.tsv" "$STORE/inputs.tsv"
echo "  $arm-$VAR done $(( $(date +%s) - t0 ))s  sha256=${sha:0:16}"
