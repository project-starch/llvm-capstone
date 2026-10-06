#!/bin/bash
# build-arms.sh [arm...]: one domain image per arm, from one source tree.
#
# The four Capstone arms differ ONLY in the heap the image links, which is the
# whole reason the unprotected one exists (docs/ref/runtime-terms-glossary.md
# section 6). CPYD_HEAP names the arm rather than the knob, because the same
# knob value means different arms in different ports and a result that records
# the knob cannot be read a month later.
#
#   sysalloc-none     level0 heap, per-object bounds OFF
#   sysalloc-bounds   level0 heap as applications get it since PR #170 -- BASELINE
#   sysalloc-sublet   the Sublet heap, pymalloc NOT sublet
#   sublet-pymalloc   the Sublet heap AND patch 0014, so pymalloc's pools and
#                     arenas are issued and revoked too
#
# The last two are the pair this corpus exists for: a nested defect should be
# invisible to the first and visible to the second.
#
# Four steps per arm, because three of them can fail quietly:
#   prepare   configures CPython against this arm's SDK
#   survey    compiles CPython's own object list with its own flags. A loop over
#             *.c would not reproduce them, and the survey is written to be able
#             to fail rather than to pass
#   link      the strict link; nothing may be left undefined
#   descriptor  REGION_DATA must be well under 4 MiB. At CONTEXTS=15 it is
#             exactly the buddy allocator's largest block and capstone-exec
#             refuses the image with "cannot allocate launch regions" -- a
#             failure that happens at RUN time, one arm and hours later, so it
#             is checked here
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../../../.." && pwd)}
KIT=${KIT:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/cpython-arms}
APP=$REPO/capstone/ports/cpython/app
ARMS=${*:-"sysalloc-none sysalloc-bounds sysalloc-sublet sublet-pymalloc"}

# What prepare needs and does not derive. CPY_BUILD_PYTHON is a host CPython of
# the same version: cross-configure needs one to run its own tools, and a
# different version silently changes generated sources. --native wants the TREE.
export CAPSTONE_LLVM_BUILD_DIR=${CAPSTONE_LLVM_BUILD_DIR:-$HOME/llvm-capstone/llvm/cmake-build-debug}
NATIVE=${NATIVE:-$HOME/arms/cpython/shared/build-plain}
export CPY_BUILD_PYTHON=${CPY_BUILD_PYTHON:-$NATIVE/python}
JOBS=${JOBS:-12}
for v in CAPSTONE_LLVM_BUILD_DIR CPY_BUILD_PYTHON NATIVE; do
  [[ -e ${!v} ]] || { echo "$v=${!v} does not exist" >&2; exit 2; }
done
OBJCOPY=$CAPSTONE_LLVM_BUILD_DIR/bin/llvm-objcopy
[[ -x $OBJCOPY ]] || { echo "no llvm-objcopy at $OBJCOPY" >&2; exit 2; }

mkdir -p "$KIT/images"
touch "$KIT/images/inputs.tsv"

for arm in $ARMS; do
  case $arm in
    sysalloc-none)    heap=none ;;
    sysalloc-bounds)  heap=bounds ;;
    sysalloc-sublet)  heap=sublet ;;
    sublet-pymalloc)  heap=sublet-pymalloc ;;
    *) echo "unknown arm $arm" >&2; exit 2 ;;
  esac
  # Each arm gets its own tree. Sharing one CPY_ROOT would have a build
  # overwrite an image a run is still reading, which has happened in this lane.
  ROOT=$KIT/tree-$arm
  LOG=$KIT/build-$arm.log
  echo "=== $arm (CPYD_HEAP=$heap) -> $LOG"
  t0=$(date +%s)

  BUILD=$(CPY_ROOT=$ROOT CPYD_HEAP=$heap bash "$APP/prepare-cpython-capstone.sh" 2>>"$LOG" | tail -1)
  [[ -d ${BUILD:-} ]] || { echo "$arm: prepare gave no build dir (see $LOG)" >&2; exit 2; }
  echo "  prepare ok: $BUILD  ($(( $(date +%s) - t0 ))s)"

  CPY_ROOT=$ROOT CPYD_HEAP=$heap python3 "$APP/survey-cpython-capstone.py" \
      "$BUILD" --jobs "$JOBS" >>"$LOG" 2>&1 \
    || { echo "$arm: survey failed (see $LOG)" >&2; exit 2; }
  echo "  survey ok  ($(( $(date +%s) - t0 ))s)"

  CPY_ROOT=$ROOT CPYD_HEAP=$heap python3 "$APP/link-cpython-capstone.py" \
      "$BUILD" --native "$NATIVE" --out "$ROOT/link" >>"$LOG" 2>&1 \
    || { echo "$arm: link failed (see $LOG)" >&2; exit 2; }
  IMG=$ROOT/link/python.dom
  [[ -f $IMG ]] || { echo "$arm: no $IMG after a link that returned 0" >&2; exit 2; }

  # REGION_DATA gate: 7 u64s in the application descriptor; transports is
  # 1 + contexts, and REGION_DATA is transports * exchange.
  "$OBJCOPY" --dump-section .capstone_application="$KIT/desc-$arm.bin" "$IMG" /dev/null 2>/dev/null
  python3 - "$KIT/desc-$arm.bin" <<'PY' || exit 2
import struct, sys
f = struct.unpack_from("<7Q", open(sys.argv[1], "rb").read(), 0)
t = 1 + f[6]
mib = t * f[5] / 2**20
print("  descriptor: contexts=%d exchange=%d transports=%d REGION_DATA=%.2f MiB"
      % (f[6], f[5], t, mib))
if mib >= 3.5:
    sys.exit("  REFUSING: REGION_DATA %.2f MiB is at or near the buddy allocator's "
             "4 MiB maximum block; capstone-exec would refuse this image at run time" % mib)
PY

  cp "$IMG" "$KIT/images/python-$arm.dom"
  sha=$(sha256sum "$KIT/images/python-$arm.dom" | cut -d" " -f1)
  # The hash is the only thing that says later what was actually run, so it is
  # recorded beside the image rather than in a run log that can be separated
  # from it. Replace any earlier line for this arm instead of appending.
  grep -v "^$arm	" "$KIT/images/inputs.tsv" > "$KIT/images/inputs.tsv.new" 2>/dev/null || true
  printf '%s\t%s\t%s\t%s\n' "$arm" "$heap" "$sha" "$(date -u +%FT%TZ)" >> "$KIT/images/inputs.tsv.new"
  mv "$KIT/images/inputs.tsv.new" "$KIT/images/inputs.tsv"
  echo "  $arm done in $(( $(date +%s) - t0 ))s  sha256=${sha:0:16}"
done
echo
echo "images and hashes: $KIT/images/inputs.tsv"
cat "$KIT/images/inputs.tsv"
