#!/usr/bin/env bash
# Build the PHP port for the VIRTUAL Capstone profile -- a capability application that runs as
# an ordinary Linux user process, not a physical domain.
#
#   capstone/container/run.sh capstone/ports/php/app/build-virtual.sh [probe|php]
#
# WHY THIS IS A NEW SCRIPT AND NOT A FLAG ON build-engine.sh. The physical build is a hand-rolled
# clang/ld.lld invocation against my_first_domain/link.ld, with -ffreestanding, -nostdlibinc and
# a 29-header stubinc farm standing in for libc. Under the virtual profile the SDK owns the
# triple, the ISA features, the headers and the link script -- ports/common/cmake/toolchains/
# capstone-application.cmake says so explicitly, and warns that restating any of them creates a
# second source of truth. Nothing of the physical command line survives, and the loader's
# .capstone_virtual ELF marker refuses an image built the other way, so there is no shared path
# to flag between them. build-engine.sh stays exactly as it is: it is the comparison column.
#
# JOBS IS DEFAULTED DOWN, DELIBERATELY. runtime/virtual/build-sdk.sh defaults to JOBS=16 for
# musl's 1361 translation units. On a host that is also running something else -- and this one
# has no swap -- 16 is how you OOM the machine and take the other work with it. 4 is the default
# here; raise it with JOBS= when the machine is idle.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../tests/capstone-test-env.sh"
R=$CAPSTONE_REPO_ROOT
WHAT=${1:-probe}

OUT=${PHP_VIRTUAL_OUT:-$CAPSTONE_TMP_ROOT/virtual-runtime}
SDK="$OUT/sdk"
CC="$SDK/capstone-cc"
export MUSL_CACHE_ROOT=${MUSL_CACHE_ROOT:-$OUT/musl-source}

# ---- 1. the musl tree ------------------------------------------------------------------
# prepare-musl-capstone.sh is idempotent: a patch that already applies in reverse is skipped,
# and one that fits neither way is an error rather than a silently partial tree.
MUSL=$(bash "$R/capstone/ports/musl-capstone/prepare-musl-capstone.sh" 2>/dev/null | tail -1)
[ -d "$MUSL/arch/capstone64" ] || { echo "musl not prepared (got '$MUSL')" >&2; exit 2; }
echo "  musl: $MUSL"

# ---- 2. the SDK -----------------------------------------------------------------------
# Rebuilt only when absent or when PHP_VIRTUAL_FRESH_SDK=1. The archive is 1361 objects and
# nothing in this port changes it, so rebuilding per run is pure cost.
if [ ! -x "$CC" ] || [ "${PHP_VIRTUAL_FRESH_SDK:-0}" = 1 ]; then
  echo "  building the virtual SDK (JOBS=${JOBS:-4}) ..."
  JOBS=${JOBS:-4} bash "$R/capstone/runtime/virtual/build-sdk.sh" "$OUT" "$MUSL" \
      ${PHP_VIRTUAL_SDK_ARGS:-} || { echo "SDK build FAILED" >&2; exit 1; }
else
  echo "  SDK: reusing $SDK (PHP_VIRTUAL_FRESH_SDK=1 to rebuild)"
fi
[ -x "$CC" ] || { echo "no capstone-cc at $CC" >&2; exit 1; }

# The SDK records what it is. An image built against the physical profile is rejected by the
# launcher with ENOEXEC rather than misbehaving, but the error arrives after a boot -- cheaper
# to assert it here.
# sdk.json carries no "profile" key, so key on the two things that actually differ:
# -capstone-image-gp in codegen (mandatory for virtual, absent for physical) and a libc
# archive named ...-virtual.a. Checked here because the alternative is a boot and ENOEXEC.
if ! grep -q 'capstone-image-gp' "$SDK/sdk.json" || ! grep -q 'libc-capstone-virtual' "$SDK/sdk.json"; then
  echo "SDK at $SDK is NOT the virtual profile:" >&2
  grep -E '"codegen"|"libc"' "$SDK/sdk.json" >&2
  echo "  (delete $SDK and re-run, or check CAPSTONE_APPLICATION_PROFILE)" >&2
  exit 1
fi

# ---- 3. what to build -----------------------------------------------------------------
case "$WHAT" in
  probe)
    # The seam probe: can a lent block be split so that retiring the payload leaves PHP's
    # header live? Everything in the allocator plan depends on the answer.
    SRC="$R/capstone/ports/php/zend-alloc/sublet/probe-sublet-split.c"
    IMG="$OUT/probe-sublet-split.dom"
    echo "  compiling the seam probe ..."
    # -I $R/capstone so <sublet/sublet.h> resolves to capstone/sublet/sublet.h, which is the
    # header the virtual heap itself uses (runtime/virtual/vm.h includes ../../sublet/sublet.h).
    "$CC" -O1 -I"$R/capstone" "$SRC" -o "$IMG" || { echo "probe build FAILED" >&2; exit 1; }
    echo "  built $IMG"
    "$CAPSTONE_LLVM_BIN/llvm-readobj" --sections "$IMG" 2>/dev/null \
      | awk '/Name: .capstone_virtual/{f=1} f&&/Size:/{print "    .capstone_virtual size: " $2; exit}'
    "$CAPSTONE_LLVM_BIN/llvm-objdump" -h "$IMG" 2>/dev/null \
      | awk '$2==".text"||$2==".data"||$2==".bss"{printf "    %-10s %d\n",$2,strtonum("0x" $3)}'
    ;;
  php)
    echo "  PHP itself is not wired up yet -- the seam probe comes first (see the plan)." >&2
    exit 2
    ;;
  *) echo "usage: build-virtual.sh [probe|php]" >&2; exit 2 ;;
esac
