#!/bin/bash
# PostgreSQL's memory manager under Sublet, as a Capstone domain.
#
#   build-mmgr-sublet.sh
#
# The same seven files build-mmgr-domain.sh builds, from the same tree, with
# one more patch on aset.c and a different level below. That is the whole of
# the difference between the two arms, and it is why they can be compared:
#
#   the manager      the same source, plus port/aset-sublet.patch
#   the chunk header the same, plus port/memorychunk-sublet.patch, which
#                    spends the padding a capability already forced on two
#                    indices rather than adding a byte
#   the level below  port/freestanding/pg_subpool.c, a sub-pool per context
#                    under one handle, instead of pg_level0.c's first fit
#   the loop         tools/replay_core.inc, identical, through replay_sublet.c
#
# The unported context types, generation, slab and bump, are still compiled
# because mcxt.c's method table names their functions. Their calls to malloc
# reach port/freestanding/pg_subpool_libc.c, which refuses loudly. The two
# recorded pgbench rungs create only aset contexts, so that is a guard and not
# a gap.
#
# Needs the Capstone clang and lld, and a configured PostgreSQL tree, which
# build-mmgr-host.sh leaves behind.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$HERE/../../.." && pwd)
PG_VERSION=${PG_VERSION:-17.5}
OUT=${OUT:-${CAPSTONE_TMP_ROOT:-/tmp}/pg-mmgr-host}
SRC=$OUT/postgresql-$PG_VERSION
DOM_OUT=${DOM_OUT:-$OUT/sublet}
LLVM=${CAPSTONE_LLVM_BUILD_DIR:-$REPO_ROOT/llvm/build-rel}
CLANG=${CLANG:-$LLVM/bin/clang}
LD_LLD=${LD_LLD:-$LLVM/bin/ld.lld}
READOBJ=${READOBJ:-$LLVM/bin/llvm-readobj}

START_SRC=$REPO_ROOT/capstone/my_first_domain/start.S
LINKER_SCRIPT=$REPO_ROOT/capstone/my_first_domain/link.ld
DOMREQ_SRC=$REPO_ROOT/capstone/tests/runtime-qemu/domreq.S

PG_PAYLOAD=${PG_PAYLOAD:-65536}
PG_ARENA=${PG_ARENA:-$((64 * 1024 * 1024))}
PG_TRACE=${PG_TRACE:-$((64 * 1024 * 1024))}
PG_SCRATCH=${PG_SCRATCH:-$((32 * 1024 * 1024))}
PG_DOMAIN_STACK=${PG_DOMAIN_STACK:-$((256 * 1024))}
PG_DOMAIN_DATA=${PG_DOMAIN_DATA:-$PG_DOMAIN_STACK}

# -O0 -g to read a fault, -O2 with no -g to measure, and never both: the fork's
# clang asserts in Value::stripAndAccumulateConstantOffsets when it emits debug
# info for optimised code on this target. Both arms must use the same level,
# and the run says which.
DOMAIN_OPT=${DOMAIN_OPT:--O0 -g}

FILES="aset.c mcxt.c generation.c slab.c bump.c alignedalloc.c memdebug.c"

for t in "$CLANG" "$LD_LLD"; do
  [ -x "$t" ] || { echo "no $t; set CAPSTONE_LLVM_BUILD_DIR" >&2; exit 1; }
done
[ -f "$SRC/src/include/pg_config.h" ] || {
  echo "no configured PostgreSQL tree at $SRC; run build-mmgr-host.sh first" >&2; exit 1; }

mkdir -p "$DOM_OUT/obj" "$DOM_OUT/include/utils"

# The headers configure got wrong for this target, shadowed rather than patched
# so that the host arm keeps the tree it was configured for.
cp "$SRC/src/include/utils/memutils_memorychunk.h" "$DOM_OUT/include/utils/"
patch -s -F0 -p0 "$DOM_OUT/include/utils/memutils_memorychunk.h" \
      < "$HERE/port/memorychunk-capstone.patch"
patch -s -F0 -p0 "$DOM_OUT/include/utils/memutils_memorychunk.h" \
      < "$HERE/port/memorychunk-sublet.patch"
sed 's/^#define MAXIMUM_ALIGNOF 8$/#define MAXIMUM_ALIGNOF 16/' \
    "$SRC/src/include/pg_config.h" > "$DOM_OUT/include/pg_config.h"
grep -qx '#define MAXIMUM_ALIGNOF 16' "$DOM_OUT/include/pg_config.h" || {
  echo "MAXIMUM_ALIGNOF was not 8 in $SRC/src/include/pg_config.h; read it and decide" >&2
  exit 2; }

FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
       -ffreestanding -fno-builtin $DOMAIN_OPT
       -ffunction-sections -fdata-sections
       -nostdlibinc -isystem "$HERE/port/stubinc"
       -I"$DOM_OUT/include"
       -I"$SRC/src/include" -I"$SRC/src/backend"
       -I"$REPO_ROOT/capstone/sublet"
       -I"$HERE" -I"$HERE/port" -I"$HERE/tools"
       -DPG_REPLAY_PAYLOAD_SIZE=${PG_PAYLOAD}UL
       -DPG_REPLAY_ARENA_SIZE=${PG_ARENA}UL
       -DPG_REPLAY_TRACE_SIZE=${PG_TRACE}UL
       -DPG_REPLAY_SCRATCH_SIZE=${PG_SCRATCH}UL)

echo "== the manager, for capstone64, with the Sublet patch"
OBJS=()
for f in $FILES; do
  src=$SRC/src/backend/utils/mmgr/$f
  if [ "$f" = aset.c ]; then
    cp "$src" "$DOM_OUT/aset.c"
    patch -s -F0 -p0 "$DOM_OUT/aset.c" < "$HERE/port/aset-capstone.patch"
    patch -s -F0 -p0 "$DOM_OUT/aset.c" < "$HERE/port/aset-sublet.patch"
    src=$DOM_OUT/aset.c
  fi
  "$CLANG" "${FLAGS[@]}" -c "$src" -o "$DOM_OUT/obj/${f%.c}.o"
  OBJS+=("$DOM_OUT/obj/${f%.c}.o")
done

echo "== the port"
# The repository's freestanding string set, because its copies preserve tags.
BEEBS_STRING=$REPO_ROOT/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c
"$CLANG" "${FLAGS[@]}" -c "$BEEBS_STRING" -o "$DOM_OUT/obj/beebs_string.o"
OBJS+=("$DOM_OUT/obj/beebs_string.o")

for f in port/pg_stubs.c port/freestanding/pg_string.c \
         port/freestanding/pg_subpool.c port/freestanding/pg_subpool_libc.c \
         port/freestanding/pg_printf_domain.c tools/replay_sublet.c; do
  o=$DOM_OUT/obj/$(basename "${f%.c}").o
  "$CLANG" "${FLAGS[@]}" -c "$HERE/$f" -o "$o"
  OBJS+=("$o")
done

echo "== the domain's entry and its declared requirement"
"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
    -ffreestanding -O0 -c "$START_SRC" -o "$DOM_OUT/obj/start.o"
"$CLANG" -target capstone64-unknown-elf -ffreestanding \
    -DCAPSTONE_DOMREQ_DATA=$PG_DOMAIN_DATA \
    -DCAPSTONE_DOMREQ_STACK=$PG_DOMAIN_STACK \
    -c "$DOMREQ_SRC" -o "$DOM_OUT/obj/domreq.o"

OUT_DOM=$DOM_OUT/pg_mmgr_sublet.dom
_segs() { "$READOBJ" --program-headers "$OUT_DOM" | grep -E 'Offset|VirtualAddress|FileSize|MemSize'; }

echo "== linking"
"$LD_LLD" --gc-sections -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
    "$DOM_OUT/obj/start.o" "${OBJS[@]}"
_before=$(_segs)
"$LD_LLD" --gc-sections -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
    "$DOM_OUT/obj/start.o" "${OBJS[@]}" "$DOM_OUT/obj/domreq.o"
if [[ "$(_segs)" != "$_before" ]]; then
  echo "domreq.S moved a loaded byte; the declaration must be non-alloc" >&2
  exit 2
fi

# The image has a ceiling and it is not obvious from the failure. The module
# asks the buddy allocator for the image doubled, rounded up to a power-of-two
# page count, and that allocator stops at order ten, four megabytes. An image
# above two megabytes therefore cannot be created at all, and the guest says
# only "create_dom failed". Checked here so that it says why.
_mem=$("$READOBJ" --program-headers "$OUT_DOM" | awk '/MemSize/ { print $2; exit }')
if [ "${_mem:-0}" -gt $((2 * 1024 * 1024)) ]; then
  echo "the domain image is $_mem bytes, above the two megabytes the module can create:" >&2
  echo "  it asks the buddy allocator for the image doubled, and that stops at four." >&2
  echo "  The static tables in port/pg_subpool.h are the usual reason; they are sized" >&2
  echo "  from the recording and can be overridden with -DPG_CHUNK_MAX and friends." >&2
  exit 2
fi
echo "image $_mem bytes of the two megabytes the module can create"
echo "declared dom_data $PG_DOMAIN_DATA (stack $PG_DOMAIN_STACK)"
echo "regions the host must make: payload $PG_PAYLOAD, arena $PG_ARENA (linear), trace $PG_TRACE, scratch $PG_SCRATCH"
# The number A7 reports, from the patch itself rather than recounted here:
# its header states the split between code and comment, which is the honest
# form, since a line of explanation is not a line of change.
sed -n '/^# *added/,/^# *removed/p' "$HERE/port/aset-sublet.patch" | sed 's/^# *//'

echo "built $OUT_DOM"
