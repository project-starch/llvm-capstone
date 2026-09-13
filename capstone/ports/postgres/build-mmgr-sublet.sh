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
# Everything else the two builds share is in domain-build.sh beside this.
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

PG_PAYLOAD=${PG_PAYLOAD:-65536}
PG_ARENA=${PG_ARENA:-$((64 * 1024 * 1024))}
PG_TRACE=${PG_TRACE:-$((64 * 1024 * 1024))}
# The identity tables plus the one the data check adds. For the tpcb rung that
# is about twenty-five megabytes of capabilities and five of lengths, so the
# region is sized well above it rather than exactly: a region too small is a
# named refusal at entry, but a rebuild is minutes.
PG_SCRATCH=${PG_SCRATCH:-$((48 * 1024 * 1024))}
PG_DOMAIN_STACK=${PG_DOMAIN_STACK:-$((256 * 1024))}
PG_DOMAIN_DATA=${PG_DOMAIN_DATA:-$PG_DOMAIN_STACK}

# -O0 -g to read a fault, -O2 with no -g to measure, and never both: the fork's
# clang asserts in Value::stripAndAccumulateConstantOffsets when it emits debug
# info for optimised code on this target. Both arms must use the same level,
# and the run says which.
DOMAIN_OPT=${DOMAIN_OPT:--O0 -g}
PG_CHECK_DATA=${PG_CHECK_DATA--DREPLAY_CHECK_DATA}

FILES="aset.c mcxt.c generation.c slab.c bump.c alignedalloc.c memdebug.c"

source "$HERE/domain-build.sh"
pgdom_tools
pgdom_headers "$DOM_OUT" "$HERE/port/memorychunk-capstone.patch" \
                         "$HERE/port/memorychunk-sublet.patch"

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
       -DPG_REPLAY_SCRATCH_SIZE=${PG_SCRATCH}UL
       $PG_CHECK_DATA)

echo "== the manager, for capstone64, with the Sublet patch"
OBJS=()
pgdom_manager "$DOM_OUT" "$HERE/port/aset-capstone.patch" \
                         "$HERE/port/aset-sublet.patch"

echo "== the port"
for f in port/pg_stubs.c port/freestanding/pg_string.c \
         port/freestanding/pg_subpool.c port/freestanding/pg_subpool_libc.c \
         port/freestanding/pg_printf_domain.c tools/replay_sublet.c; do
  o=$DOM_OUT/obj/$(basename "${f%.c}").o
  "$CLANG" "${FLAGS[@]}" -c "$HERE/$f" -o "$o"
  OBJS+=("$o")
done

pgdom_link "$DOM_OUT" "$DOM_OUT/pg_mmgr_sublet.dom"

echo "declared dom_data $PG_DOMAIN_DATA (stack $PG_DOMAIN_STACK)"
echo "regions the host must make: payload $PG_PAYLOAD, arena $PG_ARENA (linear), trace $PG_TRACE, scratch $PG_SCRATCH"
# The number A7 reports, from the patch itself rather than recounted here: its
# header states the split between code and comment, which is the honest form,
# since a line of explanation is not a line of change.
sed -n '/^# *added/,/^# *removed/p' "$HERE/port/aset-sublet.patch" | sed 's/^# *//'
echo "built $DOM_OUT/pg_mmgr_sublet.dom"
