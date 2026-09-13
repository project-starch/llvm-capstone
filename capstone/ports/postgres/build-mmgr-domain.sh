#!/bin/bash
# PostgreSQL's memory manager as a Capstone domain, without the discipline.
#
#   build-mmgr-domain.sh
#
# The same seven files the host arm builds, for capstone64, linked into a
# domain image with the freestanding half of the port: the string functions,
# the level below, the printf helpers over the payload, and the replay driver.
#
# What differs from the host arm, and nothing else does: the compiler, the
# level below, and where formatted output goes. The loop is the same file.
#
# What differs from build-mmgr-sublet.sh is one patch on aset.c, one on the
# chunk header, the level below, and the driver. Everything else the two share
# is in domain-build.sh beside this.
#
# Needs the Capstone clang and lld. CAPSTONE_LLVM_BUILD_DIR if they are not at
# the default, and a configured PostgreSQL tree, which build-mmgr-host.sh
# leaves behind.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$HERE/../../.." && pwd)
PG_VERSION=${PG_VERSION:-17.5}
OUT=${OUT:-${CAPSTONE_TMP_ROOT:-/tmp}/pg-mmgr-host}
SRC=$OUT/postgresql-$PG_VERSION
DOM_OUT=${DOM_OUT:-$OUT/domain}

# The regions, and both halves must agree on their sizes, so they are build
# parameters here and the host is told what was used.
PG_PAYLOAD=${PG_PAYLOAD:-65536}
PG_ARENA=${PG_ARENA:-$((32 * 1024 * 1024))}
PG_TRACE=${PG_TRACE:-$((64 * 1024 * 1024))}
# dom_data and the stack. The manager does not recurse on data and the replay
# loop is flat, so this is the stack of a few frames, not an interpreter's.
PG_DOMAIN_STACK=${PG_DOMAIN_STACK:-$((256 * 1024))}
PG_DOMAIN_DATA=${PG_DOMAIN_DATA:-$PG_DOMAIN_STACK}

# Whether the replay checks that objects hold their contents and not only that
# the counts add up. On by default, because a discipline that revokes and
# re-hands memory could break the contract palloc makes without any counter
# noticing, and off with PG_CHECK_DATA= when a run is timing rather than
# checking: it reads and writes every object's bytes twice.
PG_CHECK_DATA=${PG_CHECK_DATA--DREPLAY_CHECK_DATA}

FILES="aset.c mcxt.c generation.c slab.c bump.c alignedalloc.c memdebug.c"

source "$HERE/domain-build.sh"
pgdom_tools
pgdom_headers "$DOM_OUT" "$HERE/port/memorychunk-capstone.patch"

FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
       -ffreestanding -fno-builtin -O0 -g
       -ffunction-sections -fdata-sections
       -nostdlibinc -isystem "$HERE/port/stubinc"
       -I"$DOM_OUT/include"
       -I"$SRC/src/include" -I"$SRC/src/backend"
       -I"$HERE" -I"$HERE/port" -I"$HERE/tools"
       -DPG_REPLAY_PAYLOAD_SIZE=${PG_PAYLOAD}UL
       -DPG_REPLAY_ARENA_SIZE=${PG_ARENA}UL
       -DPG_REPLAY_TRACE_SIZE=${PG_TRACE}UL
       $PG_CHECK_DATA)

echo "== the manager, for capstone64"
OBJS=()
# aset.c carries the two lines a sixteen-byte pointer forces; the allocator
# says so itself, with a static assertion. See port/aset-capstone.patch.
pgdom_manager "$DOM_OUT" "$HERE/port/aset-capstone.patch"

echo "== the port"
for f in port/pg_stubs.c port/freestanding/pg_string.c \
         port/freestanding/pg_level0.c port/freestanding/pg_printf_domain.c \
         tools/replay_domain.c; do
  o=$DOM_OUT/obj/$(basename "${f%.c}").o
  "$CLANG" "${FLAGS[@]}" -c "$HERE/$f" -o "$o"
  OBJS+=("$o")
done

pgdom_link "$DOM_OUT" "$DOM_OUT/pg_mmgr_capstone.dom"

echo "declared dom_data $PG_DOMAIN_DATA (stack $PG_DOMAIN_STACK)"
echo "regions the host must make: payload $PG_PAYLOAD, arena $PG_ARENA, trace $PG_TRACE"
echo "built $DOM_OUT/pg_mmgr_capstone.dom"
