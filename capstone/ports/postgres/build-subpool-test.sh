#!/bin/bash
# The level below, under Sublet, as a domain that tests itself.
#
#   build-subpool-test.sh
#
# Small on purpose: no PostgreSQL source is needed, because the piece under
# test is the port's own and the manager does not call it yet. That is also why
# it is worth building separately. A fault inside a patched aset.c is hard to
# read back to its cause, and this image narrows the ground it could be on to
# one file.
#
# Needs the Capstone clang and lld, CAPSTONE_LLVM_BUILD_DIR if they are not at
# the default.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$HERE/../../.." && pwd)
OUT=${OUT:-${CAPSTONE_TMP_ROOT:-/tmp}/pg-subpool-test}
LLVM=${CAPSTONE_LLVM_BUILD_DIR:-$REPO_ROOT/llvm/build-rel}
CLANG=${CLANG:-$LLVM/bin/clang}
LD_LLD=${LD_LLD:-$LLVM/bin/ld.lld}
READOBJ=${READOBJ:-$LLVM/bin/llvm-readobj}

START_SRC=$REPO_ROOT/capstone/my_first_domain/start.S
LINKER_SCRIPT=$REPO_ROOT/capstone/my_first_domain/link.ld
DOMREQ_SRC=$REPO_ROOT/capstone/tests/runtime-qemu/domreq.S

# The host makes the same regions the replay makes, so one host serves both and
# the fourth region is shared and ignored.
PG_PAYLOAD=${PG_PAYLOAD:-65536}
PG_ARENA=${PG_ARENA:-$((32 * 1024 * 1024))}
PG_TRACE=${PG_TRACE:-$((1 * 1024 * 1024))}
PG_DOMAIN_STACK=${PG_DOMAIN_STACK:-$((256 * 1024))}
PG_DOMAIN_DATA=${PG_DOMAIN_DATA:-$PG_DOMAIN_STACK}

# -O0 -g to read a fault, -O2 with no -g to measure, and never both: the fork's
# clang asserts in Value::stripAndAccumulateConstantOffsets when it emits debug
# info for optimised code on this target, because a capability is wider than
# the index type the offset arithmetic assumes. -O1, -O2 and -Og all crash with
# -g and all pass without it. Every build script in this repository uses -O0,
# which is why the crash has not been met before; it is named here because a
# cycle count at -O0 is not a cycle count, and the measurement has to say which
# it used.
DOMAIN_OPT=${DOMAIN_OPT:--O0 -g}

for t in "$CLANG" "$LD_LLD"; do
  [ -x "$t" ] || { echo "no $t; set CAPSTONE_LLVM_BUILD_DIR" >&2; exit 1; }
done

mkdir -p "$OUT/obj"

FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
       -ffreestanding -fno-builtin $DOMAIN_OPT
       -ffunction-sections -fdata-sections
       -nostdlibinc -isystem "$HERE/port/stubinc"
       -I"$REPO_ROOT/capstone/sublet"
       -I"$HERE" -I"$HERE/port" -I"$HERE/tools"
       -DPG_REPLAY_PAYLOAD_SIZE=${PG_PAYLOAD}UL
       -DPG_REPLAY_ARENA_SIZE=${PG_ARENA}UL
       -DPG_REPLAY_TRACE_SIZE=${PG_TRACE}UL)

# The two files this port owns are built warning-clean, and the warnings are
# on for them: a cast that loses a capability is the mistake this target
# punishes hardest, and the compiler names it.
OBJS=()
for f in port/freestanding/pg_subpool.c tools/subpool_domain.c; do
  o=$OUT/obj/$(basename "${f%.c}").o
  "$CLANG" "${FLAGS[@]}" -Wall -Wextra -Werror -c "$HERE/$f" -o "$o"
  OBJS+=("$o")
done

# The printf helpers and the repository's freestanding string set. Neither is
# under test and both cast a pointer to an integer on purpose, the first to
# print an address and the second to ask whether a copy is aligned, so the
# warnings are off rather than answered here.
for f in "$HERE/port/freestanding/pg_printf_domain.c" \
         "$REPO_ROOT/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c"; do
  o=$OUT/obj/$(basename "${f%.c}").o
  "$CLANG" "${FLAGS[@]}" -c "$f" -o "$o"
  OBJS+=("$o")
done

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
    -ffreestanding -O0 -c "$START_SRC" -o "$OUT/obj/start.o"
"$CLANG" -target capstone64-unknown-elf -ffreestanding \
    -DCAPSTONE_DOMREQ_DATA=$PG_DOMAIN_DATA \
    -DCAPSTONE_DOMREQ_STACK=$PG_DOMAIN_STACK \
    -c "$DOMREQ_SRC" -o "$OUT/obj/domreq.o"

OUT_DOM=$OUT/pg_subpool_test.dom
_segs() { "$READOBJ" --program-headers "$OUT_DOM" | grep -E 'Offset|VirtualAddress|FileSize|MemSize'; }

"$LD_LLD" --gc-sections -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
    "$OUT/obj/start.o" "${OBJS[@]}"
_before=$(_segs)
"$LD_LLD" --gc-sections -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
    "$OUT/obj/start.o" "${OBJS[@]}" "$OUT/obj/domreq.o"
# The declaration is non-alloc, so nothing loaded may move. Verified and not
# asserted: four added instructions have flipped a passing run on this project.
if [[ "$(_segs)" != "$_before" ]]; then
  echo "domreq.S moved a loaded byte; the declaration must be non-alloc" >&2
  exit 2
fi

echo "declared dom_data $PG_DOMAIN_DATA (stack $PG_DOMAIN_STACK)"
echo "regions the host must make: payload $PG_PAYLOAD, arena $PG_ARENA, trace $PG_TRACE"
echo "built $OUT_DOM"
