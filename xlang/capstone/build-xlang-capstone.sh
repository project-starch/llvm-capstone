#!/usr/bin/env bash
# Build one xlang shim as a Capstone domain.
#
#   ROW=mruby_range_uaf ./build-xlang-capstone.sh
#   ROW=mruby_range_uaf NO_REVOKE=1 ./build-xlang-capstone.sh   # the control
#
# Far smaller than build-sqlite-capstone.sh: the shims need no SQLite, no VFS,
# no OS shim and no soft-float builtins (every slot is a uint64_t). Three
# objects: start.S, the mock, the parameterised domain.
#
# -O0 is NOT a preference. At -O1+ the compiler hoists the cached pointer's load
# above the free or elides the dangling access entirely, so the access the
# mechanism must police is never emitted and the row "passes" for the wrong
# reason. The CHERI column builds -O0 for the same reason.
# See ALLOCATOR-CONTRACT.md §4.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

: "${CAPSTONE_REPO_ROOT:=$(cd "$HERE/../.." && pwd)}"
export CAPSTONE_REPO_ROOT
# shellcheck source=/dev/null
source "$CAPSTONE_REPO_ROOT/capstone/tests/capstone-test-env.sh"

ROW=${ROW:-mruby_range_uaf}
SHIM="$HERE/../cheri/shims/$ROW.c"
[ -f "$SHIM" ] || { echo "no such shim: $SHIM" >&2; exit 2; }

CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
START_SRC=${START_SRC:-$CAPSTONE_REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$CAPSTONE_REPO_ROOT/capstone/my_first_domain/link.ld}
ROF_DIR="$CAPSTONE_REPO_ROOT/capstone/benchmarks/sqlite"   # revoke_on_free_alloc.h

SUFFIX=""
EXTRA=()
if [ -n "${NO_REVOKE:-}" ]; then
  SUFFIX="_norevoke"
  EXTRA+=(-DXLANG_NO_REVOKE)
fi
# Diagnostic build: reports what the share callbacks delivered and returns,
# without touching the allocator. Use it to tell "share plumbing broken" apart
# from "allocator faults" -- a bare cslcc assertion cannot.
if [ -n "${PROBE:-}" ]; then
  SUFFIX="${SUFFIX}_probe"
  EXTRA+=(-DXLANG_PROBE_SHARES)
fi
# Delivery probe: allocate / revoke / dereference at offset 0, no arithmetic.
# Answers whether our allocator's revoke reaches the MODELLED fault path
# (cause 25) rather than QEMU's assert-on-untagged gap.
if [ -n "${DELIVERY:-}" ]; then
  SUFFIX="${SUFFIX}_delivery"
  EXTRA+=(-DXLANG_PROBE_DELIVERY)
fi

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/xlang-capstone}
OUT_DOM=${OUT_DOM:-$OUT_DIR/xlang_$ROW$SUFFIX.dom}
OBJ_DIR="$OUT_DIR/obj_$ROW$SUFFIX"
mkdir -p "$OBJ_DIR"

for required in "$SHIM" "$START_SRC" "$LINKER_SCRIPT" \
                "$ROF_DIR/revoke_on_free_alloc.h"; do
  [ -f "$required" ] || { echo "missing build input: $required" >&2; exit 1; }
done

# ROF_MAX_SLOTS defaults to 16384, i.e. 512 KiB of .bss for a domain that makes
# about five allocations. 64 is ample: the widest row allocates mrb_state,
# mrb_context, the stack, the wedge, and one realloc.
FLAGS=(-target capstone64-unknown-elf
       -Xclang -target-feature -Xclang +m
       -ffreestanding -O0 -g
       -DROF_MAX_SLOTS=64
       -I"$HERE" -I"$ROF_DIR")

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$START_SRC" -o "$OBJ_DIR/start.o"

"$CLANG" "${FLAGS[@]}" -c "$HERE/mock_mruby_capstone.c" -o "$OBJ_DIR/mock.o"

"$CLANG" "${FLAGS[@]}" "${EXTRA[@]}" \
  -DROW_SRC="\"../cheri/shims/$ROW.c\"" \
  -c "$HERE/xlang_shim_domain.c" -o "$OBJ_DIR/domain.o"

"$LD_LLD" --gc-sections -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" "$OBJ_DIR/mock.o" "$OBJ_DIR/domain.o"

# The column measures revocation. If the revoke is not in the binary it measures
# nothing, and a MISS would be indistinguishable from a broken build -- so check
# the instruction is actually there rather than trusting that it linked.
ops=$("$CLANG" "${FLAGS[@]}" -S -o - "$HERE/mock_mruby_capstone.c" 2>/dev/null |
        grep -cE '^[[:space:]]+revoke\b' || true)
if [ "$ops" -eq 0 ]; then
  echo "BUILD-INVALID: no 'revoke' instruction in the allocator" >&2
  exit 3
fi

echo "built  $OUT_DOM"
echo "       revoke sites in allocator: $ops${SUFFIX:+  (control: revoke is runtime-suppressed)}"
