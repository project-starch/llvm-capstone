#!/usr/bin/env bash
# Build one Lua-CDP shim as a Capstone domain.
#
#   ROW=luac_openssl_ctx_uaf ./build-lua-capstone.sh
#   ROW=luac_openssl_ctx_uaf NO_REVOKE=1 ./build-lua-capstone.sh   # the control
#
# This reuses the mruby column's harness UNCHANGED — the allocator TU
# (mock_mruby_capstone.c, which provides the revoking malloc/free), the
# parameterised domain (xlang_shim_domain.c), start.S and the linker script.
# The ONLY new input is the shim under $HERE/shims. See build-xlang-capstone.sh
# for the rationale on every flag; this is that script with the shim dir and the
# mock/domain paths repointed at ../../capstone.
#
# -O0 is mandatory: at -O1+ the compiler hoists the cached pointer's load above
# the free or elides the dangling access, so the access the mechanism must police
# is never emitted. See ../../capstone/ALLOCATOR-CONTRACT.md §4.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

: "${CAPSTONE_REPO_ROOT:=$(cd "$HERE/../../.." && pwd)}"
export CAPSTONE_REPO_ROOT
# shellcheck source=/dev/null
source "$CAPSTONE_REPO_ROOT/capstone/tests/capstone-test-env.sh"

ROW=${ROW:-luac_openssl_ctx_uaf}
SHIM="$HERE/../shims/$ROW.c"   # shared shim dir — the SAME file the CHERI build compiles
[ -f "$SHIM" ] || { echo "no such shim: $SHIM" >&2; exit 2; }

# The revoke-on-free allocator TU is corpus-agnostic and lives in xlang/common/;
# the parameterised domain driver is reused from the mruby column (generic,
# neutrally named). Neither ties this column to mruby's *cases*.
COMMON_DIR="$CAPSTONE_REPO_ROOT/xlang/common"
MRUBY_DIR="$CAPSTONE_REPO_ROOT/xlang/capstone"          # domain driver + hostcall hdr
MOCK_SRC="$COMMON_DIR/revoke_arena_domain.c"            # was xlang/capstone/mock_mruby_capstone.c
DOMAIN_SRC="$MRUBY_DIR/xlang_shim_domain.c"
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

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/lua-cdp-capstone}
OUT_DOM=${OUT_DOM:-$OUT_DIR/xlang_$ROW$SUFFIX.dom}
OBJ_DIR="$OUT_DIR/obj_$ROW$SUFFIX"
mkdir -p "$OBJ_DIR"

for required in "$SHIM" "$MOCK_SRC" "$DOMAIN_SRC" "$START_SRC" "$LINKER_SCRIPT" \
                "$ROF_DIR/revoke_on_free_alloc.h"; do
  [ -f "$required" ] || { echo "missing build input: $required" >&2; exit 1; }
done

# ROW_SRC is resolved by the preprocessor relative to xlang_shim_domain.c's own
# directory (xlang/capstone), so it is the shim path FROM THERE.
ROW_SRC_REL="../lua-cdp/shims/$ROW.c"

FLAGS=(-target capstone64-unknown-elf
       -Xclang -target-feature -Xclang +m
       -ffreestanding -O0 -g
       -DROF_MAX_SLOTS=64
       -I"$MRUBY_DIR" -I"$ROF_DIR")

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$START_SRC" -o "$OBJ_DIR/start.o"

"$CLANG" "${FLAGS[@]}" -c "$MOCK_SRC" -o "$OBJ_DIR/mock.o"

"$CLANG" "${FLAGS[@]}" "${EXTRA[@]}" \
  -DROW_SRC="\"$ROW_SRC_REL\"" \
  -c "$DOMAIN_SRC" -o "$OBJ_DIR/domain.o"

"$LD_LLD" --gc-sections -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" "$OBJ_DIR/mock.o" "$OBJ_DIR/domain.o"

# The column measures revocation. If the revoke is not in the binary it measures
# nothing and a MISS is indistinguishable from a broken build — so verify the
# instruction is actually present rather than trusting the link.
ops=$("$CLANG" "${FLAGS[@]}" -S -o - "$MOCK_SRC" 2>/dev/null |
        grep -cE '^[[:space:]]+revoke\b' || true)
if [ "$ops" -eq 0 ]; then
  echo "BUILD-INVALID: no 'revoke' instruction in the allocator" >&2
  exit 3
fi

echo "built  $OUT_DOM"
echo "       revoke sites in allocator: $ops${SUFFIX:+  (control: revoke is runtime-suppressed)}"
