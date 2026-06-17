#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=aha-mont64

source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
"$SCRIPT_DIR/fetch-beebs.sh" >/dev/null

BEEBS_SRC_DIR=${BEEBS_SRC_DIR:-$CAPSTONE_TMP_ROOT/beebs-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/beebs-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj-$BEEBS_BENCHMARK}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$CAPSTONE_REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$CAPSTONE_REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_${BEEBS_BENCHMARK}_capstone.dom}

SUPPORT_DIR=$BEEBS_SRC_DIR/support
SRC=$BEEBS_SRC_DIR/src/aha-mont64/mont64.c
PATCHED=$OUT_DIR/${BEEBS_BENCHMARK}_src.c

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# Capstone backend Bug #13: 64-bit constants whose lower 32 bits need both
# lui and addi (bits[31:12] != 0 AND bits[11:0] != 0) crash the backend.
# Workaround: hoist such constants to static global variables initialized in
# .data (static initializers are stored as raw bytes in the ELF, not as
# instruction-level constant materialization sequences).
python3 - "$SRC" "$PATCHED" <<'PYEOF'
import sys, re

src_path, dst_path = sys.argv[1], sys.argv[2]
with open(src_path) as f:
    text = f.read()

# Strip hosted includes; prepend type stubs.
text = re.sub(r'^#include <(stdio|stdlib|stdint)\.h>\n', '', text, flags=re.MULTILINE)

const_decls = []
const_map = {}   # hex_string → var_name
counter = [0]

def needs_hoist(val):
    """Return True if this 64-bit constant triggers Bug #13."""
    lo32 = val & 0xffffffff
    if lo32 == 0:
        return False
    # Needs lui (bits[31:12] != 0) AND addi (bits[11:0] != 0).
    return (lo32 >> 12) != 0 and (lo32 & 0xfff) != 0

def replace_const(m):
    raw = m.group(0)
    val = int(raw[:-2], 16)  # strip LL/UL suffix
    if not needs_hoist(val):
        return raw
    key = f'0x{val:016x}'
    if key not in const_map:
        name = f'_mont64_const_{counter[0]}'
        counter[0] += 1
        const_map[key] = name
        const_decls.append(f'static uint64_t {name} = {key}UL;')
    return const_map[key]

text = re.sub(r'0x[0-9a-fA-F]{5,}(?:LL|UL)', replace_const, text)

prefix = (
    'typedef unsigned long uint64_t;\n'
    'typedef long int64_t;\n'
)
with open(dst_path, 'w') as f:
    f.write(prefix)
    if const_decls:
        f.write('\n'.join(const_decls) + '\n')
    f.write(text)
PYEOF

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature
  -Xclang +m
  -ffreestanding
  -fno-builtin
  -fno-jump-tables
  "$DOMAIN_OPT_LEVEL"
  -I"$SUPPORT_DIR"
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 \
  -c "$START_SRC" \
  -o "$OBJ_DIR/start.o"

"$CLANG" "${COMMON_FLAGS[@]}" -c "$PATCHED" -o "$OBJ_DIR/mont64.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" \
  -o "$OBJ_DIR/beebs_simple_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" "$OBJ_DIR/mont64.o" "$OBJ_DIR/beebs_simple_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"
echo "Built $OUT_DOM"
