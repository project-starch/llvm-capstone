#!/usr/bin/env bash
# nginx's pool, in a Capstone domain. Upstream ngx_palloc.c byte for byte, this port's shim in
# place of ngx_core.h's forty-seven headers, the PostgreSQL port's pg_level0.c as the level below
# so both ports stand on the same one, and a driver that checks what it wrote.
#
# Two link passes like the other ports: the globals offset is not known until the first link says
# how large the text is, and under -capstone-gp-captable that offset decides the carve.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
source "$REPO/capstone/tests/capstone-test-env.sh"

bash "$SCRIPT_DIR/fetch-nginx.sh" >/dev/null
NGX_SRC_DIR=${NGX_SRC_DIR:-$CAPSTONE_TMP_ROOT/nginx-${NGX_VERSION:-1.28.0}}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/nginx-domain}
OBJ=$OUT_DIR/obj; mkdir -p "$OBJ"
DOM_NAME=${DOM_NAME:-ngx-pool}
CLANG=$CAPSTONE_CLANG
LD_LLD=$CAPSTONE_LD_LLD
LADDER=$REPO/capstone/tests/runtime-qemu/silicon-ladder
GPFREE=$REPO/capstone/tests/runtime-qemu/gp-free-domain
PG=$REPO/capstone/ports/postgres/port/freestanding

# The allocator, with only its two nginx includes redirected. Everything else is upstream, which
# is what the port-effort count needs to stay honest.
cp "$NGX_SRC_DIR/src/core/ngx_palloc.c" "$NGX_SRC_DIR/src/core/ngx_palloc.h" "$OBJ/"
sed -i 's|#include <ngx_config.h>|#include "ngx_shim.h"|; s|#include <ngx_core.h>||' "$OBJ/ngx_palloc.c"
sed -i 's|#include <ngx_config.h>||; s|#include <ngx_core.h>||' "$OBJ/ngx_palloc.h"
cp "$SCRIPT_DIR/adapted/ngx_shim.h" "$OBJ/"

CFLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
        -mllvm -capstone-gp-captable -ffreestanding -fno-jump-tables
        -std=c99 -O0 -w ${NGX_STOP_AFTER:+-DNGX_STOP_AFTER=$NGX_STOP_AFTER} -I"$OBJ" -I"$SCRIPT_DIR/port" -I"$PG/../stubinc")

# ONE TRANSLATION UNIT, and not for build speed. Under -capstone-gp-captable the image gets a
# .capstone_gp_initdesc block per translation unit that has globals, and the entry glue reads only
# the first, so an image built from several objects has a capability table for the globals of one
# of them. See I-8; capstone/tests/gp-initdesc-blocks.py refuses an image that got this wrong, and
# this build runs it below. Every other port in this repository amalgamates for the same reason,
# which nobody had written down.
#
# It is about fifteen hundred lines all told, one clang process and no parallel make, so it is not
# the kind of compile that has taken this machine down before.
AMALGAM=$OBJ/ngx_all.c
{
  echo '#include "ngx_shim.h"'
  # The level below first: the allocator above it must see malloc and free declared, and both
  # files are included rather than copied so the two ports stand on the same level 0.
  echo "#include \"$PG/pg_level0.c\""
  echo "#include \"$PG/pg_string.c\""
  echo "#include \"$REPO/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c\""
  echo "#include \"$SCRIPT_DIR/port/ngx_level0_wrap.c\""
  echo "#include \"$OBJ/ngx_palloc.c\""
  echo "#include \"$SCRIPT_DIR/port/ngx_domain.c\""
} > "$AMALGAM"
printf "   amalgam: %s lines\n" "$(cat "$OBJ/ngx_palloc.c" "$PG/pg_level0.c" "$PG/pg_string.c" \
    "$REPO/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c" \
    "$SCRIPT_DIR/port/ngx_level0_wrap.c" "$SCRIPT_DIR/port/ngx_domain.c" | wc -l)"
"$CLANG" "${CFLAGS[@]}" -c -o "$OBJ/ngx_all.o" "$AMALGAM"

link() {  # $1 = globals offset literal, $2 = output
  local lds="$OBJ/link.ld"
  sed "s/0x10000 + 0x1000/0x10000 + $1/" "$GPFREE/link-gpfree.ld" > "$lds"
  "$CLANG" -target capstone64-unknown-elf -ffreestanding \
      -c "$LADDER/start-gp-captable-interp.S" -o "$OBJ/start.o"
  # Ends the .gct section so the entry glue can compute the table's extent. Without it
  # __capstone_gct_end is undefined, the glue carves a table of whatever the arithmetic then
  # yields, and the first global past that faults with "Cap mem access OOB". Which is what
  # happened: four entries carved against twenty-four wanted.
  "$CLANG" -target capstone64-unknown-elf -ffreestanding \
      -c "$LADDER/../gct-section-end.S" -o "$OBJ/gct.o"
  "$LD_LLD" -T "$lds" -o "$2" "$OBJ/start.o" "$OBJ/ngx_all.o" "$OBJ/gct.o"
}

# Pass one at an offset large enough that it cannot overlap itself, only to learn the text size.
link 0x800000 "$OUT_DIR/pass1.dom"
TEXT=$("$CAPSTONE_LLVM_BIN/llvm-readelf" -SW "$OUT_DIR/pass1.dom" | python3 -c '
import re, sys
for line in sys.stdin:
    m = re.search(r"\s\.text\s+\S+\s+\S+\s+\S+\s+(\S+)", line)
    if m:
        print(int(m.group(1), 16)); break')
GOFF=$(( ((TEXT + 0xFFFF) / 0x10000) * 0x10000 ))
printf "   .text = %s bytes -> globals offset 0x%x\n" "$TEXT" "$GOFF"
link "$(printf '0x%x' $GOFF)" "$OUT_DIR/$DOM_NAME.dom"

# The gates the MicroPython port accumulated, for the same reasons. Cheap, and each one has cost
# somebody a boot that said nothing.
DIS=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$OUT_DIR/$DOM_NAME.dom")
NCJALR=$(grep -cE '\bcjalr\b' <<<"$DIS" || true)
NGPACC=$(grep -cE '(ldc[[:space:]].*\(gp\)|cincoffset[[:space:]]+[a-z0-9]+,[[:space:]]*gp,)' <<<"$DIS" || true)
NHDR=$("$CAPSTONE_LLVM_BIN/llvm-readelf" -SW "$OUT_DIR/$DOM_NAME.dom" | grep -c "capstone_gp_table" || true)
NEND=$("$CAPSTONE_LLVM_BIN/llvm-nm" "$OUT_DIR/$DOM_NAME.dom" | grep -c "__capstone_gct_end" || true)
echo "   cjalr=$NCJALR  gp-accesses=$NGPACC  gp_table sections=$NHDR  gct_end=$NEND"
[[ "$NCJALR" == "0" ]] || { echo "FAIL: cjalr present, not gp-free" >&2; exit 1; }
[[ "$NGPACC" -ge 1 ]]  || { echo "FAIL: no gp[i] global access" >&2; exit 1; }
[[ "$NHDR" == "1" ]]   || { echo "FAIL: want exactly one gp-table header, got $NHDR" >&2; exit 1; }
[[ "$NEND" == "1" ]]   || { echo "FAIL: __capstone_gct_end missing, the glue cannot size the table" >&2; exit 1; }

# I-8: one descriptor block, or the glue carves a table for the wrong count and the domain faults
# before the program runs. This is the gate that was missing when that happened.
python3 "$REPO/capstone/tests/gp-initdesc-blocks.py" "$OUT_DIR/$DOM_NAME.dom" | sed 's/^/   /'

echo "Built $OUT_DIR/$DOM_NAME.dom"
