#!/usr/bin/env bash
# Phase 0 gate: how much of the Zend engine compiles for capstone64, and how big is it?
#
#   capstone/container/run.sh capstone/ports/php/engine/compile-sweep.sh
#
# Reports a pass/fail ratio per translation unit and the total .text of what linked,
# modelled on ports/musl-capstone's 1355/1361 survey. This decides whether the domain
# image ceiling is the binding constraint at all -- the largest domain that has ever run
# in this tree is ~1.62 MB of code.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../tests/capstone-test-env.sh"

P=${PHP_SRC:-/corpus/build/php-5.0.0}
[ -d "$P/Zend" ] || { echo "PHP source not at $P -- is the corpus mounted?" >&2; exit 2; }

OUT=${OUT:-$CAPSTONE_TMP_ROOT/php-engine}
rm -rf "$OUT"; mkdir -p "$OUT/obj" "$OUT/log"

# The REQUIRED set from the dependency census. zend_reflection_api is deliberately absent:
# 3426 lines and the single largest object (57 KB .text), not on the execution path.
TUS="zend_language_scanner zend_language_parser zend_compile zend_execute zend_execute_API
 zend_opcode zend_operators zend_variables zend_hash zend_API zend_alloc zend_mm zend
 zend_llist zend_ptr_stack zend_stack zend_constants zend_list zend_qsort zend_stream
 zend_objects zend_objects_API zend_object_handlers zend_exceptions zend_interfaces
 zend_iterators zend_builtin_functions zend_extensions zend_ini zend_default_classes
 zend_ts_hash zend_sprintf zend_dynamic_array zend_multibyte"

# -nostdlibinc, NOT -nostdinc: the latter also hides clang's own freestanding headers
# (stddef.h, stdarg.h, limits.h, float.h), which PHP legitimately needs and which are not
# a libc. -fno-builtin so a memcpy the compiler would otherwise synthesise still goes
# through the freestanding one we supply.
# -std=gnu89 matches PHP's own CFLAGS_CLEAN. Two deliberate departures from it:
#
#   -fno-common, NOT PHP's -fcommon. -fcommon makes a tentative definition a COMMON
#   symbol, and the Capstone path then asks getSectionPrefixForGlobal() for a section for
#   it. That function (llvm/lib/CodeGen/TargetLoweringObjectFileImpl.cpp:616-632) handles
#   Text/ReadOnly/BSS/ThreadData/ThreadBSS/Data/ReadOnlyWithRel and NOT Common, so it hits
#   llvm_unreachable("Unknown section kind") and the compiler ABORTS (exit 134). Measured:
#   6 of 6 affected TUs fail with -fcommon and 6 of 6 compile with -fno-common. -fno-common
#   is clang's default since 11 and the modern correct choice anyway; the cost is that
#   tentative definitions duplicated across TUs become link errors instead of being merged,
#   which is a link-time problem with an obvious fix, not a silent one.
#
#   NOT -Wno-implicit-function-declaration, which PHP does pass. On this target an
#   implicitly-declared pointer-returning function truncates a capability to int -- see
#   stubinc/dlfcn.h. Every such function is declared instead.
FLAGS=(-target capstone64-unknown-elf
    # +m IS MANDATORY. Without the M extension there is no hardware multiply, so EVERY
    # multiply becomes a libcall to __muldi3 -- including the two 32-bit multiplies INSIDE
    # compiler-rt's own __muldi3 (muldi3.c: x.s.high * y.s.low + x.s.low * y.s.high). That
    # makes __muldi3 call itself, unboundedly. Measured: without +m it self-calls twice and
    # a single zend_hash_init_ex exhausts a 3.87 MB stack; with +m it makes zero calls and
    # the same call returns correctly. ports/sqlite passes this flag for the same reason.
    -Xclang -target-feature -Xclang +m
    -ffreestanding -fno-builtin -O0 -std=gnu89 -fno-common
       -nostdlibinc -isystem "$HERE/stubinc"
       -I"$P" -I"$P/Zend" -I"$P/main" -I"$P/TSRM" -I"$P/ext" -I"$P/ext/standard")

pass=0; fail=0; failed=""
for t in $TUS; do
  src="$P/Zend/$t.c"
  if [ ! -f "$src" ]; then echo "  MISSING  $t.c"; fail=$((fail+1)); failed="$failed $t"; continue; fi
  if "$CAPSTONE_CLANG" "${FLAGS[@]}" -c "$src" -o "$OUT/obj/$t.o" >"$OUT/log/$t.log" 2>&1; then
    pass=$((pass+1))
  else
    fail=$((fail+1)); failed="$failed $t"
    printf '  FAIL  %-28s %s\n' "$t" "$(grep -cE ': error' "$OUT/log/$t.log") errors"
  fi
done

echo
echo "compiled: $pass / $((pass+fail))"
[ -n "$failed" ] && echo "failures:$failed"

echo
echo "=== .text of what compiled ==="
# llvm-size is not in the toolchain's built target set; objdump -h is.
"$CAPSTONE_LLVM_BIN/llvm-objdump" -h "$OUT"/obj/*.o 2>/dev/null \
  | awk '$2==".text" { t += strtonum("0x" $3) }
         $2==".data" { d += strtonum("0x" $3) }
         $2==".bss"  { b += strtonum("0x" $3) }
         END { printf "  text %d  data %d  bss %d\n", t, d, b }'
echo "  (largest objects)"
for o in "$OUT"/obj/*.o; do
  printf '%8d %s\n' "$("$CAPSTONE_LLVM_BIN/llvm-objdump" -h "$o" | awk '$2==".text"{print strtonum("0x" $3)}')" "$(basename "$o")"
done | sort -rn | head -8
