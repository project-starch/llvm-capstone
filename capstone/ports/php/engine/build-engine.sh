#!/usr/bin/env bash
# Build the PHP engine domain. Phase 2 rung A: link + zend_startup.
#   capstone/container/run.sh capstone/ports/php/engine/build-engine.sh [rung]
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../tests/capstone-test-env.sh"
R=$CAPSTONE_REPO_ROOT
P=${PHP_SRC:-/corpus/build/php-5.0.0}
[ -d "$P/Zend" ] || { echo "PHP source not at $P -- corpus not mounted?" >&2; exit 2; }
OUT=${OUT:-$CAPSTONE_TMP_ROOT/php-engine}
mkdir -p "$OUT/obj" "$OUT/log"
RUNG=${1:-A}

# See compile-sweep.sh for why -fno-common and why implicit declarations are not allowed.
# -gline-tables-only: fault-locate.py turns a capability fault into a source line, and
# without line tables it can only name the nearest preceding symbol -- which on a
# --gc-sections link with many static functions is routinely the wrong function.
CF=(-target capstone64-unknown-elf
    # +m IS MANDATORY. Without the M extension there is no hardware multiply, so EVERY
    # multiply becomes a libcall to __muldi3 -- including the two 32-bit multiplies INSIDE
    # compiler-rt's own __muldi3 (muldi3.c: x.s.high * y.s.low + x.s.low * y.s.high). That
    # makes __muldi3 call itself, unboundedly. Measured: without +m it self-calls twice and
    # a single zend_hash_init_ex exhausts a 3.87 MB stack; with +m it makes zero calls and
    # the same call returns correctly. ports/sqlite passes this flag for the same reason.
    -Xclang -target-feature -Xclang +m
    -ffreestanding -fno-builtin -O0 -std=gnu89 -fno-common
    -gline-tables-only
    -nostdlibinc -isystem "$HERE/stubinc"
    -I"$P" -I"$P/Zend" -I"$P/main" -I"$P/TSRM" -I"$P/ext" -I"$P/ext/standard")

# THE HEAP. zend_capstone_alloc.h defaults to a 16 KB arena and 64 slots, which is right
# for the standalone CRASH-008/UAF probes and nowhere near enough for the engine: running
# zend_startup registers hundreds of hash entries, zend_arena_carve returns NULL, and the
# first caller that does not check -- zend_ini_startup, whose malloc feeds straight into
# zend_hash_init_ex -- passes a NULL HashTable and stores through it (cause 24, in
# _zend_hash_init). That was diagnosed as a capability defect for some time; it is simply
# heap exhaustion. 512 KB / 4096 slots gets all of zend_startup through.
ZEND_ARENA_BYTES=${ZEND_ARENA_BYTES:-524288}
ZEND_MAX_SLOTS=${ZEND_MAX_SLOTS:-4096}
CF+=(-DZEND_ARENA_BYTES=$ZEND_ARENA_BYTES -DZEND_MAX_SLOTS=$ZEND_MAX_SLOTS)

# The arena lives in .bss, so it is funded out of the SAME declared dom_data as the stack.
# The stack is 512 KB rather than 1 MB because the +m fix removed the __muldi3 recursion
# that made rung B eat 2.6 MB; measured alloca high-water on the startup path is under 1 KB.
# Raising the DECLARATION instead is not available: at data=3 MB the MONITOR itself faults
# during create_domain (cause 5 at pc 0x8002164c, a 16-byte read one grain below its own
# bounds), so the total must stay on the geometry that works.

# SHRINK-STACK IS OFF, AND THAT IS A FIT REQUIREMENT, NOT A PREFERENCE.
#
# capstone-shrink-stack defaults ON in the compiler and narrows every address-taken stack
# object to its size. On this engine it costs ~33% of .text (645,204 -> 959,884), and that
# does not fit: the domain total is code + the declaration, against a 4 MiB buddy ceiling.
#
#   off:  loadable 1,834,828 + declared 2,097,152 = 3,931,980   fits, 262 KB spare
#   on:   loadable 2,149,508 + declared 2,097,152 = 4,246,660   OVER by ~52 KB
#
# Exceeding it does NOT fail cleanly. The MONITOR faults inside create_domain -- cause 5 at
# pc = 0x8002164c, pcc_base = 0x8001a1d0, a 16-byte read at 0x8008ffd0 against bounds
# starting 0x8008ffe0, one grain below its own base -- and every stage then reports that
# same fault, which reads like a domain bug and is not one. Seen twice: at data=3 MB with
# narrowing off, and at data=2 MB with narrowing on. Worth filing against the monitor.
#
# WHAT IT COSTS: stack objects get whole-frame bounds, so a stack overflow is no longer
# caught. Acceptable here -- every corpus bug this port targets is in an emalloc'd buffer,
# which our allocator bounds precisely -- but it is a real reduction in what the port
# demonstrates, and it should be revisited once caplifive-buildroot 2b8ad05 (large domains
# from CMA, tested to 32 MiB) lands and the 4 MiB wall goes away.
#
# Narrowing was ALSO tested as a suspect for the stage-9 fault and was not the cause.
CF+=(-mllvm -capstone-shrink-stack=false)

# EXTRA_CF: space-separated extra flags, for A/B-ing a codegen option across the whole
# engine without editing this file (e.g. EXTRA_CF="-mllvm -capstone-shrink-stack=false").
# shellcheck disable=SC2206
[ -n "${EXTRA_CF:-}" ] && CF+=(${EXTRA_CF})

# PHP_DIAG_ALLOC=1: compile a PATCHED COPY of Zend/zend_alloc.c instead of the pristine one,
# with a csdebugprint (.insn r 0x5b,0x1,0x43 -- prints "Print = Cap(...)" when tagged and
# "Print = Scalar(0x..)" when not) on the pointer _emalloc hands back inside _estrndup.
#
# WHY A COPY AND A FLAG. Every Zend TU is normally byte-identical to the corpus tree and the
# matched-pair result depends on that, so the experiment arms must never see this. It exists to
# answer one question the tag watch cannot: is the pointer ALREADY untagged when _estrndup
# receives it, or does it lose the tag afterwards? Everything else in the producer chain has been
# eliminated, and this is the one link never directly observed.
ZEND_ALLOC_SRC="$P/Zend/zend_alloc.c"
if [ "${PHP_DIAG_ALLOC:-0}" = "1" ]; then
  ZEND_ALLOC_SRC="$OUT/zend_alloc_diag.c"
  sed "/_emalloc(length+1/r $HERE/diag/estrndup-probe.inc" \
      "$P/Zend/zend_alloc.c" > "$ZEND_ALLOC_SRC"
  if ! grep -q "0x43" "$ZEND_ALLOC_SRC"; then
    echo "  PHP_DIAG_ALLOC: the probe did not splice in -- refusing to build a silent no-op" >&2
    exit 2
  fi
  echo "  PHP_DIAG_ALLOC: probing _estrndup's view of _emalloc (diagnostic build, NOT an experiment arm)"
fi

TUS="zend_language_scanner zend_language_parser zend_compile zend_execute zend_execute_API
 zend_opcode zend_operators zend_variables zend_hash zend_API zend_alloc zend_mm zend
 zend_llist zend_ptr_stack zend_stack zend_constants zend_list zend_qsort zend_stream
 zend_objects zend_objects_API zend_object_handlers zend_exceptions zend_interfaces
 zend_iterators zend_builtin_functions zend_extensions zend_ini zend_default_classes
 zend_ts_hash zend_sprintf zend_dynamic_array zend_multibyte"

OBJS=()
ZCF=("${CF[@]}")
# PHP_DEPTH_WATCH instruments only the Zend TUs; the support layer must stay uninstrumented
# or the hook recurses into itself.
# -finstrument-functions is NOT usable here: it crashes clang on this target with
# "Calling a function with a bad signature!" (llvm/lib/IR/Instructions.cpp:759) because the
# __cyg_profile hooks' void* params do not match the capability pointer type. The depth
# watchdog lives in malloc() instead -- see libc/php_capstone_malloc.c.
for t in $TUS; do
  _src="$P/Zend/$t.c"; [ "$t" = "zend_alloc" ] && _src="$ZEND_ALLOC_SRC"
  "$CAPSTONE_CLANG" "${ZCF[@]}" -c "$_src" -o "$OUT/obj/$t.o" 2>"$OUT/log/$t.log" \
    || { echo "FAILED $t"; head -5 "$OUT/log/$t.log"; exit 1; }
  OBJS+=("$OUT/obj/$t.o")
done
echo "  zend: ${#OBJS[@]} objects"

# Support layer. beebs_freestanding_string.c is REUSED, not rewritten: it carries the
# tag-preserving ldc/stc memcpy. A byte-loop memcpy strips the tag off any capability it
# copies, and PHP copies zvals -- whose union holds pointers -- constantly.
SUP=("$R/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c"
     "$HERE/libc/php_capstone_libc.c" "$HERE/libc/php_capstone_stdio.c"
     "$HERE/libc/php_capstone_os.c"   "$HERE/libc/php_capstone_malloc.c"
     "$HERE/libc/php_capstone_php_stubs.c" "$HERE/libc/php_capstone_depth.c"
     "$HERE/libc/php_capstone_ext_stubs.c")

# ext/standard TUs, compiled BYTE-IDENTICAL from the corpus tree. url.c compiles against the
# real php.h with no source change once stubinc supplies the headers php.h reaches for
# (sys/stat.h, sys/socket.h, netinet/in.h, arpa/inet.h, netdb.h, dirent.h, utime.h, time.h);
# it is 23,264 bytes of .text. Its only undefined symbols outside the engine are the two
# streams entry points reached from PHP_FUNCTION(get_headers) plus php_error_docref1, all
# three supplied by libc/php_capstone_ext_stubs.c.
EXT="$P/ext/standard/url.c"
for s in $EXT; do
  b=$(basename "$s" .c)
  "$CAPSTONE_CLANG" "${ZCF[@]}" -c "$s" -o "$OUT/obj/ext_$b.o" 2>"$OUT/log/ext_$b.log" \
    || { echo "FAILED ext/$b"; head -20 "$OUT/log/ext_$b.log"; exit 1; }
  OBJS+=("$OUT/obj/ext_$b.o")
done
for s in "${SUP[@]}"; do
  b=$(basename "$s" .c)
  "$CAPSTONE_CLANG" "${CF[@]}" -c "$s" -o "$OUT/obj/$b.o" 2>"$OUT/log/$b.log" \
    || { echo "FAILED $b"; head -20 "$OUT/log/$b.log"; exit 1; }
  OBJS+=("$OUT/obj/$b.o")
done

for a in "$HERE/libc/capstone_setjmp.S" "$R/capstone/my_first_domain/start.S" \
         "$R/capstone/tests/runtime-qemu/gct-section-end.S"; do
  b=$(basename "$a" .S)
  "$CAPSTONE_CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
    -ffreestanding -O0 -c "$a" -o "$OUT/obj/$b.o" || exit 1
  OBJS+=("$OUT/obj/$b.o")
done

# 64-bit divide and soft float, compiled from the in-tree compiler-rt exactly as
# ports/sqlite/build-sqlite-capstone.sh does.
B="$R/compiler-rt/lib/builtins"
for b in divdi3 moddi3 udivdi3 umoddi3 udivmoddi4 muldi3 adddf3 subdf3 muldf3 divdf3 \
         fixdfdi fixdfsi fixunsdfdi fixunsdfsi floatdidf floatsidf floatundidf floatunsidf \
         comparedf2 fp_mode; do
  [ -f "$B/$b.c" ] || continue
  "$CAPSTONE_CLANG" "${CF[@]}" -c "$B/$b.c" -o "$OUT/obj/rt_$b.o" 2>"$OUT/log/rt_$b.log" \
    || { echo "  builtin $b FAILED"; continue; }
  OBJS+=("$OUT/obj/rt_$b.o")
done

"$CAPSTONE_CLANG" "${CF[@]}" -c "$HERE/rung_${RUNG}_domain.c" -o "$OUT/obj/domain.o" \
  2>"$OUT/log/domain.log" || { echo "FAILED domain"; head -20 "$OUT/log/domain.log"; exit 1; }
OBJS+=("$OUT/obj/domain.o")

# DECLARE WHAT THE DOMAIN NEEDS.
#
# Without .capstone_domreq the module sizes headroom as max(2*code_len, 512 KiB), a proxy
# that is not causally linked to how deep the domain recurses. Rung B proved that the hard
# way: zend_startup faulted with
#     rs1 = x8 (frame pointer), addr = 0x101d27f98, bounds = (0x101d28000, ...)
# i.e. a write 104 bytes BELOW the stack capability's base -- stack exhaustion, not a logic
# bug. PHP's compiler and VM recurse, and at -O0 with 16-byte pointers a frame is about
# twice its size on an ordinary target (the same reason ports/sqlite declares 1 MiB).
#
# DOMREQ_DATA is the WHOLE dom_data requirement (initialiser blob + cap table + per-global
# storage + stack), because that is the one number the module can use without knowing the
# carve. DOMREQ_STACK is carried alongside for diagnostics.
# 512 KB, not 1 MB: the arena above shares this declaration, and the +m fix removed the
# recursion that needed a big stack. See the note beside ZEND_ARENA_BYTES.
PHP_DOMAIN_STACK=${PHP_DOMAIN_STACK:-$((512 * 1024))}
PHP_DOMAIN_DATA=${PHP_DOMAIN_DATA:-$((2 * 1024 * 1024))}
"$CAPSTONE_CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 \
  -DCAPSTONE_DOMREQ_DATA=$PHP_DOMAIN_DATA -DCAPSTONE_DOMREQ_STACK=$PHP_DOMAIN_STACK \
  -c "$R/capstone/tests/runtime-qemu/domreq.S" -o "$OUT/obj/domreq.o" || exit 1

echo "  linking $((${#OBJS[@]} + 1)) objects (declaring data=$PHP_DOMAIN_DATA stack=$PHP_DOMAIN_STACK) ..."
_segs() { "$CAPSTONE_LLVM_BIN/llvm-readobj" --program-headers "$1" | grep -E "Offset|VirtualAddress|FileSize|MemSize"; }
"$CAPSTONE_LD_LLD" --gc-sections -T "$R/capstone/my_first_domain/link.ld" \
  -o "$OUT/php_rung${RUNG}.dom" "${OBJS[@]}" 2>"$OUT/log/link.log"
rc=$?
if [ $rc -eq 0 ]; then
  before=$(_segs "$OUT/php_rung${RUNG}.dom")
  "$CAPSTONE_LD_LLD" --gc-sections -T "$R/capstone/my_first_domain/link.ld" \
    -o "$OUT/php_rung${RUNG}.dom" "${OBJS[@]}" "$OUT/obj/domreq.o" 2>>"$OUT/log/link.log"
  rc=$?
  # The declaration is non-SHF_ALLOC and MUST NOT perturb the loaded image; the build
  # asserts it rather than trusting it, exactly as ports/postgres/domain-build.sh does.
  if [ $rc -eq 0 ] && [ "$(_segs "$OUT/php_rung${RUNG}.dom")" != "$before" ]; then
    echo "  domreq.S moved a loaded byte -- the declaration must be non-alloc" >&2; exit 2
  fi
fi
if [ $rc -ne 0 ]; then
  echo "  LINK FAILED -- unresolved symbols:"
  grep -oE "undefined symbol: .*" "$OUT/log/link.log" | sort -u | sed 's/^/    /' | head -40
  exit 1
fi
echo "  built $OUT/php_rung${RUNG}.dom"
"$CAPSTONE_LLVM_BIN/llvm-objdump" -h "$OUT/php_rung${RUNG}.dom" \
  | awk '$2==".text"||$2==".data"||$2==".bss"||$2==".rodata"{printf "    %-10s %d\n",$2,strtonum("0x" $3)}'
