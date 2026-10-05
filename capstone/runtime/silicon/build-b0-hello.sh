#!/usr/bin/env bash
# B0 (docs/plans/b0-silicon-delegated-runtime.md): build b0-hello.c as a SILICON-ABI delegated application.
#
# - gp-captable: globals are reached through a capability table built at entry (start-gp-captable-interp.S with
#   CAPSTONE_GLUE_YIELD), not through QEMU's fabricated gp.
# - Full LTO: the application, the runtime, the builtins and the musl members it pulls in become ONE module, so
#   the cap-table slots are globally unique (I-8). Every C object is bitcode; a native object that defines a global
#   would bring the slot collision back.
# - The four silicon -mllvm options are re-passed to the LTO plugin: codegen happens in the linker.
# - Two-pass link: pass 1 measures .text, pass 2 places the globals region above it.
#
# Required: CAPSTONE_LLVM_BIN (a toolchain with dev's CURRENT Capstone backend -- the plan's toolchain section),
#           B0_MUSL_ARCHIVE (the gp-captable LTO archive: MUSL_CAPSTONE_EXTRA_CFLAGS, see the plan).
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAP=$(cd "$HERE/../.." && pwd)
BIN=${CAPSTONE_LLVM_BIN:?set CAPSTONE_LLVM_BIN to the current toolchain}
ARCHIVE=${B0_MUSL_ARCHIVE:?set B0_MUSL_ARCHIVE to the gp-captable LTO musl archive}
MUSL=${PORT_MUSL_ROOT:-/tmp/capstone/musl-src/musl-1.2.5}
OUT=${OUT_DIR:-/tmp/capstone/b0/hello}
# B0_APP: which application in this directory to build (b0-hello, or b0-memcpy for B0.8); the image is $OUT/$B0_APP.dom.
APP=${B0_APP:-b0-hello}
[ -n "${B0_APP_SRCS:-}" ] || [ -f "$(dirname "$0")/$APP.c" ] || { echo "no application $APP.c" >&2; exit 2; }
CC=$BIN/clang
[ -f "$MUSL/obj/include/bits/alltypes.h" ] || { echo "no prepared musl headers at $MUSL" >&2; exit 2; }
rm -rf "$OUT"; mkdir -p "$OUT/obj"

SIL=(-mllvm -capstone-gp-captable -mllvm -capstone-shrink-stack=false
     -mllvm -capstone-shrink-globals=false -mllvm -capstone-merge-string-constants=true)
PLUG=(--plugin-opt=-capstone-gp-captable --plugin-opt=-capstone-shrink-stack=false
      --plugin-opt=-capstone-shrink-globals=false --plugin-opt=-capstone-merge-string-constants=true)
RES=$("$CC" -print-resource-dir)
BASE=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m -Xclang -target-feature -Xclang +a
      -ffreestanding -fno-builtin -nostdinc
      -isystem "$MUSL/arch/capstone64" -isystem "$MUSL/arch/generic" -isystem "$MUSL/obj/include"
      -isystem "$MUSL/include" -isystem "$RES/include"
      -ffunction-sections -fdata-sections -fno-jump-tables -Wno-int-conversion -O1 -flto
      -DCAPSTONE_GP_CAPTABLE_ABI=1 "${SIL[@]}" -I"$CAP/runtime/include" -D_XOPEN_SOURCE=700 ${B0_CFLAGS_EXTRA:-})
# delegate-bench's sizes (runtime/tests/application/CMakeLists.txt): the small ones.
# B1: B0_CONTEXT_BYTES > 0 builds minted contexts in (the glue's CAPSTONE_GLUE_CONTEXTS; the arena is split off the
# top of dom_data at the first entry), with B0_CONTEXTS of them able to run at once (Application.cmake's CONTEXTS).
# B0_DATA and B0_ARENA override the data region and the level0 arena (thread stacks come from the latter).
DATA=${B0_DATA:-262144}; STACK=65536; ARENA=${B0_ARENA:-65536}; EXCH=65536
CTXB=${B0_CONTEXT_BYTES:-0}; NCTX=${B0_CONTEXTS:-0}
[ $((CTXB % 4096)) -eq 0 ] || { echo "B0_CONTEXT_BYTES must be a multiple of 4096" >&2; exit 2; }
APPDEFS=(-DCAPSTONE_DOMREQ_DATA=$((DATA + 256 + CTXB)) -DCAPSTONE_DOMREQ_STACK=$STACK -DCAPSTONE_CONTEXT_ARENA_BYTES=$CTXB
         -DCAPSTONE_LEVEL0_ARENA_BYTES=$ARENA -DCAPSTONE_APPLICATION_HEAP_BYTES=0
         -DCAPSTONE_APPLICATION_EXCHANGE_BYTES=$EXCH -DCAPSTONE_APPLICATION_CONTEXTS=$NCTX)
M=$CAP/ports/musl-capstone/runtime
CORE_INC=(-I"$MUSL/src/include" -I"$MUSL/src/internal" -I"$MUSL/obj/src/internal" -I"$MUSL/src/multibyte")

cc() {  # $1 = source, $2 = object, rest = extra flags
  local s=$1 o=$2; shift 2
  "$CC" "${BASE[@]}" "$@" -c "$s" -o "$OUT/obj/$o"
}
# the core, as Application.cmake lists it, minus start-musl.S (replaced by the gp-captable glue below)
for f in hostcall tls atomic_libcalls context lock delegate signals; do cc "$M/$f.c" "core_$f.o" "${CORE_INC[@]}"; done
cc "$M/posix_spawn_delegate.c" core_posix_spawn_delegate.o "${CORE_INC[@]}" -I"$MUSL/src/process"
# The memcpy plain-data guard (string_bounds_safe.c, ISSUES R-29) is on for every silicon build: the type query it
# asks is total on the deployed bitstream. B0_MEMCPY_GUARD=0 turns it off for an A/B.
while read -r o; do [ -n "$o" ] && cc "$M/$o.c" "ovr_$o.o" "${CORE_INC[@]}" \
  -DCAPSTONE_MEMCPY_PLAIN_GUARD="${B0_MEMCPY_GUARD:-1}"; done < "$M/libc_overrides.list"
for f in launch delegate spawn msghdr; do cc "$CAP/runtime/common/$f.c" "common_$f.o" "${CORE_INC[@]}"; done
# printf: musl's vfprintf.o is dropped from the gp-captable archive (its long double needs fp128 constant pools,
# C-43), so the narrowed one is generated from musl's own source and linked ahead of the archive.
python3 "$HERE/gen-vfprintf-double.py" "$MUSL" "$OUT/gen/vfprintf-double.c" > "$OUT/gen-vfprintf.log" || {
  cat "$OUT/gen-vfprintf.log" >&2; exit 2; }
cc "$OUT/gen/vfprintf-double.c" ovr_vfprintf_double.o "${CORE_INC[@]}"
# strtod/atof/scanf (B1.0b): floatscan.o is dropped the same way and strtod.o/vfscanf.o call it, so the three are
# generated narrowed to double (gen-floatscan-double.py) and linked ahead of the archive.
python3 "$HERE/gen-floatscan-double.py" "$MUSL" "$OUT/gen" > "$OUT/gen-floatscan.log" || {
  cat "$OUT/gen-floatscan.log" >&2; exit 2; }
for f in floatscan strtod vfscanf; do cc "$OUT/gen/$f-double.c" "ovr_${f}_double.o" "${CORE_INC[@]}"; done
cc "$CAP/runtime/domain/application.c" app_application.o "${APPDEFS[@]}"
cc "$M/level0.c" app_level0.o "${APPDEFS[@]}"
# B2 (memcached on silicon): an application of many sources. B0_APP_SRCS lists them (absolute paths) and
# B0_APP_CFLAGS their own flags (include directories, -DHAVE_CONFIG_H); each is compiled like the single-file
# applications above, into obj/, and joins the same full-LTO link. Without B0_APP_SRCS, $HERE/$APP.c as before.
if [ -n "${B0_APP_SRCS:-}" ]; then
  n=0
  for f in $B0_APP_SRCS; do
    n=$((n + 1)); cc "$f" "app_$(basename "${f%.c}")_$n.o" "${APPDEFS[@]}" ${B0_APP_CFLAGS:-} || { echo "app source $f FAILED" >&2; exit 1; }
  done
  echo "compiled $n application sources"
else
  cc "$HERE/$APP.c" "app_$APP.o" "${APPDEFS[@]}"
fi
cc "$HERE/glue-data.c" glue_data.o
# builtins, as bitcode too -- into their own directory, VERIFIED through codegen, then a LAZY archive. lld keeps
# every bitcode definition of a libcall, and the fp128 (tf) soft-float family cannot be selected under gp-captable
# (C-43, constant pools with no cap-table slot), so those objects are dropped by name before the link.
mkdir -p "$OUT/rt"
while read -r b; do [ -n "$b" ] && "$CC" "${BASE[@]}" -c "$CAP/../compiler-rt/lib/builtins/$b.c" \
  -o "$OUT/rt/rt_$(basename "$b").o"; done < "$CAP/runtime/cmake/softfloat.list"
python3 "$HERE/lto-codegen-verify.py" "$BIN/llc" "${B0_JOBS:-8}" "${SIL[*]}" "$OUT"/rt/*.o > "$OUT/rt-dropped.txt"
while IFS=$'\t' read -r o why; do [ -n "$o" ] && { echo "builtins: dropping $(basename "$o"): ${why:0:90}"; rm -f "$o"; }; done \
  < "$OUT/rt-dropped.txt"
"$BIN/llvm-ar" rcs "$OUT/librt-b0.a" "$OUT"/rt/*.o
# assembly: native objects (no C globals)
ASM=(-target capstone64-unknown-elf -ffreestanding -DCAPSTONE_GP_CAPTABLE_ABI=1)
GLUE_CTX=(); [ "$CTXB" -gt 0 ] && GLUE_CTX=(-DCAPSTONE_GLUE_CONTEXTS=1 -DCAPSTONE_CONTEXT_ARENA_BYTES=$CTXB -I"$CAP/runtime/include")
# B0_GLUE_EXTRA: extra -D for the glue only, e.g. -DCAPSTONE_GLUE_CONTEXTS_PEEK (QEMU-only prints; never on a board).
[ -n "${B0_GLUE_EXTRA:-}" ] && GLUE_CTX+=($B0_GLUE_EXTRA)
# CAPSTONE_GLUE_CARVE_ALIGN (ISSUES R-11): carve points stay multiples of the region's cursorless granule, which
# matters once the data region spans more than one 2 MiB window (memcached's does).
"$CC" "${ASM[@]}" -DCAPSTONE_GLUE_YIELD=1 -DCAPSTONE_GLUE_NO_MCSR=1 -DCAPSTONE_GLUE_CARVE_ALIGN=1 "${GLUE_CTX[@]}" -c "$CAP/tests/runtime-qemu/silicon-ladder/start-gp-captable-interp.S" \
  -o "$OUT/glue.o"   # outside obj/, so the obj/*.o glob below does not list it twice
for f in set_thread_area setjmp altstack; do "$CC" "${ASM[@]}" -c "$M/$f.S" -o "$OUT/obj/asm_$f.o"; done
"$CC" "${ASM[@]}" "${APPDEFS[@]}" -c "$CAP/runtime/domain/domreq.S" -o "$OUT/obj/asm_domreq.o"
"$CC" "${ASM[@]}" -c "$CAP/runtime/domain/gct-section-end.S" -o "$OUT/obj/asm_gct.o"
"$CC" "${ASM[@]}" -c "$HERE/glue-accessors.S" -o "$OUT/obj/asm_accessors.o"
echo "compiled $(ls "$OUT/obj" | wc -l) objects"

link() {  # $1 = globals offset literal, $2 = output
  sed "s/0x10000 + 0x1000/0x10000 + $1/" "$HERE/link-gpfree-app.ld" > "$OUT/link.ld"
  grep -q "0x10000 + $1" "$OUT/link.ld" || { echo "globals offset substitution failed" >&2; exit 2; }
  "$BIN/ld.lld" -T "$OUT/link.ld" --gc-sections "${PLUG[@]}" -o "$2" \
    "$OUT"/glue.o "$OUT"/obj/*.o "$ARCHIVE" "$OUT/librt-b0.a" 2> "$OUT/link-$(basename "$2").err" || {
      cat "$OUT/link-$(basename "$2").err" >&2; return 1; }
}
link 0x800000 "$OUT/pass1.dom"
TEXT=$("$BIN/llvm-readelf" -SW "$OUT/pass1.dom" | python3 -c '
import sys, re
for l in sys.stdin:
    m = re.match(r"\s*\[\s*\d+\]\s+(\.text)\s+\S+\s+[0-9a-f]+\s+[0-9a-f]+\s+([0-9a-f]+)", l)
    if m: print(int(m.group(2), 16)); break
else: print(0)')
[ "${TEXT:-0}" -gt 0 ] || { echo "could not measure .text from pass 1" >&2; exit 2; }
GOFF=$(( ((TEXT + 0xFFFF) / 0x10000) * 0x10000 )); [ $GOFF -lt 65536 ] && GOFF=65536
printf '.text = %d bytes -> globals offset 0x%x\n' "$TEXT" "$GOFF"
link "$(printf '0x%x' $GOFF)" "$OUT/$APP.dom"
ls -la "$OUT/$APP.dom"
