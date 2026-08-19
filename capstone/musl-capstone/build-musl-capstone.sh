#!/usr/bin/env bash
# Build libc-capstone.a from whatever of musl the compiler currently accepts.
#
# Deliberately builds the PARTIAL set rather than waiting for 100 %: the useful
# question is not "does all of musl compile" but "what does a given program
# actually pull in", and only a linkable archive can answer that. Undefined
# symbols at link time are the work list; see README.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../tests/capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/musl-capstone-build}
OBJ_DIR="$OUT_DIR/obj"
ARCHIVE=${ARCHIVE:-$OUT_DIR/libc-capstone.a}
AR=${CAPSTONE_LLVM_AR:-$CAPSTONE_LLVM_BIN/llvm-ar}

MUSL_SRC_DIR=$(bash "$SCRIPT_DIR/prepare-musl-capstone.sh" | tail -1)

rm -rf "$OBJ_DIR"
mkdir -p "$OBJ_DIR"

# The survey owns the flags and the file set; --objects makes it keep the
# output. It exits 1 on a regression, which must not stop the build here: a
# partial archive is the point. Only a harness error (2) is fatal.
set +e
python3 "$SCRIPT_DIR/survey-musl-capstone.py" "$MUSL_SRC_DIR" \
        --objects "$OBJ_DIR" > "$OUT_DIR/survey.txt"
survey_status=$?
set -e
if [[ $survey_status -ge 2 ]]; then
  cat "$OUT_DIR/survey.txt" >&2
  echo "survey could not measure; build aborted" >&2
  exit 2
fi

# ---------------------------------------------------------------- libc-ext
#
# Three musl malloc members compile but CANNOT WORK: their bodies live in files
# that do not compile for this target (sizeof(void*) static asserts), so what is
# left is free() calling an absent __libc_free, realloc() an absent
# __libc_realloc, and lite_malloc an absent __mmap. Left in the archive next to
# libc-ext/malloc.c they would also make `malloc` ambiguous -- which definition
# the linker picks would depend on archive order. Dropped, so the working one is
# the only one.
for orphan in src_malloc_free.o src_malloc_realloc.o src_malloc_lite_malloc.o; do
  [[ -e "$OBJ_DIR/$orphan" ]] || {
    echo "expected musl member $orphan not produced; the malloc situation has" >&2
    echo "changed and libc-ext/malloc.c may now be shadowing something real" >&2
    exit 2
  }
  rm -f "$OBJ_DIR/$orphan"
done

# musl's __errno_location is REPLACED, not orphaned: it compiles and would work
# on a machine where the thread pointer survives a round trip through uintptr_t.
# On this one it does not (see libc-ext/errno.c), so ours must be the only
# definition -- otherwise which one a program gets depends on archive order, and
# a program that got musl's would fault on its first failing syscall.
#
# THE OTHERS ON THIS LIST ARRIVED FOR A DIFFERENT REASON, and the difference is
# the point. Until C-21/C-25/C-26 landed on the shared compiler branch, musl's
# own strlen, memcpy, calloc, vfprintf and __lock did not compile at all, so
# libc-ext was the only definition and nothing collided. They compile now. They
# are still wrong here, and each is wrong in its own way:
#
#   strlen    reads a whole size_t at a time, so it reads up to 7 bytes past the
#             last one. Ordinarily harmless; against a capability whose bounds
#             end at the string, it is a bounds fault. libc-ext is byte at a time.
#   memcpy    moves bytes, and a capability's tag rides beside the memory rather
#             than in it -- 16 byte moves deliver the right bytes with the tag
#             CLEARED, and the pointer faults later somewhere innocent. See
#             libc-ext/cap-copy.h; this one is not a preference.
#   calloc    is the other half of libc-ext/malloc.c's arena. musl's calls a
#             malloc that is not the one a domain has.
#   vfprintf  is generated with the double path compiled at -O0, which is what
#             keeps fmt_fp off the capability-arithmetic wall (see below).
#   __lock    would take a real lock in a domain that has one thread, reading
#             libc.need_locks to decide -- which is the >64-bit constant wall.
#
# So: NOT "these do not build", but "these build and must not be used". If one
# of them stops being produced, that is a signal worth stopping for -- either
# musl moved the file or a compiler change took it back out -- hence the check.
for shadowed in src_errno___errno_location.o src_string_strlen.o \
                src_string_memcpy.o src_malloc_calloc.o \
                src_stdio_vfprintf.o src_thread___lock.o; do
  [[ -e "$OBJ_DIR/$shadowed" ]] || {
    echo "expected musl member $shadowed not produced; the libc-ext file that" >&2
    echo "shadows it may now be shadowing nothing, or musl has moved it" >&2
    exit 2
  }
  rm -f "$OBJ_DIR/$shadowed"
done

# Our own sources, built with the SURVEY's flags. --print-flags rather than a
# second copy of the list: the archive's members and these must agree on the
# target, the ABI and -fno-jump-tables, and two lists drift.
mapfile -t EXT_FLAGS < <(python3 "$SCRIPT_DIR/survey-musl-capstone.py" \
                                 "$MUSL_SRC_DIR" --print-flags)
[[ ${#EXT_FLAGS[@]} -gt 0 ]] || { echo "no compile flags from survey" >&2; exit 2; }

GEN_DIR="$OUT_DIR/gen"
python3 "$SCRIPT_DIR/libc-ext/gen-vfprintf-double.py" \
        "$MUSL_SRC_DIR" "$GEN_DIR/vfprintf_double.c"

# vfprintf AT -O0, and this is not a preference.
#
# At -O1 the optimiser splits fmt_fp's digit pointer (`*--s = '0' + x % 10`)
# into a base and an index. The index is i128, the same width as a capability,
# and gets selected as capability arithmetic: `li a1, 12` followed by
# `cincoffsetimm a1, a1, -1`, an integer in the base position. QEMU rejects it
# with `helper_cscincoffsetimm: Assertion 'rs1_v->tag' failed` and the domain
# dies on the first %f. Measured on the printf probe, which reaches "%d %u %x
# %o %c %s" and then faults on "%08.3f".
#
# -O0 has ZERO such sites, -O1 six, -O2 three (scan-cap-base.py). The gate below
# is what keeps this from silently regressing if someone raises the level; the
# real fix is in the backend's operand-role classifier
# (llvm/test/CodeGen/Capstone/cap-cincoffset-base.ll, isCapstoneCapabilityValue),
# which already handles the cases it can see and misses this one.
EXT_FLAGS_O0=()
for f in "${EXT_FLAGS[@]}"; do [[ $f == -O1 ]] && f=-O0; EXT_FLAGS_O0+=("$f"); done

"${CAPSTONE_CLANG:?}" "${EXT_FLAGS_O0[@]}" -S "$GEN_DIR/vfprintf_double.c" \
                      -o "$GEN_DIR/vfprintf_double.s"
"${CAPSTONE_CLANG:?}" "${EXT_FLAGS_O0[@]}" -c "$GEN_DIR/vfprintf_double.c" \
                      -o "$OBJ_DIR/ext_vfprintf.o"

# The heap size is fixed when the LIBC is built, not when a program is, because
# the allocator carves from a static array. 256 KiB is the default and suits a
# probe; an interpreter wants more (mruby's state plus mrblib's ireps do not fit
# in it). Raising it costs every domain that references malloc, so it is an
# explicit knob rather than a generous default:
#
#   CAPSTONE_LIBC_HEAP_BYTES=$((4*1024*1024)) bash build-musl-capstone.sh
#
# The per-program override this really wants is a weak heap symbol a program can
# replace, which needs a size symbol beside it; not built until something needs
# two different sizes in one tree.
HEAP_FLAGS=()
[[ -n "${CAPSTONE_LIBC_HEAP_BYTES:-}" ]] && \
  HEAP_FLAGS=(-DCAPSTONE_LIBC_HEAP_BYTES="$CAPSTONE_LIBC_HEAP_BYTES")

# cap-copy.c AT -O0, for a different reason than vfprintf and a worse one.
#
# It has to decide whether two addresses are congruent mod 16, because a
# capability tag only exists at a 16-byte-aligned address. The frontend compiles
# `(uintptr_t)p & 15` correctly at 64 bits; at -O1 InstCombine re-widens it to
# the POINTER's width and leaves `i128 = and %x, 15`, which instruction selection
# cannot match, so the compiler dies outright. Four spellings were tried and all
# four crash, including clang's own __builtin_is_aligned -- it is the fold, not
# the source. ISSUES.md C-28; the fix belongs in the backend, alongside the C-25
# lowering for i128 sub.
#
# Only this file drops. memcpy.c, memmove.c and malloc.c call into it and stay at
# -O1, which is why the copy lives in its own translation unit at all.
for src in "$SCRIPT_DIR"/libc-ext/*.c; do
  name=$(basename "$src" .c)
  if [[ $name == cap-copy ]]; then
    flags=("${EXT_FLAGS_O0[@]}")
  else
    flags=("${EXT_FLAGS[@]}")
  fi
  "${CAPSTONE_CLANG:?}" "${flags[@]}" "${HEAP_FLAGS[@]}" -S "$src" -o "$GEN_DIR/ext_$name.s"
  "${CAPSTONE_CLANG:?}" "${flags[@]}" "${HEAP_FLAGS[@]}" -c "$src" -o "$OBJ_DIR/ext_$name.o"
done

# Self-test FIRST: a scanner that cannot flag its own synthetic case would pass
# everything silently, which is the failure mode this project keeps paying for.
python3 "$SCRIPT_DIR/libc-ext/scan-cap-base.py" --self-test
if ! python3 "$SCRIPT_DIR/libc-ext/scan-cap-base.py" "$GEN_DIR"/*.s; then
  echo "an integer is being used as a capability base; see the comment above" >&2
  exit 1
fi

# Soft-float builtins, in the libc rather than in every program.
#
# capstone64 has no hardware double (+m only), so printf's own arithmetic --
# comparing, scaling and converting the value it is formatting -- is calls to
# compiler-rt. Without them "printf exists" is not a linkable statement, which is
# why they belong beside it: build-lua-probe.sh already carries a 22-name list
# for exactly this, and a second program would have carried a third copy.
#
# NO `|| continue` HERE, unlike that loop. Every name below compiles today
# (measured), so a failure is a regression and must stop the build rather than
# quietly leave a symbol out for the next program to discover at link time. The
# 128-bit family (addtf3, trunctfdf2, floatsitf, ...) is absent on purpose: it
# does not compile at all on this target, which is ISSUES.md C-20.
BUILTINS_DIR="$CAPSTONE_REPO_ROOT/compiler-rt/lib/builtins"
for builtin in adddf3 subdf3 muldf3 divdf3 comparedf2 fixdfsi fixdfdi fixunsdfsi \
               fixunsdfdi floatsidf floatdidf floatunsidf floatundidf addsf3 \
               subsf3 mulsf3 divsf3 comparesf2 floatsisf extendsfdf2 truncdfsf2 \
               fp_mode; do
  "${CAPSTONE_CLANG:?}" -target capstone64-unknown-elf \
    -Xclang -target-feature -Xclang +m \
    -ffreestanding -fno-builtin -fno-optimize-sibling-calls -fno-jump-tables \
    -O0 -w -I"$BUILTINS_DIR" -c "$BUILTINS_DIR/$builtin.c" \
    -o "$OBJ_DIR/rt_$builtin.o" || {
      echo "compiler-rt builtin $builtin no longer compiles for capstone64" >&2
      exit 2
    }
done

objects=("$OBJ_DIR"/*.o)
if [[ ${#objects[@]} -eq 0 || ! -e "${objects[0]}" ]]; then
  echo "no objects produced in $OBJ_DIR" >&2
  exit 2
fi

rm -f "$ARCHIVE"
"$AR" rcs "$ARCHIVE" "${objects[@]}"

# ------------------------------------------------------------------ checks
#
# Both of these have to be able to FAIL, or archiving "succeeds" whatever went
# in. The first fires if a symbol we now own is defined twice or not at all; the
# second fired for real on 2026-08-14, when 15 members turned out to carry
# absolute-addressed switch tables because -fno-jump-tables was missing.
nm_out=$("$CAPSTONE_LLVM_BIN/llvm-nm" --print-armap "$ARCHIVE")
status=0
for sym in malloc free realloc calloc vfprintf memcpy strlen __lock __adddf3 \
           __errno_location __capstone_cap_copy_fwd; do
  n=$(printf '%s\n' "$nm_out" | grep -cE "^${sym} in ")
  if [[ "$n" != 1 ]]; then
    echo "CHECK FAILED: $sym is defined by $n archive members, expected 1" >&2
    status=1
  fi
done
jt=$("$CAPSTONE_LLVM_BIN/llvm-nm" "$ARCHIVE" 2>/dev/null \
     | awk '/^[^ ].*:$/{m=$0} /LJTI/{print m}' | sort -u)
if [[ -n "$jt" ]]; then
  echo "CHECK FAILED: jump tables in the archive (ISSUES.md C-4a: they fault):" >&2
  printf '  %s\n' $jt >&2
  status=1
fi
[[ $status -eq 0 ]] || exit 1

grep -E '^(surveyed|compiled|failed)' "$OUT_DIR/survey.txt"
printf 'archived       %d objects -> %s\n' "${#objects[@]}" "$ARCHIVE"
printf 'libc-ext       vfprintf (long double narrowed to double), malloc family\n'
