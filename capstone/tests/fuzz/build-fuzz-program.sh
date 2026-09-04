#!/usr/bin/env bash
# Build one generated C program into a Capstone domain.
#
#   build-fuzz-program.sh <prog.c> <out.dom> <-O0|-O1|-O2|-Os> [xor]
#
# Same start.S/link.ld/gct-tail as tests/runtime-qemu/build-domain.sh, plus the
# csmith runtime overlay (csmith-rt/), fuzz_domain.c (the entry, returning the
# checksum through the 32-bit result channel; "xor" builds the FUZZ_XOR positive
# control) and beebs_freestanding_string.c for the aggregate copies -O2 emits.
# Every domain compile carries -mllvm -verify-machineinstrs.  Exit codes name the
# stage: 10 compile error, 11 compiler crash, 12 link error, 13 image over the
# single-region ceiling (1376256 bytes).  Prints the image size and sha256.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"
RT="$SCRIPT_DIR/csmith-rt"
DOMAIN="$CAPSTONE_REPO_ROOT/capstone/my_first_domain"
RUNTIME="$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu"
STRING_SRC="$CAPSTONE_REPO_ROOT/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c"
CEILING=1376256

SRC=${1:?usage: build-fuzz-program.sh <prog.c> <out.dom> <level> [xor]}
OUT=${2:?usage: build-fuzz-program.sh <prog.c> <out.dom> <level> [xor]}
LEVEL=${3:?usage: build-fuzz-program.sh <prog.c> <out.dom> <level> [xor]}
XOR=${4:-}
CSMITH_INC=$(sed -n 's/^csmith_include=//p' "$SCRIPT_DIR/TOOLS.lock" 2>/dev/null); CSMITH_INC=${CSMITH_INC/#\$HOME/$HOME}
[[ -f "$CSMITH_INC/csmith.h" ]] || { echo "ERROR: csmith runtime headers not found; run setup-fuzz-tools.sh" >&2; exit 2; }

OBJ=$(mktemp -d "${CAPSTONE_TMP_ROOT:-/tmp}/fuzz-build.XXXXXX")
trap 'rm -rf "$OBJ"' EXIT
# +m as in every corpus build: without it a division at -O0 becomes a __udivdi3
# libcall that no freestanding domain can satisfy.
CC=("$CAPSTONE_CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m -ffreestanding -w -mllvm -verify-machineinstrs)
DEFS=(-D_GNU_SOURCE)
[[ "$XOR" == "xor" ]] && DEFS+=(-DFUZZ_XOR=1)

compile() {  # out.o, args...
  local o=$1; shift
  local err rc
  err=$("${CC[@]}" "$@" -c -o "$o" 2>&1); rc=$?
  if [[ $rc -ne 0 ]]; then
    if grep -q 'PLEASE submit a bug report\|Stack dump\|Assertion\|UNREACHABLE' <<< "$err"; then
      echo "BUILD-CRASH: $(grep -m1 'Assertion\|LLVM ERROR\|Cannot select\|UNREACHABLE\|error:' <<< "$err" | cut -c1-200)"; return 11
    fi
    echo "BUILD-ERROR: $(grep -m1 'error' <<< "$err" | cut -c1-200)"; return 10
  fi
}

compile "$OBJ/start.o" "$DOMAIN/start.S" || exit $?
compile "$OBJ/gct.o" "$RUNTIME/gct-section-end.S" || exit $?
compile "$OBJ/prog.o" "$LEVEL" -I"$RT" -I"$CSMITH_INC" -include "$RT/capstone_platform.h" "${DEFS[@]}" "$SRC" || exit $?
compile "$OBJ/entry.o" "$LEVEL" "${DEFS[@]}" "$RT/fuzz_domain.c" || exit $?
compile "$OBJ/string.o" "$LEVEL" "$STRING_SRC" || exit $?

mkdir -p "$(dirname -- "$OUT")"
err=$("$CAPSTONE_LD_LLD" -T "$DOMAIN/link.ld" -o "$OUT" "$OBJ/start.o" "$OBJ/prog.o" "$OBJ/entry.o" "$OBJ/string.o" "$OBJ/gct.o" 2>&1) \
  || { echo "LINK-ERROR: $(head -c 300 <<< "$err")"; exit 12; }
SIZE=$(stat -c%s "$OUT")
if (( SIZE > CEILING )); then echo "SIZE-SKIP: $SIZE bytes > $CEILING"; rm -f "$OUT"; exit 13; fi
echo "BUILT $OUT size=$SIZE sha256=$(sha256sum "$OUT" | cut -c1-16)"
