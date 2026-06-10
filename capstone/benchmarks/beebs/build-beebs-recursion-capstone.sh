#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

"$SCRIPT_DIR/fetch-beebs.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
BEEBS_SRC_DIR=${BEEBS_SRC_DIR:-$CAPSTONE_TMP_ROOT/beebs-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/beebs-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj}
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_recursion_capstone.dom}

RECURSION_SRC=$BEEBS_SRC_DIR/src/recursion/librecursion.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
PATCHED_RECURSION_SRC=$OUT_DIR/librecursion_capstone.c

if [[ ! -f "$RECURSION_SRC" || ! -f "$SUPPORT_DIR/support.h" ]]; then
  echo "missing BEEBS recursion source tree: $BEEBS_SRC_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$OBJ_DIR"

cat > "$PATCHED_RECURSION_SRC" <<'EOF'
volatile int In;
static int n;

#define CAPSTONE_DELIN(rd) \
  __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static __attribute__((noinline)) volatile int *beebs_recursion_in_ptr(void) {
  volatile int *p = &In;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int *beebs_recursion_n_ptr(void) {
  int *p = &n;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int beebs_recursion_in_get(void) {
  volatile int *p = beebs_recursion_in_ptr();
  return *p;
}

static __attribute__((noinline)) void beebs_recursion_in_set(int value) {
  volatile int *p = beebs_recursion_in_ptr();
  *p = value;
}

static __attribute__((noinline)) int beebs_recursion_n_get(void) {
  int *p = beebs_recursion_n_ptr();
  return *p;
}

static __attribute__((noinline)) void beebs_recursion_n_set(int value) {
  int *p = beebs_recursion_n_ptr();
  *p = value;
}

int fib(int i) {
  if (i == 0)
    return 1;
  if (i == 1)
    return 1;
  return fib(i - 1) + fib(i - 2);
}

int anka(int i);

int kalle(int i) {
  if (i <= 0)
    return 0;
  return anka(i - 1);
}

int anka(int i) {
  if (i <= 0)
    return 1;
  return kalle(i - 1);
}

int benchmark(void) {
  beebs_recursion_in_set(fib(beebs_recursion_n_get()));
  return beebs_recursion_in_get();
}

void initialise_benchmark(void) {
  beebs_recursion_n_set(10);
}

int verify_benchmark(int r) {
  return r == 89;
}
EOF

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature
  -Xclang +m
  -ffreestanding
  -fno-builtin
  "$DOMAIN_OPT_LEVEL"
  -I"$SUPPORT_DIR"
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 \
  -c "$START_SRC" \
  -o "$OBJ_DIR/start.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$PATCHED_RECURSION_SRC" \
  -o "$OBJ_DIR/beebs_recursion.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_recursion_domain.c" \
  -o "$OBJ_DIR/beebs_recursion_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/beebs_recursion.o" \
  "$OBJ_DIR/beebs_recursion_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
