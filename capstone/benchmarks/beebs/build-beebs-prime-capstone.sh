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
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_prime_capstone.dom}

PRIME_SRC=$BEEBS_SRC_DIR/src/prime/libprime.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
PATCHED_PRIME_SRC=$OUT_DIR/libprime_capstone.c

if [[ ! -f "$PRIME_SRC" || ! -f "$SUPPORT_DIR/support.h" ]]; then
  echo "missing BEEBS prime source tree: $BEEBS_SRC_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$OBJ_DIR"

cat > "$PATCHED_PRIME_SRC" <<'EOF'
typedef unsigned char bool;
typedef unsigned long ulong;

volatile int result = 0;
ulong x;
ulong y;

#define CAPSTONE_DELIN(rd) \
  __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static __attribute__((noinline)) volatile int *beebs_prime_result_ptr(void) {
  volatile int *p = &result;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) ulong *beebs_prime_x_ptr(void) {
  ulong *p = &x;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) ulong *beebs_prime_y_ptr(void) {
  ulong *p = &y;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int beebs_prime_result_get(void) {
  volatile int *p = beebs_prime_result_ptr();
  return *p;
}

static __attribute__((noinline)) void beebs_prime_result_set(int value) {
  volatile int *p = beebs_prime_result_ptr();
  *p = value;
}

static __attribute__((noinline)) ulong beebs_prime_x_get(void) {
  ulong *p = beebs_prime_x_ptr();
  return *p;
}

static __attribute__((noinline)) void beebs_prime_x_set(ulong value) {
  ulong *p = beebs_prime_x_ptr();
  *p = value;
}

static __attribute__((noinline)) ulong beebs_prime_y_get(void) {
  ulong *p = beebs_prime_y_ptr();
  return *p;
}

static __attribute__((noinline)) void beebs_prime_y_set(ulong value) {
  ulong *p = beebs_prime_y_ptr();
  *p = value;
}

bool divides(ulong n, ulong m) {
  return (m % n == 0);
}

bool even(ulong n) {
  return divides(2, n);
}

bool prime(ulong n) {
  ulong i;
  if (even(n))
    return (n == 2);
  for (i = 3; i * i <= n; i += 2) {
    if (divides(i, n))
      return 0;
  }
  return (n > 1);
}

static void beebs_prime_swap_globals(void) {
  ulong tmp = beebs_prime_x_get();
  beebs_prime_x_set(beebs_prime_y_get());
  beebs_prime_y_set(tmp);
}

void swap(ulong *a, ulong *b) {
  ulong tmp = *a;
  *a = *b;
  *b = tmp;
}

int benchmark(void) {
  beebs_prime_swap_globals();
  beebs_prime_result_set(!(prime(beebs_prime_x_get()) &&
                           prime(beebs_prime_y_get())));
  return 0;
}

void initialise_benchmark(void) {
  beebs_prime_x_set(21649L);
  beebs_prime_y_set(513239L);
}

int verify_benchmark(int unused) {
  (void)unused;
  return beebs_prime_result_get() == 0;
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
  -c "$PATCHED_PRIME_SRC" \
  -o "$OBJ_DIR/beebs_prime.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" \
  -o "$OBJ_DIR/beebs_prime_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/beebs_prime.o" \
  "$OBJ_DIR/beebs_prime_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
