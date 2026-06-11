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
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_janne_complex_capstone.dom}

JANNE_COMPLEX_SRC=$BEEBS_SRC_DIR/src/janne_complex/libjanne_complex.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
PATCHED_JANNE_COMPLEX_SRC=$OUT_DIR/libjanne_complex_capstone.c

if [[ ! -f "$JANNE_COMPLEX_SRC" || ! -f "$SUPPORT_DIR/support.h" ]]; then
  echo "missing BEEBS janne_complex source tree: $BEEBS_SRC_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$OBJ_DIR"

cat > "$PATCHED_JANNE_COMPLEX_SRC" <<'EOF'
static int a, b;

#define CAPSTONE_DELIN(rd) \
  __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static __attribute__((noinline)) int *beebs_janne_complex_a_ptr(void) {
  int *p = &a;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int *beebs_janne_complex_b_ptr(void) {
  int *p = &b;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int beebs_janne_complex_a_get(void) {
  int *p = beebs_janne_complex_a_ptr();
  return *p;
}

static __attribute__((noinline)) void beebs_janne_complex_a_set(int value) {
  int *p = beebs_janne_complex_a_ptr();
  *p = value;
}

static __attribute__((noinline)) int beebs_janne_complex_b_get(void) {
  int *p = beebs_janne_complex_b_ptr();
  return *p;
}

static __attribute__((noinline)) void beebs_janne_complex_b_set(int value) {
  int *p = beebs_janne_complex_b_ptr();
  *p = value;
}

int complex(int a_arg, int b_arg) {
  while (a_arg < 30) {
    while (b_arg < a_arg) {
      if (b_arg > 5)
        b_arg = b_arg * 3;
      else
        b_arg = b_arg + 2;

      if (b_arg >= 10 && b_arg <= 12)
        a_arg = a_arg + 10;
      else
        a_arg = a_arg + 1;
    }

    a_arg = a_arg + 2;
    b_arg = b_arg - 10;
  }

  return 1;
}

int benchmark(void) {
  return complex(beebs_janne_complex_a_get(), beebs_janne_complex_b_get());
}

void initialise_benchmark(void) {
  beebs_janne_complex_a_set(1);
  beebs_janne_complex_b_set(1);
}

int verify_benchmark(int r) {
  return r == 1;
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
  -c "$PATCHED_JANNE_COMPLEX_SRC" \
  -o "$OBJ_DIR/beebs_janne_complex.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_janne_complex_domain.c" \
  -o "$OBJ_DIR/beebs_janne_complex_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/beebs_janne_complex.o" \
  "$OBJ_DIR/beebs_janne_complex_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
