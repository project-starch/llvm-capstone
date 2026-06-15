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
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_bubblesort_capstone.dom}

BUBBLESORT_SRC=$BEEBS_SRC_DIR/src/bubblesort/libbubblesort.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
PATCHED_BUBBLESORT_SRC=$OUT_DIR/libbubblesort_capstone.c

if [[ ! -f "$BUBBLESORT_SRC" || ! -f "$SUPPORT_DIR/support.h" ]]; then
  echo "missing BEEBS bubblesort source tree: $BEEBS_SRC_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$OBJ_DIR"

cat > "$PATCHED_BUBBLESORT_SRC" <<'EOF'
#define FALSE 0
#define TRUE 1
#define NUMELEMS 100
#define MAXDIM (NUMELEMS + 1)

int Array[MAXDIM], Seed;
int factor;

#define CAPSTONE_DELIN(rd) \
  __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static __attribute__((noinline)) int *beebs_bubblesort_array_ptr(void) {
  int *p = Array;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int *beebs_bubblesort_factor_ptr(void) {
  int *p = &factor;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int beebs_bubblesort_get(int idx) {
  int *p = beebs_bubblesort_array_ptr();
  return p[idx];
}

static __attribute__((noinline)) void beebs_bubblesort_set(int idx, int value) {
  int *p = beebs_bubblesort_array_ptr();
  p[idx] = value;
}

static __attribute__((noinline)) void beebs_bubblesort_factor_set(int value) {
  int *p = beebs_bubblesort_factor_ptr();
  *p = value;
}

static __attribute__((noinline)) int beebs_bubblesort_factor_get(void) {
  int *p = beebs_bubblesort_factor_ptr();
  return *p;
}

void BubbleSort(int unused[]) {
  (void)unused;
  int sorted = FALSE;
  int temp, index, i;

  for (i = 0; i < NUMELEMS; i++) {
    sorted = TRUE;
    for (index = 0; index < NUMELEMS; index++) {
      if (index >= NUMELEMS - i)
        break;
      if (beebs_bubblesort_get(index) > beebs_bubblesort_get(index + 1)) {
        temp = beebs_bubblesort_get(index);
        beebs_bubblesort_set(index, beebs_bubblesort_get(index + 1));
        beebs_bubblesort_set(index + 1, temp);
        sorted = FALSE;
      }
    }

    if (sorted)
      break;
  }
}

int benchmark(void) {
  BubbleSort(Array);
  return 0;
}

void initialise_benchmark(void) {
  int index;

  beebs_bubblesort_factor_set(-1);
  int fact = beebs_bubblesort_factor_get();
  for (index = 0; index < NUMELEMS; index++)
    beebs_bubblesort_set(index, index * fact);
}

int verify_benchmark(int result) {
  (void)result;
  for (int i = 0; i < NUMELEMS; i++) {
    int expected = i - (NUMELEMS - 1);
    if (beebs_bubblesort_get(i) != expected)
      return 0;
  }
  return 1;
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
  -c "$PATCHED_BUBBLESORT_SRC" \
  -o "$OBJ_DIR/beebs_bubblesort.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" \
  -o "$OBJ_DIR/beebs_bubblesort_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/beebs_bubblesort.o" \
  "$OBJ_DIR/beebs_bubblesort_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
