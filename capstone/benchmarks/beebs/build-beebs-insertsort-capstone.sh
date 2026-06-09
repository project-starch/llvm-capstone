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
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_insertsort_capstone.dom}

INSERTSORT_SRC=$BEEBS_SRC_DIR/src/insertsort/libinsertsort.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
PATCHED_INSERTSORT_SRC=$OUT_DIR/libinsertsort_capstone.c

if [[ ! -f "$INSERTSORT_SRC" || ! -f "$SUPPORT_DIR/support.h" ]]; then
  echo "missing BEEBS insertsort source tree: $BEEBS_SRC_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$OBJ_DIR"

sed '/^int$/,$d' \
  "$INSERTSORT_SRC" > "$PATCHED_INSERTSORT_SRC"
cat >> "$PATCHED_INSERTSORT_SRC" <<'EOF'
#define CAPSTONE_DELIN(rd) \
  __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static __attribute__((noinline)) unsigned int *beebs_insertsort_a_ptr(void) {
  unsigned int *p = a;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int beebs_insertsort_get(int idx) {
  unsigned int *p = beebs_insertsort_a_ptr();
  return (int)p[idx];
}

static __attribute__((noinline)) void beebs_insertsort_set(int idx, int value) {
  unsigned int *p = beebs_insertsort_a_ptr();
  p[idx] = (unsigned int)value;
}

int benchmark(void) {
  int i = 2;
  while (i <= 10) {
    int j = i;
    while (beebs_insertsort_get(j) < beebs_insertsort_get(j - 1)) {
      int temp = beebs_insertsort_get(j);
      beebs_insertsort_set(j, beebs_insertsort_get(j - 1));
      beebs_insertsort_set(j - 1, temp);
      j--;
    }
    i++;
  }
  return 0;
}

void initialise_benchmark(void) {
  beebs_insertsort_set(0, 0);
  beebs_insertsort_set(1, 11);
  beebs_insertsort_set(2, 10);
  beebs_insertsort_set(3, 9);
  beebs_insertsort_set(4, 8);
  beebs_insertsort_set(5, 7);
  beebs_insertsort_set(6, 6);
  beebs_insertsort_set(7, 5);
  beebs_insertsort_set(8, 4);
  beebs_insertsort_set(9, 3);
  beebs_insertsort_set(10, 2);
}

int verify_benchmark(int unused) {
  (void)unused;
  if (beebs_insertsort_get(0) != 0) return 0;
  if (beebs_insertsort_get(1) != 2) return 0;
  if (beebs_insertsort_get(2) != 3) return 0;
  if (beebs_insertsort_get(3) != 4) return 0;
  if (beebs_insertsort_get(4) != 5) return 0;
  if (beebs_insertsort_get(5) != 6) return 0;
  if (beebs_insertsort_get(6) != 7) return 0;
  if (beebs_insertsort_get(7) != 8) return 0;
  if (beebs_insertsort_get(8) != 9) return 0;
  if (beebs_insertsort_get(9) != 10) return 0;
  if (beebs_insertsort_get(10) != 11) return 0;
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
  -c "$PATCHED_INSERTSORT_SRC" \
  -o "$OBJ_DIR/beebs_insertsort.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_insertsort_domain.c" \
  -o "$OBJ_DIR/beebs_insertsort_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/beebs_insertsort.o" \
  "$OBJ_DIR/beebs_insertsort_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
