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
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_cnt_capstone.dom}

CNT_SRC=$BEEBS_SRC_DIR/src/cnt/cnt.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
PATCHED_CNT_SRC=$OUT_DIR/cnt_capstone.c

if [[ ! -f "$CNT_SRC" || ! -f "$SUPPORT_DIR/support.h" ]]; then
  echo "missing BEEBS cnt source tree: $BEEBS_SRC_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$OBJ_DIR"

cat > "$PATCHED_CNT_SRC" <<'EOF'
#define MAXSIZE 10

typedef int matrix[MAXSIZE][MAXSIZE];

int Seed;
matrix Array;
int Postotal, Negtotal, Poscnt, Negcnt;

#define CAPSTONE_DELIN(rd) \
  __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static __attribute__((noinline)) int (*beebs_cnt_array_ptr(void))[MAXSIZE] {
  int (*p)[MAXSIZE] = Array;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int *beebs_cnt_seed_ptr(void) {
  int *p = &Seed;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int *beebs_cnt_postotal_ptr(void) {
  int *p = &Postotal;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int *beebs_cnt_negtotal_ptr(void) {
  int *p = &Negtotal;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int *beebs_cnt_poscnt_ptr(void) {
  int *p = &Poscnt;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int *beebs_cnt_negcnt_ptr(void) {
  int *p = &Negcnt;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int beebs_cnt_array_get(int row, int col) {
  int (*p)[MAXSIZE] = beebs_cnt_array_ptr();
  return p[row][col];
}

static __attribute__((noinline)) void beebs_cnt_array_set(int row, int col,
                                                          int value) {
  int (*p)[MAXSIZE] = beebs_cnt_array_ptr();
  p[row][col] = value;
}

static __attribute__((noinline)) int beebs_cnt_seed_get(void) {
  int *p = beebs_cnt_seed_ptr();
  return *p;
}

static __attribute__((noinline)) void beebs_cnt_seed_set(int value) {
  int *p = beebs_cnt_seed_ptr();
  *p = value;
}

static __attribute__((noinline)) void beebs_cnt_set_totals(int postotal,
                                                           int poscnt,
                                                           int negtotal,
                                                           int negcnt) {
  int *postotal_p = beebs_cnt_postotal_ptr();
  int *poscnt_p = beebs_cnt_poscnt_ptr();
  int *negtotal_p = beebs_cnt_negtotal_ptr();
  int *negcnt_p = beebs_cnt_negcnt_ptr();
  *postotal_p = postotal;
  *poscnt_p = poscnt;
  *negtotal_p = negtotal;
  *negcnt_p = negcnt;
}

int RandomInteger(void) {
  int next = ((beebs_cnt_seed_get() * 133) + 81) % 8095;
  beebs_cnt_seed_set(next);
  return next;
}

int InitSeed(void) {
  beebs_cnt_seed_set(0);
  return 0;
}

int Initialize(matrix unused) {
  (void)unused;
  for (int outer = 0; outer < MAXSIZE; outer++)
    for (int inner = 0; inner < MAXSIZE; inner++)
      beebs_cnt_array_set(outer, inner, RandomInteger());

  return 0;
}

int Sum(matrix unused) {
  (void)unused;
  int ptotal = 0;
  int ntotal = 0;
  int pcnt = 0;
  int ncnt = 0;

  for (int outer = 0; outer < MAXSIZE; outer++) {
    for (int inner = 0; inner < MAXSIZE; inner++) {
      int value = beebs_cnt_array_get(outer, inner);
      if (value < 0) {
        ptotal += value;
        pcnt++;
      } else {
        ntotal += value;
        ncnt++;
      }
    }
  }

  beebs_cnt_set_totals(ptotal, pcnt, ntotal, ncnt);
  return ntotal;
}

int Test(matrix array) {
  return Sum(array);
}

int benchmark(void) {
  return Test(Array);
}

void initialise_benchmark(void) {
  InitSeed();
  Initialize(Array);
}

int verify_benchmark(int nt) {
  return nt == 396675;
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
  -c "$PATCHED_CNT_SRC" \
  -o "$OBJ_DIR/beebs_cnt.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_cnt_domain.c" \
  -o "$OBJ_DIR/beebs_cnt_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/beebs_cnt.o" \
  "$OBJ_DIR/beebs_cnt_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
