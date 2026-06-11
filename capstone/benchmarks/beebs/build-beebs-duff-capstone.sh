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
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_duff_capstone.dom}

DUFF_SRC=$BEEBS_SRC_DIR/src/duff/libduff.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
PATCHED_DUFF_SRC=$OUT_DIR/libduff_capstone.c

if [[ ! -f "$DUFF_SRC" || ! -f "$SUPPORT_DIR/support.h" ]]; then
  echo "missing BEEBS duff source tree: $BEEBS_SRC_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$OBJ_DIR"

cat > "$PATCHED_DUFF_SRC" <<'EOF'
#define ARRAYSIZE 100
#define INVOCATION_COUNT 43

char source[ARRAYSIZE];
char target[ARRAYSIZE];

#define CAPSTONE_DELIN(rd) \
  __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static __attribute__((noinline)) char *beebs_duff_source_ptr(void) {
  char *p = source;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) char *beebs_duff_target_ptr(void) {
  char *p = target;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) char beebs_duff_source_get(long idx) {
  char *p = beebs_duff_source_ptr();
  return p[idx];
}

static __attribute__((noinline)) void beebs_duff_source_set(long idx,
                                                            char value) {
  char *p = beebs_duff_source_ptr();
  p[idx] = value;
}

static __attribute__((noinline)) char beebs_duff_target_get(long idx) {
  char *p = beebs_duff_target_ptr();
  return p[idx];
}

static __attribute__((noinline)) void beebs_duff_target_set(long idx,
                                                            char value) {
  char *p = beebs_duff_target_ptr();
  p[idx] = value;
}

void duffcopy(int count) {
  int n = (count + 7) / 8;
  long idx = 0;

  switch (count % 8) {
  case 0:
    do {
      beebs_duff_target_set(idx, beebs_duff_source_get(idx));
      idx++;
  case 7:
      beebs_duff_target_set(idx, beebs_duff_source_get(idx));
      idx++;
  case 6:
      beebs_duff_target_set(idx, beebs_duff_source_get(idx));
      idx++;
  case 5:
      beebs_duff_target_set(idx, beebs_duff_source_get(idx));
      idx++;
  case 4:
      beebs_duff_target_set(idx, beebs_duff_source_get(idx));
      idx++;
  case 3:
      beebs_duff_target_set(idx, beebs_duff_source_get(idx));
      idx++;
  case 2:
      beebs_duff_target_set(idx, beebs_duff_source_get(idx));
      idx++;
  case 1:
      beebs_duff_target_set(idx, beebs_duff_source_get(idx));
      idx++;
    } while (--n > 0);
  }
}

int benchmark(void) {
  duffcopy(INVOCATION_COUNT);
  return 0;
}

void initialise_benchmark(void) {
  for (long i = 0; i < ARRAYSIZE; i++) {
    beebs_duff_source_set(i, (char)(ARRAYSIZE - i));
    beebs_duff_target_set(i, 0);
  }
}

int verify_benchmark(int unused) {
  (void)unused;
  for (long i = 0; i < INVOCATION_COUNT; i++)
    if (beebs_duff_source_get(i) != beebs_duff_target_get(i))
      return 0;
  return 1;
}
EOF

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature
  -Xclang +m
  -ffreestanding
  -fno-builtin
  -fno-jump-tables
  "$DOMAIN_OPT_LEVEL"
  -I"$SUPPORT_DIR"
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 \
  -c "$START_SRC" \
  -o "$OBJ_DIR/start.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$PATCHED_DUFF_SRC" \
  -o "$OBJ_DIR/beebs_duff.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_duff_domain.c" \
  -o "$OBJ_DIR/beebs_duff_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/beebs_duff.o" \
  "$OBJ_DIR/beebs_duff_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
