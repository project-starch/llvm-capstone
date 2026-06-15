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
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_tarai_capstone.dom}

TARAI_SRC=$BEEBS_SRC_DIR/src/tarai/libtarai.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
PATCHED_TARAI_SRC=$OUT_DIR/libtarai_capstone.c

if [[ ! -f "$TARAI_SRC" || ! -f "$SUPPORT_DIR/support.h" ]]; then
  echo "missing BEEBS tarai source tree: $BEEBS_SRC_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$OBJ_DIR"

cat > "$PATCHED_TARAI_SRC" <<'EOF'
int x, y, z;

#define CAPSTONE_DELIN(rd) \
  __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static __attribute__((noinline)) int *beebs_tarai_x_ptr(void) {
  int *p = &x;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int *beebs_tarai_y_ptr(void) {
  int *p = &y;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int *beebs_tarai_z_ptr(void) {
  int *p = &z;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int beebs_tarai_x_get(void) {
  int *p = beebs_tarai_x_ptr();
  return *p;
}

static __attribute__((noinline)) void beebs_tarai_x_set(int value) {
  int *p = beebs_tarai_x_ptr();
  *p = value;
}

static __attribute__((noinline)) int beebs_tarai_y_get(void) {
  int *p = beebs_tarai_y_ptr();
  return *p;
}

static __attribute__((noinline)) void beebs_tarai_y_set(int value) {
  int *p = beebs_tarai_y_ptr();
  *p = value;
}

static __attribute__((noinline)) int beebs_tarai_z_get(void) {
  int *p = beebs_tarai_z_ptr();
  return *p;
}

static __attribute__((noinline)) void beebs_tarai_z_set(int value) {
  int *p = beebs_tarai_z_ptr();
  *p = value;
}

int tarai(int x_arg, int y_arg, int z_arg) {
  int ox = x_arg;
  int oy = y_arg;

  while (x_arg > y_arg) {
    ox = x_arg;
    oy = y_arg;

    x_arg = tarai(x_arg - 1, y_arg, z_arg);
    y_arg = tarai(y_arg - 1, z_arg, ox);

    if (x_arg <= y_arg)
      break;

    z_arg = tarai(z_arg - 1, ox, oy);
  }

  return y_arg;
}

int benchmark(void) {
  volatile int cnt = 0;
  cnt = tarai(beebs_tarai_x_get(), beebs_tarai_y_get(),
              beebs_tarai_z_get());
  return cnt;
}

void initialise_benchmark(void) {
  beebs_tarai_x_set(9);
  beebs_tarai_y_set(6);
  beebs_tarai_z_set(3);
}

int verify_benchmark(int r) {
  return r == 9;
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
  -c "$PATCHED_TARAI_SRC" \
  -o "$OBJ_DIR/beebs_tarai.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" \
  -o "$OBJ_DIR/beebs_tarai_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/beebs_tarai.o" \
  "$OBJ_DIR/beebs_tarai_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
