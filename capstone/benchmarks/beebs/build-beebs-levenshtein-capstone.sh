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
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_levenshtein_capstone.dom}

LEVENSHTEIN_SRC=$BEEBS_SRC_DIR/src/levenshtein/liblevenshtein.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
PATCHED_LEVENSHTEIN_SRC=$OUT_DIR/liblevenshtein_capstone.c

if [[ ! -f "$LEVENSHTEIN_SRC" || ! -f "$SUPPORT_DIR/support.h" ]]; then
  echo "missing BEEBS levenshtein source tree: $BEEBS_SRC_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$OBJ_DIR"

cat > "$PATCHED_LEVENSHTEIN_SRC" <<'EOF'
#define NUM_STRINGS 5
#define MAX_STRING_LEN 10

static const char string0[] = "srrjngre";
static const char string1[] = "asfcjnsdkj";
static const char string2[] = "string";
static const char string3[] = "msd";
static const char string4[] = "strings";

#define CAPSTONE_DELIN(rd) \
  __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static int min(int x, int y) {
  return x < y ? x : y;
}

static __attribute__((noinline)) const char *beebs_levenshtein_string_ptr(
    long idx) {
  const char *p = string4;
  switch (idx) {
  case 0:
    p = string0;
    break;
  case 1:
    p = string1;
    break;
  case 2:
    p = string2;
    break;
  case 3:
    p = string3;
    break;
  default:
    p = string4;
    break;
  }
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) long beebs_levenshtein_strlen(
    const char *s) {
  long len = 0;
  while (s[len] != 0)
    len++;
  return len;
}

int levenshtein_distance(const char *s, const char *t) {
  long sl = beebs_levenshtein_strlen(s);
  long tl = beebs_levenshtein_strlen(t);
  int d[(MAX_STRING_LEN + 1) * (MAX_STRING_LEN + 1)];

  for (long i = 0; i <= sl; i++)
    d[i * (MAX_STRING_LEN + 1)] = (int)i;

  for (long j = 0; j <= tl; j++)
    d[j] = (int)j;

  for (long j = 1; j <= tl; j++) {
    for (long i = 1; i <= sl; i++) {
      long cur = i * (MAX_STRING_LEN + 1) + j;
      long prev_row = (i - 1) * (MAX_STRING_LEN + 1);
      long row = i * (MAX_STRING_LEN + 1);
      if (s[i - 1] == t[j - 1]) {
        d[cur] = d[prev_row + j - 1];
      } else {
        d[cur] = min(d[prev_row + j] + 1,
                     min(d[row + j - 1] + 1, d[prev_row + j - 1] + 1));
      }
    }
  }

  return d[sl * (MAX_STRING_LEN + 1) + tl];
}

void initialise_benchmark(void) {}

int benchmark(void) {
  volatile unsigned sum = 0;

  for (long i = 0; i < NUM_STRINGS; i++) {
    for (long j = 0; j < NUM_STRINGS; j++) {
      const char *s = beebs_levenshtein_string_ptr(i);
      const char *t = beebs_levenshtein_string_ptr(j);
      sum += levenshtein_distance(s, t);
    }
  }

  return sum;
}

int verify_benchmark(int r) {
  return r == 122;
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
  -c "$PATCHED_LEVENSHTEIN_SRC" \
  -o "$OBJ_DIR/beebs_levenshtein.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_simple_domain.c" \
  -o "$OBJ_DIR/beebs_levenshtein_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/beebs_levenshtein.o" \
  "$OBJ_DIR/beebs_levenshtein_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
