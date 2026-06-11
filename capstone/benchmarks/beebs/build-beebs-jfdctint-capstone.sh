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
OUT_DOM=${OUT_DOM:-$OUT_DIR/beebs_jfdctint_capstone.dom}

JFDCTINT_SRC=$BEEBS_SRC_DIR/src/jfdctint/libjfdctint.c
SUPPORT_DIR=$BEEBS_SRC_DIR/support
PATCHED_JFDCTINT_SRC=$OUT_DIR/libjfdctint_capstone.c

if [[ ! -f "$JFDCTINT_SRC" || ! -f "$SUPPORT_DIR/support.h" ]]; then
  echo "missing BEEBS jfdctint source tree: $BEEBS_SRC_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# Avoid lowering the upstream local expected-array initializer into the current
# unsupported memcpy call path; keep the DCT body unchanged.
awk '
  /^int verify_benchmark/ { exit }
  { print }
' "$JFDCTINT_SRC" > "$PATCHED_JFDCTINT_SRC"

cat >> "$PATCHED_JFDCTINT_SRC" <<'EOF'
static int beebs_jfdctint_expected(long idx) {
  if (idx == 0) return 1956823;
  if (idx == 1) return 184557;
  if (idx == 2) return -39350;
  if (idx == 3) return -94393;
  if (idx == 4) return -77163;
  if (idx == 5) return 5995;
  if (idx == 6) return 162871;
  if (idx == 7) return -3428;
  if (idx == 8) return 31856;
  if (idx == 9) return 57575;
  if (idx == 10) return -49784;
  if (idx == 11) return 43664;
  if (idx == 12) return 63854;
  if (idx == 13) return -9784;
  if (idx == 14) return 11398;
  if (idx == 15) return -23444;
  if (idx == 16) return 13102;
  if (idx == 17) return 59509;
  if (idx == 18) return 63748;
  if (idx == 19) return -34407;
  if (idx == 20) return -57064;
  if (idx == 21) return 11667;
  if (idx == 22) return 37414;
  if (idx == 23) return 41934;
  if (idx == 24) return 20234;
  if (idx == 25) return 25212;
  if (idx == 26) return -44504;
  if (idx == 27) return 25562;
  if (idx == 28) return -46366;
  if (idx == 29) return -4562;
  if (idx == 30) return -40816;
  if (idx == 31) return -64820;
  if (idx == 32) return -203745;
  if (idx == 33) return -15884;
  if (idx == 34) return -134082;
  if (idx == 35) return -126104;
  if (idx == 36) return 66045;
  if (idx == 37) return 23372;
  if (idx == 38) return -87152;
  if (idx == 39) return -147968;
  if (idx == 40) return 41739;
  if (idx == 41) return -20979;
  if (idx == 42) return -36653;
  if (idx == 43) return 23706;
  if (idx == 44) return 613;
  if (idx == 45) return 41593;
  if (idx == 46) return 34760;
  if (idx == 47) return -60639;
  if (idx == 48) return 30493;
  if (idx == 49) return -10396;
  if (idx == 50) return 13944;
  if (idx == 51) return -13980;
  if (idx == 52) return 52343;
  if (idx == 53) return -40116;
  if (idx == 54) return -55093;
  if (idx == 55) return 37532;
  if (idx == 56) return 61998;
  if (idx == 57) return -22500;
  if (idx == 58) return 25991;
  if (idx == 59) return -57098;
  if (idx == 60) return -18228;
  if (idx == 61) return 47265;
  if (idx == 62) return -48356;
  if (idx == 63) return 38613;
  return 0;
}

int verify_benchmark(int unused) {
  (void)unused;
  for (long i = 0; i < 64; ++i)
    if (data[i] != beebs_jfdctint_expected(i))
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
  "$DOMAIN_OPT_LEVEL"
  -I"$SUPPORT_DIR"
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 \
  -c "$START_SRC" \
  -o "$OBJ_DIR/start.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$PATCHED_JFDCTINT_SRC" \
  -o "$OBJ_DIR/beebs_jfdctint.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/beebs_jfdctint_domain.c" \
  -o "$OBJ_DIR/beebs_jfdctint_domain.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/beebs_jfdctint.o" \
  "$OBJ_DIR/beebs_jfdctint_domain.o"

"$LLVM_READOBJ" -h "$OUT_DOM"

if [[ ! -f "$OUT_DOM" ]]; then
  echo "failed to build $OUT_DOM" >&2
  exit 1
fi

echo "Built $OUT_DOM"
