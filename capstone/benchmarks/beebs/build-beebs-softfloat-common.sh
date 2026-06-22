#!/usr/bin/env bash
# Reusable compiler-rt soft-float builtin set for Capstone FP benchmarks.
#
# Capstone has no FP hardware ABI in the bare-metal domain, so all float/double
# operations lower to compiler-rt soft-float libcalls.  Source this after
# defining: CLANG, OBJ_DIR, COMPILER_RT (compiler-rt/lib/builtins), and the
# COMMON_FLAGS array.  It compiles the builtins into $OBJ_DIR and appends the
# object paths to the `softfloat_objs` array for the caller to link.
#
# See capstone/agent-handoff/design/capstone-softfloat-libm.md.

if [[ -z "${CLANG:-}" || -z "${OBJ_DIR:-}" || -z "${COMPILER_RT:-}" ]]; then
  echo "build-beebs-softfloat-common.sh: CLANG, OBJ_DIR, COMPILER_RT must be set" >&2
  exit 1
fi

# double, float, conversion, and fp-environment builtins.  Extend here if a new
# benchmark surfaces an additional undefined __*sf/__*df symbol.
CAPSTONE_SOFTFLOAT_BUILTINS=(
  adddf3 subdf3 muldf3 divdf3 fixdfsi floatsidf comparedf2
  addsf3 subsf3 mulsf3 divsf3 fixsfsi floatsisf comparesf2
  extendsfdf2 truncdfsf2
  fp_mode
)

softfloat_objs=()
for b in "${CAPSTONE_SOFTFLOAT_BUILTINS[@]}"; do
  "$CLANG" "${COMMON_FLAGS[@]}" -I"$COMPILER_RT" \
    -c "$COMPILER_RT/$b.c" -o "$OBJ_DIR/softfloat-$b.o"
  softfloat_objs+=("$OBJ_DIR/softfloat-$b.o")
done
