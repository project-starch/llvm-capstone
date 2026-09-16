#!/usr/bin/env bash
# Reusable compiler-rt soft-float builtin set for Capstone FP benchmarks.
#
# Capstone has no FP hardware ABI in the bare-metal domain, so all float/double
# operations lower to compiler-rt soft-float libcalls.  Source this after
# defining: CLANG, OBJ_DIR, COMPILER_RT (compiler-rt/lib/builtins), and the
# COMMON_FLAGS array.  It compiles the builtins into $OBJ_DIR and appends the
# object paths to the `softfloat_objs` array for the caller to link.
#
# See capstone/docs/design/capstone-softfloat-libm.md.

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
  floatdisf floatundisf
  # floatunsisf: -O2 converts an unsigned int straight to float (sqrt did so on
  # 2026-09-05 and failed to link); -O0 went through the double conversions.
  floatunsisf
  floatunsidf floatdidf floatundidf fixdfdi fixunsdfdi fixunsdfsi
  fp_mode
  # TF mode, i.e. 128-bit long double. musl's vfprintf references the whole
  # family whether or not a caller uses %Lf, so ANY program that links printf
  # needs these: they were the last eleven undefined symbols the musl port had
  # (measured 2026-09-16). comparetf2 carries __eqtf2, __netf2 and __unordtf2
  # together, so nine files close eleven symbols. All nine compile for
  # capstone64 as they stand, which was the open question: compiler-rt computes
  # the significand in __uint128_t and MVT::i128 is this target's capability
  # carrier.
  addtf3 subtf3 multf3 comparetf2 extenddftf2
  fixtfsi fixunstfsi floatsitf floatunsitf
  # The second TF wave, from running libc-test rather than one probe: musl's
  # floatscan (behind strtod, sscanf and every scanf) divides in long double
  # and converts float and double to and from it, and tgmath converts float to
  # int64. Thirteen libc-test programs failed to link on exactly these four TF
  # symbols and one on __fixsfdi (2026-09-16).
  divtf3 extendsftf2 trunctfdf2 trunctfsf2
  fixsfdi fixunssfdi
)

softfloat_objs=()
for b in "${CAPSTONE_SOFTFLOAT_BUILTINS[@]}"; do
  "$CLANG" "${COMMON_FLAGS[@]}" -I"$COMPILER_RT" \
    -c "$COMPILER_RT/$b.c" -o "$OBJ_DIR/softfloat-$b.o"
  softfloat_objs+=("$OBJ_DIR/softfloat-$b.o")
done
