#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

"$SCRIPT_DIR/fetch-coremark.sh" >/dev/null

REPO_ROOT=$CAPSTONE_REPO_ROOT
COREMARK_SRC_DIR=${COREMARK_SRC_DIR:-$CAPSTONE_TMP_ROOT/coremark-src}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/coremark-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj}
PORT_DIR="$SCRIPT_DIR/port"
CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O2}
COREMARK_TOTAL_DATA_SIZE=${COREMARK_TOTAL_DATA_SIZE:-1200}
COREMARK_ITERATIONS=${COREMARK_ITERATIONS:-10}
OUT_DOM=${OUT_DOM:-$OUT_DIR/coremark_capstone.dom}

mkdir -p "$OUT_DIR" "$OBJ_DIR"

if [[ ! -f "$COREMARK_SRC_DIR/coremark.h" ]]; then
  echo "missing CoreMark source tree: $COREMARK_SRC_DIR" >&2
  exit 1
fi

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature
  -Xclang +m
  -ffreestanding
  -fno-builtin
  "$DOMAIN_OPT_LEVEL"
  -I"$SCRIPT_DIR"
  -I"$PORT_DIR"
  -I"$COREMARK_SRC_DIR"
  -DTOTAL_DATA_SIZE="$COREMARK_TOTAL_DATA_SIZE"
  -DITERATIONS="$COREMARK_ITERATIONS"
  -DPROFILE_RUN=1
  -Dmain=coremark_main
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m -ffreestanding -O0 \
  -I"$PORT_DIR" \
  -I"$COREMARK_SRC_DIR" \
  -c "$START_SRC" \
  -o "$OBJ_DIR/start.o"

# core_main_capstone.c: local copy of upstream core_main.c with profile-run CRC
# table entries (index 2) replaced with Capstone PureCap-specific values.
# sizeof(list_head)=32 on Capstone PureCap (two 16-byte cap pointers) yields 9
# list nodes instead of 18, producing deterministically different but correct CRCs.
"$CLANG" "${COMMON_FLAGS[@]}" -c "$SCRIPT_DIR/core_main_capstone.c" -o "$OBJ_DIR/core_main.o"

# coremark_domain_entry.S: hand-written assembly for domain_main.
# The Capstone LLVM backend cannot generate correct code for domain_main at any
# -O level (see comment in coremark_domain_entry.S for details).
"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 \
  -c "$SCRIPT_DIR/coremark_domain_entry.S" \
  -o "$OBJ_DIR/coremark_domain_entry.o"

# ee_printf_asm.S: assembly trampoline for ee_printf.
# The compiler-generated va_list stores the vararg pointer as a scalar integer
# (sd, not stc); when reloaded via ld the tag is 0 and subsequent memory
# dereference crashes in cap_mem mode.  The trampoline forwards a0-a7 unchanged
# to ee_printf_impl() — no va_list is ever constructed.
"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 \
  -c "$SCRIPT_DIR/ee_printf_asm.S" \
  -o "$OBJ_DIR/ee_printf_asm.o"

# coremark_domain.c: only contains global declarations (hc_metadata, hc_payload,
# g_region_count).  -fno-zero-initialized-in-bss keeps the zero-initialized globals
# in .data rather than .bss (NOLOAD) so they fall within pc_cap bounds.
"$CLANG" "${COMMON_FLAGS[@]}" -fno-zero-initialized-in-bss \
  -c "$SCRIPT_DIR/coremark_domain.c" \
  -o "$OBJ_DIR/coremark_domain.o"

# core_state.c: compile with core_init_state renamed so the local override wins.
# -fno-jump-tables: core_state_transition has a multi-way switch.  At -O2 the compiler
# generates a jump table with 32-bit integer addresses and loads from it via plain `lw`
# (using a GP-derived integer as the base).  In cap_mem mode, `lw` requires a capability
# base, so the scalar table lookup crashes with "cap mem access requires capability".
# Disabling jump tables forces if-else chains which use only capability-safe operations.
# -fno-optimize-sibling-calls: core_bench_state ends with a crc16 tail call; the Capstone
# backend emits `cjalr ra, 0x0(a2)` (a call) rather than restoring ra then `cjalr zero`,
# so crc16 "returns" to core_state_transition's entry instead of the real caller.
# Same backend bug as core_bench_matrix; same workaround.
"$CLANG" "${COMMON_FLAGS[@]}" -fno-jump-tables -fno-optimize-sibling-calls \
  -Dcore_init_state=core_init_state_upstream_unused \
  -c "$COREMARK_SRC_DIR/core_state.c" \
  -o "$OBJ_DIR/core_state.o"

# Current benchmark-local runtime workarounds:
# - core_list_join.c at -O1: avoids the current high-opt list-path capability
#   copy trap while keeping the benchmark logic intact.
# - core_list_capstone.c: fixes upstream list sizing that still assumes 16-byte
#   per-node pointer storage and explicitly aligns the list storage for
#   capability-bearing list_head nodes on Capstone PureCap.
# - core_state_capstone.c: replaces core_init_state() with a version that uses
#   flat 2D char arrays instead of static pointer arrays.  The upstream four
#   pointer arrays (intpat, floatpat, scipat, errpat) are 16-byte capability
#   tables on Capstone PureCap; without runtime capability initialization the
#   ldc-then-cincoffset sequence faults with helper_cscincoffset/rs1_v->tag.
# - core_matrix_capstone.c: uses a local capability-safe matrix initializer while
#   the upstream function still triggers a mixed scalar/capability lowering bug.
# - core_state.c: -fno-jump-tables avoids scalar switch-table `lw` in cap_mem mode
#   (see comment at core_state.c compile step above).
# - core_util.c: avoid compiler-generated switch tables of capability-valued
#   addresses until generic static-cap table materialization exists.
#   Also compiled at -O0 to prevent loop-to-table transformation in crcu8:
#   at -O1 the compiler inlines crcu8 twice into crcu16 and shares a single
#   gp-derived LINEAR table pointer across both accesses; the first cincoffset
#   consumes the pointer and the second fires helper_cscincoffset/rs1_v->tag.
#   At -O0 the loop stays as 8 bit-manipulation iterations with no table, so
#   no static-cap LINEAR aliasing occurs.  Root fix is delin emission in the
#   LLVM backend; -O0 is the bring-up workaround.
# - core_util_capstone.c: replaces crcu8(), crcu16(), and check_data_types().
#   crcu8/crcu16: upstream locals are ee_u8 (1-byte); at -O0 they spill to
#   byte-sized stack slots inside the same 16-byte granule as a caller's saved
#   capability, clearing its tag.  Fix: widen to unsigned int.
#   check_data_types: upstream checks sizeof(ee_ptr_int)==sizeof(int*); on
#   Capstone PureCap uintptr_t=8 vs int*=16 (intentional cursor vs. cap).
#   The mismatch increments total_errors and masks real CRC failures.  Fix:
#   provide a replacement that omits the ee_ptr_int size check.
# - core_portme.c: keep zero-initialized seed globals in .data so gp-relative
#   capability bounds still cover the full volatile seed set.
# - core_portme.c runs all three algorithms (COREMARK_DEFAULT_EXECS=7) so that
#   core_init_matrix is called and mat_params is populated with valid NONLIN caps
#   before core_bench_matrix is invoked from calc_func during list traversal.
"$CLANG" "${COMMON_FLAGS[@]}" -fno-jump-tables -O1 \
  -Dcore_list_init=core_list_init_upstream_unused \
  -c "$COREMARK_SRC_DIR/core_list_join.c" \
  -o "$OBJ_DIR/core_list_join.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/core_list_capstone.c" \
  -o "$OBJ_DIR/core_list_capstone.o"

"$CLANG" "${COMMON_FLAGS[@]}" -fno-jump-tables \
  -c "$SCRIPT_DIR/core_state_capstone.c" \
  -o "$OBJ_DIR/core_state_capstone.o"

# core_matrix.c: core_init_matrix renamed so core_matrix_capstone.c override wins.
# core_matrix.c compiled at -O2:
# - core_init_matrix renamed so core_matrix_capstone.c override wins.
# - Inner matrix loops trigger LINEAR row/col pointer hoisting at runtime;
#   -O0/-O1 -fno-inline crash the LLVM backend (i128 shift / UNREACHABLE), so -O2.
# - core_bench_matrix tail-calls crc16 via `cjalr ra, imm(a2)` (a CALL, not a jump),
#   setting ra=next_inst=0x11b84=matrix_test entry.  crc16's return then lands at
#   matrix_test with garbled registers → cincoffset crash.  This is a Capstone backend
#   tail-call lowering bug (should restore ra then emit `cjalr zero`).
#   Workaround: -fno-optimize-sibling-calls disables the bad tail call so core_bench_matrix
#   generates a proper call+epilogue instead.
"$CLANG" "${COMMON_FLAGS[@]}" -fno-optimize-sibling-calls \
  -Dcore_init_matrix=core_init_matrix_upstream_unused \
  -c "$COREMARK_SRC_DIR/core_matrix.c" \
  -o "$OBJ_DIR/core_matrix.o"

"$CLANG" "${COMMON_FLAGS[@]}" \
  -c "$SCRIPT_DIR/core_matrix_capstone.c" \
  -o "$OBJ_DIR/core_matrix_capstone.o"

"$CLANG" "${COMMON_FLAGS[@]}" -fno-jump-tables -O0 \
  -Dcrcu8=crcu8_upstream_unused \
  -Dcrcu16=crcu16_upstream_unused \
  -Dcheck_data_types=check_data_types_upstream_unused \
  -c "$COREMARK_SRC_DIR/core_util.c" \
  -o "$OBJ_DIR/core_util.o"

"$CLANG" "${COMMON_FLAGS[@]}" -fno-jump-tables -O0 \
  -c "$SCRIPT_DIR/core_util_capstone.c" \
  -o "$OBJ_DIR/core_util_capstone.o"

# core_portme.c: -O0 prevents capability hoisting across loop iterations.
# At higher -O levels the backend emits cincoffset rd≠rs1 (consuming the
# LINEAR cap) once before the loop; subsequent iterations assert on tag=0.
# At -O0 fmt/pay/string caps are reloaded from their RSA stack slots each
# iteration via ldc, so no cross-iteration capability consumption occurs.
# ee_printf is defined in ee_printf_asm.S; core_portme.c provides ee_printf_impl.
"$CLANG" "${COMMON_FLAGS[@]}" -fno-zero-initialized-in-bss -O0 \
  -DCOREMARK_DEFAULT_EXECS=7 \
  -c "$SCRIPT_DIR/port/core_portme.c" \
  -o "$OBJ_DIR/core_portme.o"

"$LD_LLD" -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/core_list_join.o" \
  "$OBJ_DIR/core_list_capstone.o" \
  "$OBJ_DIR/core_main.o" \
  "$OBJ_DIR/core_matrix.o" \
  "$OBJ_DIR/core_matrix_capstone.o" \
  "$OBJ_DIR/core_state.o" \
  "$OBJ_DIR/core_state_capstone.o" \
  "$OBJ_DIR/core_util.o" \
  "$OBJ_DIR/core_util_capstone.o" \
  "$OBJ_DIR/core_portme.o" \
  "$OBJ_DIR/ee_printf_asm.o" \
  "$OBJ_DIR/coremark_domain.o" \
  "$OBJ_DIR/coremark_domain_entry.o"

"$LLVM_READOBJ" -h "$OUT_DOM"

echo "Built $OUT_DOM"

