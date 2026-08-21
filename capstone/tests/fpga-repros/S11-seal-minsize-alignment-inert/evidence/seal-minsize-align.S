# See LICENSE for license details.

#*****************************************************************************
# seal-minsize-align.S
#-----------------------------------------------------------------------------
#
# Does SEAL enforce the spec's minimum-size and base-alignment preconditions?
#
# THIS TEST IS EXPECTED TO FAIL ON THE CURRENT RTL, AND THAT FAILURE IS THE POINT.
# It exists because no test anywhere creates SEAL's rejecting condition: the only
# other dedicated SEAL test seals a 16 KiB region and exercises the accept path
# only, so it has never had a positive control.
#
# THE SPEC, capstone-spec/parts/cap-man-insn.adoc:459-462, requires
# "Illegal operand value (29)" -- mcause 30 in riscv_pkg.sv:354 -- when either:
#   * the region is smaller than CLENBYTES * sealed_cap_size_clen = 16 * 64 = 1024 bytes, or
#   * the base is not aligned to CLENBYTES = 16 bytes.
#
# WHY IT MATTERS. A sealed-return capability's load/store window is bounded from
# start alone -- capstone_dyn_unit.anvil:323-324 computes start+48 .. start+1008 and
# never consults end, because prog-model.adoc:91 says end is not applicable to sealed
# types. So this precondition is the only thing standing between an undersized region
# and a 1024-byte access window. Under-enforce it and a 64-byte capability becomes
# roughly 960 bytes of authority its holder never had.
#
# WHAT THE RTL DOES INSTEAD, read from the generated netlist at
# core/capstone_flu_unit.anvil.sv:2204-2205: the source condition
# "size < max_size || ((start & 15) != 0)" generates as 1024 || X, which folds to the
# constant 1, so the whole test reduces to size < 1. Since size = end - start + 1,
# that is unreachable for any capability with end >= start. No check survives.
#
# THREE ARMS, each differing from the control in EXACTLY ONE property:
#   arm 1  control     16 KiB region, 16-byte aligned base   -> must NOT trap
#   arm 2  undersized  64-byte region, 16-byte aligned base  -> must trap, size only
#   arm 3  misaligned  16 KiB region, base + 8               -> must trap, alignment only
#
# TESTNUM on failure:
#   1  the control arm trapped -- the harness is broken, the other arms mean nothing
#   2  ONLY the undersized region was accepted (size unchecked, alignment enforced)
#   3  ONLY the misaligned base was accepted   (alignment unchecked, size enforced)
#   5  BOTH were accepted -- the whole condition is inert, which is what the netlist
#      analysis predicts, since both collapse into the same unreachable size == 0 test
#   4  a trap arrived with an unexpected mcause
#
# Every arm runs to completion before anything is reported, so one defect does not
# hide the other. A test that stops at its first failing arm returns one bit; this
# one returns both.
#
# NOT COVERED. The spec lists a third rejecting condition, that the region
# [base+CLENBYTES, base+2*CLENBYTES) must contain a capability. The Anvil source does
# not implement it at all, so this test does not probe it and does not store a
# capability into the regions first.
#

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc
.section .text.init
.globl _start
_start:

  li gp, 1
  csrwi mie, 0

# capmode must be on before capability instructions are accepted; CAPENTER stays in
# M-mode and only turns it on -- same as the other capstone security tests.
  la a0, _start
  la a1, _end_of_text
  CAPENTER(a0, a1)

  la t0, trap_handler
  csrw mtvec, t0
  la t0, trap_count
  sw x0, 0(t0)

# Build a LINEAR RWX capability over [LO, HI] in CREG. SEAL rejects a non-LINEAR
# operand with mcause 27 and an under-permissioned one with mcause 28, so both must be
# right or the arm measures the wrong precondition.
#define MKLIN(CREG, LO, HI)   \
  CAPCREATE(CREG);            \
  li  a0, CAP_TYPE_LIN;       \
  lla a3, LO;                 \
  lla a6, HI;                 \
  li  a5, CAP_PERM_RWX;       \
  CAPTYPE(CREG, a0);          \
  CAPBOUND(CREG, a3, a6);     \
  CAPPERM(CREG, a5);

# ----------------------------------------------------------------- arm 1: control
# 16 KiB, base 16-byte aligned. Spec says seal it. If this traps, nothing below is
# interpretable.
  MKLIN(a1, seal_ok_start, seal_ok_end)
  SEAL(a2, a1)
  la   t0, trap_count
  lw   t1, 0(t0)
  bnez t1, fail_control_trapped

# ------------------------------------------------------------- arm 2: undersized
# 64 bytes, base still 16-byte aligned. Differs from the control in SIZE ONLY.
  MKLIN(a1, seal_small_start, seal_small_end)
  SEAL(a2, a1)
  la   t0, trap_count
  lw   s2, 0(t0)                 /* s2 = 1 if the undersized region was rejected */

# ------------------------------------------------------------- arm 3: misaligned
# 16 KiB region entered at base+8, so size stays far above 1024 and only the base
# alignment differs from the control.
  CAPCREATE(a1)
  li  a0, CAP_TYPE_LIN
  lla a3, seal_ok_start
  addi a3, a3, 8
  lla a6, seal_ok_end
  li  a5, CAP_PERM_RWX
  CAPTYPE(a1, a0)
  CAPBOUND(a1, a3, a6)
  CAPPERM(a1, a5)
  SEAL(a2, a1)
  la   t0, trap_count
  lw   s3, 0(t0)
  sub  s3, s3, s2                /* s3 = 1 if the misaligned base was rejected */

# ------------------------------------------------------------------ report
# Both arms have now run. s2 and s3 are 1 where the spec's rejection happened and 0
# where it did not.
  li   t2, 1
  bne  s2, t2, small_accepted
  bne  s3, t2, only_misaligned_accepted
  RVTEST_PASS                    /* both rejected: SEAL enforces the spec */

small_accepted:
  bne  s3, t2, both_accepted
  li TESTNUM, 2
  j fail
only_misaligned_accepted: li TESTNUM, 3
                          j fail
both_accepted:            li TESTNUM, 5
                          j fail
fail_control_trapped:     li TESTNUM, 1
                          j fail
unexpected_trap:          li TESTNUM, 4
fail:
  RVTEST_FAIL

# Count only the cause we are testing for; anything else is a harness failure rather
# than a result. SEAL is a 4-byte .insn and .option norvc is set above, so mepc+4
# resumes at the instruction after it.
trap_handler:
  csrr t0, mcause
  bltz t0, unexpected_trap
  li   t1, 30
  bne  t0, t1, unexpected_trap
  la   t2, trap_count
  lw   t3, 0(t2)
  addi t3, t3, 1
  sw   t3, 0(t2)
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret

_end_of_text:

  .data
RVTEST_DATA_BEGIN

.align 4
trap_count: .word 0

.align 4
seal_ok_start:
  .zero 16384
seal_ok_end:

.align 4
seal_small_start:
  .zero 64
seal_small_end:

  TEST_DATA

RVTEST_DATA_END
