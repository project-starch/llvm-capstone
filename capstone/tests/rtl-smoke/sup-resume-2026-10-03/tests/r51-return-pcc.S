# See LICENSE for license details.

#*****************************************************************************
# r51-return-pcc.S  (R-51: after a RETURN, does the resumed domain still run under its PC capability?  2026-10-05)
#-----------------------------------------------------------------------------
#
# RETURN takes an integer resume point. The RTL parks it in the seal's PC slot with tag 0 and metadata 0
# (ex_stage.sv: "RETURN: return point is a plain integer"), so when the caller CALLs the domain again the frontend
# installs a PLAIN pc and commit_stage's pc_cap_check -- type, execute permission, bounds, revocation -- stays off until
# a capability control-flow instruction re-tags the PC. capstone-qemu instead stores the domain's CURRENT pc_cap with
# only the cursor moved (capstone_helper.c swap_pc), so QEMU keeps enforcing the code bounds after a yield.
#
# The caller builds a domain whose code capability covers [stub, stub_end) only, seals it with a trap-handler capability
# in the ctvec slot, and CALLs it. The domain parks the SEALEDRET, yields with RETURN, (resume point: resume_pt, inside
# its code), the caller CALLs again, and the resumed domain does an INTEGER jalr to `outside`, which is mapped and
# executable (inside the caller's CAPENTER text) but OUTSIDE the domain's code capability.
#   -DJUMP_BEFORE: the matched arm -- the same jalr at the FIRST entry, before any yield (the PC is the tagged slot-0
#                  capability then): must fault with 28 on every tree.
# `outside` writes 0xBAD to the shared word and returns; the domain's trap handler writes mcause there and returns.
# Readings (CAPPRINT): #1 shared word after the first return (0: the yield happened, nothing written yet; JUMP_BEFORE:
# 28), #2 shared word after the second return (28 = the PC capability was enforced after the yield, 0xBAD = it was not),
# #3 mcause in the caller (0).

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc

.section .text.init
.globl _start
_start:

li gp, 1
la t0, trap_handler
csrw mtvec, t0
csrw mcause, x0

lla a0, _start
lla a1, _end_of_text
CAPENTER(a0, a1)

# s1: NONLIN RW over the shared word, kept across the switch (not an exchanged register)
CAPCREATE(s1)
li a0, CAP_TYPE_NONLIN
lla a3, buf
lla a4, buf_end
li a5, CAP_PERM_RW
CAPTYPE(s1, a0)
CAPBOUND(s1, a3, a4)
CAPPERM(s1, a5)
sd x0, 0(s1)

# a2: the domain's code capability, [stub, stub_end) only -- `outside` is NOT in it
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub
lla a4, stub_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
# a6: the domain's trap-handler capability, same bounds, cursor at dom_trap
CAPCREATE(a6)
li a0, CAP_TYPE_NONLIN
lla a3, stub
lla a4, stub_end
li a5, CAP_PERM_RWX
CAPTYPE(a6, a0)
CAPBOUND(a6, a3, a4)
CAPPERM(a6, a5)
lla a3, dom_trap
lla a4, stub
sub a3, a3, a4
CINCOFFSET(a6, a6, a3)
# a1: the seal; slot 0 = the code capability (the domain's first PC), slot 1 = ctvec, mstatus image MIE = 0
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, seal
lla a4, seal_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)
STC(a1, a6, 16)
li a0, 0x0000000A00040000
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)

# first entry
CALL(a3, a3)
nop
nop
ld t4, 0(s1)
CAPPRINT(t4)                   # #1: 0 (the domain yielded; JUMP_BEFORE: 28)
#ifndef JUMP_BEFORE
# the resume
CALL(a3, a3)
nop
nop
ld t4, 0(s1)
CAPPRINT(t4)                   # #2: 28 = enforced after the yield; 0xBAD = the resumed PC was untagged
#endif
csrr t0, mcause
CAPPRINT(t0)                   # 0
CAP_PASS(s11)

.align 4
trap_handler:
  csrr s4, mcause
  CAPPRINT(s4)
  CAP_PASS(s11)

# ---------------------------------------------------------------- the domain: its code capability covers exactly this
.align 4
stub:
  MOVC(s3, x1)                 # park the SEALEDRET
#ifdef JUMP_BEFORE
  la t1, outside
  jalr x0, 0(t1)               # before any yield: the PC is the tagged slot-0 capability -> 28 on every tree
#endif
  MOVC(x1, s3)                 # the SEALEDRET back in ra (RETURN's rd), as every RETURN below does
  la a1, resume_pt
  RETURN(ra, a1, x0)           # the yield; resume point = resume_pt, an INTEGER
resume_pt:
  MOVC(s3, x1)                 # park the new SEALEDRET
  la t1, outside
  jalr x0, 0(t1)               # after the yield: 28 if the PC capability came back, 0xBAD if the PC is untagged
dom_trap:
  csrr t2, mcause
  sd t2, 0(s1)
  MOVC(x1, s3)
  la a1, stub
  RETURN(ra, a1, x0)
stub_end:

# ---------------------------------------------------------------- outside the domain's code capability, inside the text
.align 4
outside:
  li t2, 0xBAD
  sd t2, 0(s1)
  MOVC(x1, s3)
  la a1, stub
  RETURN(ra, a1, x0)

.align 4
_end_of_text:

.data
RVTEST_DATA_BEGIN
.align 6
buf:
  .zero 64
buf_end:
.align 10
seal:
  .zero 2048
seal_end:
  TEST_DATA
RVTEST_DATA_END
