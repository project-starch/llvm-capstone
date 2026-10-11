# See LICENSE for license details.

#*****************************************************************************
# sup-arm.S  (supervised CALL on silicon, verification step 2: CSSUPERVISE's statuses)
#-----------------------------------------------------------------------------
#
# CSSUPERVISE rd, rs1(seal), rs2(save area) returns a status in rd and leaves rs1 and rs2 alone:
#   0 armed   1 the seal's revocation node is dead   2 refused   (3, an unread event, is step 3's business)
# Arms, in order (CAPPRINT values in the retirement trace):
#   #1 a live SEALED seal and a 1 KiB aligned LINEAR save area        -> 0
#   #2 the forget form (rs1 = x0)                                        -> 0
#   #3 rs1 a LINEAR capability (not sealed)                             -> 2
#   #4 rs2 a NONLIN capability (the save area must be LINEAR)           -> 2
#   #5 rs2 a 512-byte LINEAR region (too small)                         -> 2
#   #6 rs2 a LINEAR region starting 8 bytes into a 64-byte-aligned one  -> 2  (misaligned)
#   #7 rd == rs2 (a throwaway save area: the refusal still writes rd)   -> 2
#   #11 (last) a 512-byte region through SEAL: with S-11 fixed, SEAL raises 29 -- the handler's mcause ends the test
#   #8a the seal after a memory round trip (STC/LDC), no revocation      -> LCC 4
#   #8b the seal reloaded after its node was REVOKEd                    -> LCC 4 (then #8 = 1) or 7 (then #8 = 2: a load nulls it)
#   #8 cssupervise on that reloaded seal                                 -> 1, or 2 if the load already nulled it
#   #9 the seal's type after all of that (LCC selector 1)               -> 4 (SEALED: rs1 is never consumed), 7 if nulled
#   #10 mcause                                                           -> 0 (no trap)
# Mutant (positive control): statuses 1 and 2 swapped in the dyn unit -> #3..#7 read 1 and #8 reads 2.
# RETRACTION (after-audit, 2026-10-01 evening): until then `lla a3, savearea2` in the #7 block overwrote the seal with an
# integer, so #7 refused on rs1 (not on rd == rs2) and #8a/#8b/#8/#9 measured an integer's round trip -- the "a SEALED
# capability reads back untagged after STC/LDC" property and "status 1 is unreachable" came from that, not from the
# hardware. Pre-registered for the repaired test: #7 2, #8a 4, #8b 4 or 7, #8 1 or 2 (a 0 would mean REVOKE of the MREV
# handle did not kill the seal's node: a question about MREV, not about cssupervise), #9 4 or 7.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

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

# s5: the save area, LINEAR RWX, 1 KiB, 64-byte aligned
CAPCREATE(s5)
li a0, CAP_TYPE_LIN
lla a3, savearea
lla a4, savearea_end
li a5, CAP_PERM_RWX
CAPTYPE(s5, a0)
CAPBOUND(s5, a3, a4)
CAPPERM(s5, a5)

# s6: a 512-byte LINEAR region (too small)
CAPCREATE(s6)
li a0, CAP_TYPE_LIN
lla a3, small
lla a4, small_end
li a5, CAP_PERM_RWX
CAPTYPE(s6, a0)
CAPBOUND(s6, a3, a4)
CAPPERM(s6, a5)

# s7: a 1 KiB LINEAR region starting 8 bytes into the aligned block (misaligned)
CAPCREATE(s7)
li a0, CAP_TYPE_LIN
lla a3, misaligned
addi a3, a3, 8
lla a4, misaligned_end
li a5, CAP_PERM_RWX
CAPTYPE(s7, a0)
CAPBOUND(s7, a3, a4)
CAPPERM(s7, a5)

# s8: a NONLIN capability
CAPCREATE(s8)
li a0, CAP_TYPE_NONLIN
lla a3, savearea
lla a4, savearea_end
li a5, CAP_PERM_RWX
CAPTYPE(s8, a0)
CAPBOUND(s8, a3, a4)
CAPPERM(s8, a5)

# a6: a LINEAR capability that is not a seal
CAPCREATE(a6)
li a0, CAP_TYPE_LIN
lla a3, savearea
lla a4, savearea_end
li a5, CAP_PERM_RWX
CAPTYPE(a6, a0)
CAPBOUND(a6, a3, a4)
CAPPERM(a6, a5)

# the seal: a1 LINEAR over the seal region, an MREV handle on it in s9 (so its node can be revoked later), then SEAL
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub
lla a4, stub_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a0)
li t0, CAP_TYPE_LIN
lla a3, sealbuf
lla a4, sealbuf_end
li a5, CAP_PERM_RWX
CAPTYPE(a0, t0)
CAPBOUND(a0, a3, a4)
CAPPERM(a0, a5)
lla t0, seal
SPLIT(a1, a0, t0)              # a1 = [seal, sealbuf_end): a CHILD node of its own (MREV on a debug-minted
MREV(s9, a1)                   # capability would hand REVOKE the shared node every CAPCREATE uses)
STC(a1, a2, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)
SEAL(a3, a1)

CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # #1 0
CSSUPERVISE(t3, x0, x0)
addi s2, t3, 0
CAPPRINT(s2)                   # #2 0 (forget)
CSSUPERVISE(t3, a6, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # #3 2
CSSUPERVISE(t3, a3, s8)
addi s2, t3, 0
CAPPRINT(s2)                   # #4 2
CSSUPERVISE(t3, a3, s6)
addi s2, t3, 0
CAPPRINT(s2)                   # #5 2
CSSUPERVISE(t3, a3, s7)
addi s2, t3, 0
CAPPRINT(s2)                   # #6 2
# #7 uses a throwaway save area: the refusal still writes the status into rd
CAPCREATE(s10)
li a0, CAP_TYPE_LIN
lla t1, savearea2              # t1/t2, NOT a3/a4: a3 is the seal (the clobber the after-audit found, see the header)
lla t2, savearea2_end
li a5, CAP_PERM_RWX
CAPTYPE(s10, a0)
CAPBOUND(s10, t1, t2)
CAPPERM(s10, a5)
CSSUPERVISE(s10, a3, s10)
addi s2, s10, 0
CAPPRINT(s2)                   # #7 2 (rd == rs2)
# the dead-node arm: the seal is parked in MEMORY before the revoke; a register copy may be nulled by the revocation
# broadcast, and the memory copy is what a monitor holds. Nothing in this arm had been measured before 2026-10-01 evening:
# the earlier readings were of an integer (see the header)
CAPCREATE(s1)
li a0, CAP_TYPE_NONLIN
lla t1, stash                  # (not a3/a4: a3 is the seal)
lla t2, stash_end
li a5, CAP_PERM_RWX
CAPTYPE(s1, a0)
CAPBOUND(s1, t1, t2)
CAPPERM(s1, a5)
STC(s1, a3, 0)                 # park the live seal in memory (a3 is consumed: the seal is linear-family)
li t1, 300                     # S-07: a reload while the store sits in the write buffer reads the STALE tag; its reproducer
1: addi t1, t1, -1             # needed a 300-iteration drain (16 nops were not enough: LCC 7, measured 2026-10-01)
   bnez t1, 1b
fence
LDC(a3, s1, 0)                 # control: the round trip without a revocation keeps the seal
LCC(s3, a3, x1)
CAPPRINT(s3)                   # #8a 4 (a SEALED capability survives a memory round trip)
STC(s1, a3, 0)
li t1, 300
2: addi t1, t1, -1
   bnez t1, 2b
fence
REVOKE(s9)                     # its node dies; the parked copy is reachable by no register tracker
LDC(a3, s1, 0)                 # a3 := the stale seal -- or, if a load checks the node, an untagged value
LCC(s3, a3, x1)
CAPPRINT(s3)                   # #8b 4 if the stale seal reloads SEALED (then #8 reads 1), 7 if the load nulled it (then #8 reads 2)
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # #8 1 (the seal's node is dead)
LCC(s3, a3, x1)
CAPPRINT(s3)                   # #9 4
csrr t0, mcause
CAPPRINT(t0)                   # #10 0
# a 512-byte "seal": with S-11 fixed, SEAL itself refuses it (cause 29) -- the handler's reading ends the test
CAPCREATE(a7)
li a0, CAP_TYPE_LIN
lla a3, small
lla a4, small_end
li a5, CAP_PERM_RWX
CAPTYPE(a7, a0)
CAPBOUND(a7, a3, a4)
CAPPERM(a7, a5)
SEAL(a7, a7)
CSSUPERVISE(t3, a7, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # (not reached: SEAL trapped above; mcause 29 is the last reading)
CAP_PASS(s11)

.align 4
trap_handler:
  csrr s4, mcause
  CAPPRINT(s4)                 # a trap ends the test: its cause is the last reading
  CAP_PASS(s11)

.align 4
stub:
  1: j 1b
stub_end:

.align 4
_end_of_text:

.data
RVTEST_DATA_BEGIN
.align 6
savearea:
  .zero 1024
savearea_end:
.align 6
small:
  .zero 512
small_end:
.align 6
misaligned:
  .zero 1088
misaligned_end:
.align 6
savearea2:
  .zero 1024
savearea2_end:
.align 6
stash:
  .zero 64
stash_end:
.align 6
sealbuf:
  .zero 2048
seal:
  .zero 2048
sealbuf_end:
  TEST_DATA
RVTEST_DATA_END
