# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# vb-trans-plain.S  (item B / R-56 / Q-19 on silicon: a consuming ldc from a page WRITABLE from the start, 2026-10-18)
#-----------------------------------------------------------------------------
#
# The matched pair vb-trans's arm 4 lacks (the runtime lane, 2026-10-18): arm 4's refusal (a consuming ldc from a
# read-only page, csupcause 15) and its resumed admission differ by the page permission AND by the monitor's resume path
# (the PTE rewrite, the TLB flush). This arm differs from the refusal by the permission alone: the same consuming ldc,
# under translation inside a supervised domain, from a page mapped RWX from the start. The clear-refusal gate
# (load_unit.sv: clear_o = clear_wanted & ~(vctx_i && !ldc_store_ok_d)) must ADMIT it: the LINEAR capability is loaded, the
# source granule is cleared, the domain returns. A separate arm, so vb-trans.S stays byte-identical to its pre-registration.
# Setup as vb-trans.S (Sv39 identity over the first 2 MiB, every page RWX, a supervised domain under csupctl.virt).
#
# Pre-registered readings (CAPPRINT values in the retirement trace, in order), flag ON (the only build):
#   status 0 (armed); inside the domain: the reloaded source granule 0 (cleared by the admitted consuming ldc); after
#   the RETURN: csupstatus 1; scratch2[0] 0x66 (stored through the loaded LINEAR capability: it was admitted and is
#   usable); wpage[0] 0 (the parked word is gone); mcause 0 (no trap anywhere).
# vp1 (virtual-profile 9ed9a2594) reads this as built; vp2 (virtual-profile-2-d bf3db3175) needs -DVB_TABLE (a lifetime table
# at srevroot with record 2 valid, as every virt domain does on the item-A trees) and then reads 0 0 5 0x66 0 0 (read on
# bf3db3175 2026-10-18): the admitted consuming ldc's evidence is the same (0x66 stored through the loaded capability, the
# parked word gone), and csupstatus reads 5 because the domain's RETURN is refused in a virtual context (item E, cause 2).

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSUPSTATUS  0xFC0
#define CSUPCAUSE   0xFC1
#define CSUPEPC     0xFC2
#define CSUPTVAL    0xFC3
#define CSUPQUANTUM 0x7C3
#define CSUPCTL     0x7C4

#define PTE_LEAF_RWX  0xDF
#define PTE_PTR       0x01

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

# s5: the monitor's private save area, LINEAR RWX, 1 KiB
CAPCREATE(s5)
li a0, CAP_TYPE_LIN
lla a3, savearea
lla a4, savearea_end
li a5, CAP_PERM_RWX
CAPTYPE(s5, a0)
CAPBOUND(s5, a3, a4)
CAPPERM(s5, a5)

#define MINT_NONLIN_RW(REG, LO, HI, PERM) \
  CAPCREATE(REG); li a0, CAP_TYPE_NONLIN; lla a3, LO; lla a4, HI; li a5, PERM; \
  CAPTYPE(REG, a0); CAPBOUND(REG, a3, a4); CAPPERM(REG, a5)

MINT_NONLIN_RW(a6, wpage, wpage_end, CAP_PERM_RW)     # the page the LINEAR capability is parked in: WRITABLE
MINT_NONLIN_RW(t6, scratch2, scratch2_end, CAP_PERM_RW)   # the monitor's view of scratch2
MINT_NONLIN_RW(s8, pt_root, pt_end, CAP_PERM_RW)
sd x0, 0(a6)
sd x0, 8(a6)
sd x0, 0(t6)

#ifdef VB_TABLE
# ---------------------------------------------------------------- vp2 (item A): the lifetime table a virt domain needs
# RECYCLE | capacity 1024, next-unused 3, no free list, record 2 valid (CAPCREATE's node): without it every capability the
# domain touches reads DEAD in the table namespace (csupcause 25, found 2026-10-18). a2 is free here (ARM_SETUP remints it).
MINT_NONLIN_RW(a2, tbl, tbl_end, CAP_PERM_RW)
li t0, 1
slli t0, t0, 63
ori t0, t0, 1024
sd t0, 0(a2)
li t0, 3
sd t0, 8(a2)
sd x0, 16(a2)
sd x0, 24(a2)
li t0, 32
CINCOFFSET(t1, a2, t0)
sd x0, 0(t1)
li t0, 1
slli t0, t0, 60
sd t0, 8(t1)
lla t0, tbl
csrw 0x5C1, t0                 # srevroot
#endif

# a LINEAR RW capability over scratch2, parked in wpage (stored while the monitor is bare)
CAPCREATE(s6)
li a0, CAP_TYPE_LIN
lla a3, scratch2
lla a4, scratch2_end
li a5, CAP_PERM_RW
CAPTYPE(s6, a0)
CAPBOUND(s6, a3, a4)
CAPPERM(s6, a5)
STC(a6, s6, 0)

# ---------------------------------------------------------------- the page tables (Sv39, physical, built bare): identity, every page RWX
lla t0, pt_l0
lla t1, pt_root
sub a7, t0, t1                 # a7: L0's offset inside s8
lla t0, pt_l1
sub s9, t0, t1                 # s9 (x25): L1's offset inside s8
li t1, 0x80000000
li t2, 0
1:
srli t3, t1, 12
slli t3, t3, 10
ori t3, t3, PTE_LEAF_RWX
slli t4, t2, 3
add t4, t4, a7
CINCOFFSET(t0, s8, t4)
sd t3, 0(t0)
li t4, 0x1000
add t1, t1, t4
addi t2, t2, 1
li t4, 512
bne t2, t4, 1b
lla t1, pt_l0
srli t1, t1, 12
slli t1, t1, 10
ori t1, t1, PTE_PTR
CINCOFFSET(t0, s8, s9)
sd t1, 0(t0)
li t2, 1
2:
slli t3, t2, 21
li t4, 0x80000000
add t3, t3, t4
srli t3, t3, 12
slli t3, t3, 10
ori t3, t3, PTE_LEAF_RWX
slli t4, t2, 3
add t4, t4, s9
CINCOFFSET(t0, s8, t4)
sd t3, 0(t0)
addi t2, t2, 1
li t4, 512
bne t2, t4, 2b
lla t1, pt_l1
srli t1, t1, 12
slli t1, t1, 10
ori t1, t1, PTE_PTR
li t4, 16
CINCOFFSET(t0, s8, t4)
sd t1, 0(t0)
li t0, -1
csrw pmpaddr0, t0
li t0, 0x1f
csrw pmpcfg0, t0
lla t0, pt_root
srli t0, t0, 12
li t1, 8
slli t1, t1, 60
or t0, t0, t1
csrw satp, t0
sfence.vma

li t0, 0x100000
csrw CSUPQUANTUM, t0           # a quantum far longer than the arm
csrwi CSUPCTL, 2               # bit 1 = the virtual context

# ---------------------------------------------------------------- the one arm: a consuming ldc from the writable page is admitted
CAPCREATE(a2); li a0, CAP_TYPE_LIN; lla a3, stub0; lla a4, stub0_end; li a5, CAP_PERM_RWX
CAPTYPE(a2, a0); CAPBOUND(a2, a3, a4); CAPPERM(a2, a5)
CAPCREATE(a1); li a0, CAP_TYPE_LIN; lla a3, seal0; lla a4, seal0_end; li a5, CAP_PERM_RWX
CAPTYPE(a1, a0); CAPBOUND(a1, a3, a4); CAPPERM(a1, a5)
STC(a1, a2, 0); li a0, 0x0000000A00040000; sd a0, 48(a1); sd x0, 56(a1); sd x0, 64(a1); sd x0, 72(a1); sd x0, 80(a1)
SEAL(a3, a1); csrw mcause, x0; CSSUPERVISE(t3, a3, s5); addi s2, t3, 0; CAPPRINT(s2); CALL(a3, a3)   # 0: armed
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 1: the domain returned (the admitted ldc cleared its source)
ld t0, 0(t6)
CAPPRINT(t0)                   # 0x66: stored through the loaded LINEAR capability
ld t0, 0(a6)
CAPPRINT(t0)                   # 0: the parked word is gone
csrr t0, mcause
CAPPRINT(t0)                   # 0
CAP_PASS(s11)

trap_handler:
  csrr t0, mcause
  CAPPRINT(t0)
  CAP_PASS(s11)

# ---- the domain. No mint, no CSR, no CALL.
.align 6
stub0:
  MOVC(s3, x1)
  LDC(x30, a6, 0)              # consuming, the page writable: admitted; the source granule is cleared
  li t0, 0x66
  sd t0, 0(x30)                # through the loaded LINEAR capability
  LDC(x31, a6, 0)              # the cleared granule: an untagged zero word
#ifndef BOARD_TEST_CH
  CAPPRINT(x31)                # 0
#endif
  MOVC(ra, s3)
  la a1, stub0
  RETURN(ra, a1, x0)
stub0_end:

.align 4
_end_of_text:

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 12
savearea:
  .zero 1024
savearea_end:
.align 12
scratch2:
  .zero 64
scratch2_end:
.align 12
wpage:
  .zero 4096
wpage_end:
.align 12
pt_root:
  .zero 4096
.align 12
pt_l1:
  .zero 4096
.align 12
pt_l0:
  .zero 4096
pt_end:
.align 6
seal0:
  .zero 2048
seal0_end:
#ifdef VB_TABLE
.align 12
tbl:
  .zero 4096
tbl_end:
#endif
  TEST_DATA
RVTEST_DATA_END
