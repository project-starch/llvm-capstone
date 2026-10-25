# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# vb-trans.S  (item B of the virtual profile on silicon: capability accesses TRANSLATED inside a supervised domain, 2026-10-16)
#-----------------------------------------------------------------------------
#
# The monitor builds Sv39 page tables in its own data (an identity map of the first 2 MiB of DRAM in 4 KiB USER pages,
# the rest of the first GiB in 2 MiB USER megapages), points satp at them, sets csupctl.virt (bit 1) and arms a
# supervised domain. Under that flag the domain's data and fetch accesses translate with USER PTE permissions while the
# privilege stays M, the capability check stays armed (the flag is a new term in the translation enables, never a change
# to ld_st_priv_lvl), and a page fault escapes as a FAULT the monitor can resume. Six arms on the sup-sublet.S pattern
# (one seal and stub each). The domain inherits in GPRs: x26 NONLIN RW+MANAGE over `data`, s1 LINEAR RW scratch, x27
# NONLIN RW over the `shiftva` page (mapped to `shiftpa`), s4 NONLIN RW over the `nomap` page (unmapped), a6 NONLIN RW
# over the `ropage` page (mapped read-only). Register choice: every GPR the domain inherits must be one no monitor-side
# temporary aliases (s10 IS x26, t3/t4 ARE x28/x29: the first cut clobbered three capabilities with its page-table loops).
#
# Pre-registered readings (CAPPRINT values in the retirement trace, in order), flag ON (the default build):
#   arm 0 (identity): status 0 (armed); csupstatus 1 (RETURN); the two words the domain stored through s1: 0x77 (stored and
#     loaded back through x26 under translation), 0x1234 (what the monitor had put at data+8)
#   arm 1 (va != pa): status 0; csupstatus 1; the monitor reads the PHYSICAL `shiftpa` page: 0x5a5a; the identity location
#     `shiftva`: 0. The store went where the page table said. (Flag OFF, or an RTL without the flag: 0 then 0x5a5a.)
#   arm 2 (page fault, resumed): status 0; csupstatus 5 (fault); csupcause 13 (load page fault); csupepc - fault2 = 0;
#     csuptval - nomap = 0; after the monitor stores 0x99 into the page, maps it, fences and resumes (csupctl 3):
#     csupstatus 1; the word the domain stored through s1: 0x99
#   arm 3 (read-only page, plain store): status 0; 5; csupcause 15 (store page fault); csupepc - fault3 = 0; csuptval - ropage = 0
#   arm 4 (consuming ldc from a read-only page): status 0; 5; csupcause 15 (QEMU's rule, the Q-19 ruling: the decision
#     is at writeback once the loaded value is known to be consuming; the page's store permission was captured with the
#     clear address at the read's translation, the clear is never queued and the ldc retires with STORE_PAGE_FAULT,
#     carried to the capability writeback port by the R-56 merge); csupepc - fault4 = 0; after the monitor makes the
#     page writable and resumes:
#     the domain's own CAPPRINT of the reloaded granule reads 0 (the source WAS cleared by the resumed ldc; the first
#     cut read it with lcc, which does not trap on an untagged operand and returned 7), then csupstatus 1 (RETURN)
#   arm 5 (bounds under translation): status 0; 5; csupcause 28 (a store one page past x26's end: the capability check is
#     armed with translation on); csupepc - fault5 = 0
#   arm 6 (NON-consuming ldc from a read-only page through an RW capability, the case Q-19's ruling keeps): status 0;
#     inside the domain two CAPPRINTs, the loaded capability and x26 it was copied from, EQUAL and non-zero (the load
#     completes, nothing is cleared); csupstatus 1 (RETURN)
#   mcause 0 (no trap was taken anywhere)
# Flag OFF (-DVB_FLAG_OFF: csupctl 0, satp bare): arms 0 and 1 only: 0, 1, [cause 0, epc-stub0, tval: diagnostic], 0x77, 0x1234,
#   0, 1, [cause 0], 0, 0x5a5a, mcause 0. The bracketed readouts were added after the first run and are not pre-registered.
# A trap anywhere ends the test with the trap handler's mcause as the last reading.
# On the board (BOARD_TEST_CH defined by the bare wrapper) the three in-domain CAPPRINTs compile out (the wrapper's
# CAPPRINT mints, and a mint inside a supervised domain is refused), so the board vector is the monitor-side readings:
# 36 values for the flag-on build, 13 for the flag-off build.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSUPQUANTUM 0x7C3
#define CSUPCTL     0x7C4
#define CSUPSTATUS  0xFC0
#define CSUPCAUSE   0xFC1
#define CSUPEPC     0xFC2
#define CSUPTVAL    0xFC3

#ifdef VB_FLAG_OFF
#define VB_CTL_ARM    0
#define VB_CTL_RESUME 1
#else
#define VB_CTL_ARM    2
#define VB_CTL_RESUME 3
#endif

#define PTE_LEAF_RWX  0xDF   /* V R W X U A D */
#define PTE_LEAF_RO   0xD3   /* V R   U A D */
#define PTE_PTR       0x01   /* V */

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

MINT_NONLIN_RW(x26, data, data_end, 14)        # RW + MANAGE
li t0, 0x1234
sd t0, 8(x26)
sd x0, 0(x26)
MINT_NONLIN_RW(x27, shiftva, shiftva_end, CAP_PERM_RW)
MINT_NONLIN_RW(s4, nomap, nomap_end, CAP_PERM_RW)
MINT_NONLIN_RW(a6, ropage, ropage_end, CAP_PERM_RW)
# the monitor's own capabilities: s7 over the physical `shiftpa` page, s8 over the three page tables (every monitor
# access goes through a capability: after CAPENTER an integer-addressed load or store raises 24)
MINT_NONLIN_RW(s7, shiftpa, shiftpa_end, CAP_PERM_RW)
MINT_NONLIN_RW(s8, pt_root, pt_end, CAP_PERM_RW)
MINT_NONLIN_RW(t5, ropage2, ropage2_end, CAP_PERM_RW)   # arm 6: x26 parked here, the page mapped read-only
STC(t5, x26, 0)

# s1: a LINEAR RW scratch the domain stores its readings through
CAPCREATE(s1)
li a0, CAP_TYPE_LIN
lla a3, scratch
lla a4, scratch_end
li a5, CAP_PERM_RW
CAPTYPE(s1, a0)
CAPBOUND(s1, a3, a4)
CAPPERM(s1, a5)
sd x0, 0(s1)
sd x0, 8(s1)

# a LINEAR RW capability parked in the read-only page for arm 4's consuming ldc (stored while the monitor is bare)
CAPCREATE(s6)
li a0, CAP_TYPE_LIN
lla a3, scratch2
lla a4, scratch2_end
li a5, CAP_PERM_RW
CAPTYPE(s6, a0)
CAPBOUND(s6, a3, a4)
CAPPERM(s6, a5)
STC(a6, s6, 0)

#ifndef VB_FLAG_OFF
# ---------------------------------------------------------------- the page tables (Sv39, physical, built bare)
# L0: 512 x 4 KiB identity leaves over [0x80000000, 0x80200000)
lla t0, pt_l0
lla t1, pt_root
sub a7, t0, t1                 # a7: L0's offset inside s8 (s9/s10 are x25/x26: never temporaries here)
lla t0, pt_l1
sub s9, t0, t1                 # s9 (x25): L1's offset inside s8 (t5 holds arm 6's capability)
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
# L0[shiftva] -> shiftpa. L0_SLOT leaves in REG a capability at the L0 entry of LABEL's page
#define L0_SLOT(LABEL, REG) \
  lla REG, LABEL; li t1, 0x80000000; sub REG, REG, t1; srli REG, REG, 12; slli REG, REG, 3; add t1, REG, a7; CINCOFFSET(REG, s8, t1)
L0_SLOT(shiftva, t0)
lla t2, shiftpa
srli t2, t2, 12
slli t2, t2, 10
ori t2, t2, PTE_LEAF_RWX
sd t2, 0(t0)
# L0[nomap] -> invalid
L0_SLOT(nomap, t0)
sd x0, 0(t0)
# L0[ropage] -> read-only
L0_SLOT(ropage, t0)
lla t2, ropage
srli t2, t2, 12
slli t2, t2, 10
ori t2, t2, PTE_LEAF_RO
sd t2, 0(t0)
# L0[ropage2] -> read-only (arm 6)
L0_SLOT(ropage2, t0)
lla t2, ropage2
srli t2, t2, 12
slli t2, t2, 10
ori t2, t2, PTE_LEAF_RO
sd t2, 0(t0)
# L1: [0] -> L0; [1..511] 2 MiB identity megapages
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
# root[2] -> L1 (va[38:30] = 2 for 0x80000000)
lla t1, pt_l1
srli t1, t1, 12
slli t1, t1, 10
ori t1, t1, PTE_PTR
li t4, 16
CINCOFFSET(t0, s8, t4)
sd t1, 0(t0)
# the walker's table reads are PMP-checked as S-mode: one all-permitting NAPOT entry, as firmware programs before S/U runs
li t0, -1
csrw pmpaddr0, t0
li t0, 0x1f
csrw pmpcfg0, t0
# satp = SV39 | root ppn
lla t0, pt_root
srli t0, t0, 12
li t1, 8
slli t1, t1, 60
or t0, t0, t1
csrw satp, t0
sfence.vma
#endif

li t0, 0x100000
csrw CSUPQUANTUM, t0           # a quantum far longer than any arm
csrwi CSUPCTL, VB_CTL_ARM      # first entries; bit 1 = the virtual context

#define ARM_SETUP(STUB, STUB_END, SEALX, SEALX_END) \
  CAPCREATE(a2); li a0, CAP_TYPE_LIN; lla a3, STUB; lla a4, STUB_END; li a5, CAP_PERM_RWX; \
  CAPTYPE(a2, a0); CAPBOUND(a2, a3, a4); CAPPERM(a2, a5); \
  CAPCREATE(a1); li a0, CAP_TYPE_LIN; lla a3, SEALX; lla a4, SEALX_END; li a5, CAP_PERM_RWX; \
  CAPTYPE(a1, a0); CAPBOUND(a1, a3, a4); CAPPERM(a1, a5); \
  STC(a1, a2, 0); li a0, 0x0000000A00040000; sd a0, 48(a1); sd x0, 56(a1); sd x0, 64(a1); sd x0, 72(a1); sd x0, 80(a1); \
  SEAL(a3, a1); csrw mcause, x0; CSSUPERVISE(t3, a3, s5); addi s2, t3, 0; CAPPRINT(s2); CALL(a3, a3)

#define ARM_FAULT_READOUT(FAULTK) \
  csrr t0, CSUPSTATUS; CAPPRINT(t0); csrr t0, CSUPCAUSE; CAPPRINT(t0); \
  csrr t0, CSUPEPC; la t1, FAULTK; sub t0, t0, t1; CAPPRINT(t0)

#define ARM_TVAL_READOUT(PAGE) \
  csrr t0, CSUPTVAL; la t1, PAGE; sub t0, t0, t1; CAPPRINT(t0)

#define ARM_RESUME \
  csrwi CSUPCTL, VB_CTL_RESUME; CSSUPERVISE(t3, a3, s5); CALL(a3, a3)

# ---------------------------------------------------------------- arm 0: identity-mapped data through x26
ARM_SETUP(stub0, stub0_end, seal0, seal0_end)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 1
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 0 (diagnostic until the first green run: on a fault, the cause)
csrr t0, CSUPEPC
la t1, stub0
sub t0, t0, t1
CAPPRINT(t0)                   # diagnostic: csupepc - stub0
csrr t0, CSUPTVAL
CAPPRINT(t0)                   # diagnostic: csuptval
ld t0, 0(s1)
CAPPRINT(t0)                   # 0x77
ld t0, 8(s1)
CAPPRINT(t0)                   # 0x1234

# ---------------------------------------------------------------- arm 1: a store through x27 lands at the PHYSICAL page
ARM_SETUP(stub1, stub1_end, seal1, seal1_end)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 1
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 0 (diagnostic, as in arm 0)
ld t0, 0(s7)
CAPPRINT(t0)                   # 0x5a5a (flag on) / 0 (flag off)
ld t0, 0(x27)
CAPPRINT(t0)                   # 0 (flag on) / 0x5a5a (flag off)

#ifndef VB_FLAG_OFF
# ---------------------------------------------------------------- arm 2: a load from an unmapped page faults 13 and resumes
ARM_SETUP(stub2, stub2_end, seal2, seal2_end)
ARM_FAULT_READOUT(fault2)      # 5, 13, 0
ARM_TVAL_READOUT(nomap)        # 0
li t0, 0x99
sd t0, 0(s4)                   # bare (the monitor is not in the virtual context): the physical page
L0_SLOT(nomap, t0)
lla t2, nomap
srli t2, t2, 12
slli t2, t2, 10
ori t2, t2, PTE_LEAF_RWX
sd t2, 0(t0)                   # map it
sfence.vma
ARM_RESUME
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 1
ld t0, 0(s1)
CAPPRINT(t0)                   # 0x99

# ---------------------------------------------------------------- arm 3: a plain store to a read-only page faults 15
ARM_SETUP(stub3, stub3_end, seal3, seal3_end)
ARM_FAULT_READOUT(fault3)      # 5, 15, 0
ARM_TVAL_READOUT(ropage)       # 0

# ---------------------------------------------------------------- arm 4: a consuming ldc from a read-only page is refused 15; writable, it clears
ARM_SETUP(stub4, stub4_end, seal4, seal4_end)
ARM_FAULT_READOUT(fault4)      # 5, 15, 0
L0_SLOT(ropage, t0)
lla t2, ropage
srli t2, t2, 12
slli t2, t2, 10
ori t2, t2, PTE_LEAF_RWX
sd t2, 0(t0)                   # writable now
sfence.vma
ARM_RESUME
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 1: the resumed ldc cleared its source and the domain returned

# ---------------------------------------------------------------- arm 5: the capability bounds check is armed under translation
ARM_SETUP(stub5, stub5_end, seal5, seal5_end)
ARM_FAULT_READOUT(fault5)      # 5, 28, 0

# ---------------------------------------------------------------- arm 6: a non-consuming ldc from a read-only page completes
ARM_SETUP(stub6, stub6_end, seal6, seal6_end)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 1
#endif

csrr t0, mcause
CAPPRINT(t0)                   # 0
CAP_PASS(s11)

trap_handler:
  csrr t0, mcause
  CAPPRINT(t0)
  CAP_PASS(s11)

# ---- the domains. No mint, no CSR, no CALL.
.align 6
stub0:
  MOVC(s3, x1)
  li t0, 0x77
  sd t0, 0(x26)
  ld t1, 0(x26)
  sd t1, 0(s1)
  ld t1, 8(x26)
  sd t1, 8(s1)
  MOVC(ra, s3)
  la a1, stub0
  RETURN(ra, a1, x0)
stub0_end:

.align 6
stub1:
  MOVC(s3, x1)
  li t0, 0x5a5a
  sd t0, 0(x27)
  MOVC(ra, s3)
  la a1, stub1
  RETURN(ra, a1, x0)
stub1_end:

.align 6
stub2:
  MOVC(s3, x1)
fault2:
  ld t2, 0(s4)
  sd t2, 0(s1)
  MOVC(ra, s3)
  la a1, stub2
  RETURN(ra, a1, x0)
stub2_end:

.align 6
stub3:
  li t0, 0x33
fault3:
  sd t0, 0(a6)
  1: j 1b                      # never reached
stub3_end:

.align 6
stub4:
  MOVC(s3, x1)
fault4:
  LDC(x30, a6, 0)              # consuming: the loaded value is LINEAR, the clear needs the page's W bit
  LDC(x31, a6, 0)              # after the resume: the granule was cleared, so this reads an untagged zero word
#ifndef BOARD_TEST_CH
  CAPPRINT(x31)                # 0 (cleared) against 0x80003440, the parked capability's cursor, if the clear did not land
#endif
  MOVC(ra, s3)
  la a1, stub4
  RETURN(ra, a1, x0)
stub4_end:

.align 6
stub5:
  li t0, 0x1000
  CINCOFFSET(t1, x26, t0)
fault5:
  sd x0, 0(t1)                 # one page past the end of x26: 28
  1: j 1b                      # never reached
stub5_end:

.align 6
stub6:
  MOVC(s3, x1)
  LDC(x31, t5, 0)              # NONLIN value: nothing to clear, the page's W bit is not needed
#ifndef BOARD_TEST_CH
  CAPPRINT(x31)                # the parked copy of x26: equal to the next reading
  CAPPRINT(x26)
#endif
  MOVC(ra, s3)
  la a1, stub6
  RETURN(ra, a1, x0)
stub6_end:

.align 4
_end_of_text:

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 6
savearea:
  .zero 1024
savearea_end:
.align 6
scratch:
  .zero 64
scratch_end:
.align 6
scratch2:
  .zero 64
scratch2_end:
.align 12
data:
  .zero 4096
data_end:
.align 12
shiftva:
  .zero 4096
shiftva_end:
.align 12
shiftpa:
  .zero 4096
shiftpa_end:
.align 12
nomap:
  .zero 4096
nomap_end:
.align 12
ropage:
  .zero 4096
ropage_end:
.align 12
ropage2:
  .zero 4096
ropage2_end:
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
.align 6
seal1:
  .zero 2048
seal1_end:
.align 6
seal2:
  .zero 2048
seal2_end:
.align 6
seal3:
  .zero 2048
seal3_end:
.align 6
seal4:
  .zero 2048
seal4_end:
.align 6
seal5:
  .zero 2048
seal5_end:
.align 6
seal6:
  .zero 2048
seal6_end:
  TEST_DATA
RVTEST_DATA_END
