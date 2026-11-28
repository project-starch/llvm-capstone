# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# r79-cpmp.S -- R-79's CPMP side: does the CPMP judge an S-mode access with its OWN access type? (2026-11-24)
#-----------------------------------------------------------------------------
#
# R-79 (llvm-capstone docs/ref/ISSUES.md): pmp_data_if judged each access with the NEIGHBOURING cycle's access type
# (lsu_is_store_i was the current cycle's st_translation_req while the request judged was the previous cycle's). The
# CPMP's permission pick reads the same input (pmp_data_if.sv cpmp_perm_ok: perm[1] for a store, perm[2] for a read).
# The PMP side is fixed on silicon (the B5b-1 seal); the CPMP side needs capability mode, so this arm runs on the IDS55
# build of the same tree (XDEF +CAPSTONE_IDS55_FMT), where CAPENTER is legal and the CPMP trackers are live.
# The S arms copy r79-pmp.S's instruction sequences, because R-79's outcome depends on adjacency: a following store
# that depends on the load's value is not adjacent and does not trigger it (csmint-pmp p4).
# M fills the pages BEFORE CAPENTER (after it, an M access through an integer base is refused). Then CAPENTER, PMP
# entry 0 RWX over everything, and four NONLIN CPMP entries, each a capability over one region:
#   CPMP 0  r_page   perm 4 (RO)     CPMP 1  wo_page  perm 2 (WO)
#   CPMP 2  w_page   perm 6 (RW, res lives here)      CPMP 3  s_code's page, perm 1 (XO, S's fetch)
# mret to S, bare (satp 0). Each S trap returns to M, which records mcause and mtval in s1/s3 without touching memory
# and resumes S at mepc + 4. At the ecall M mints a capability over each region it reads and prints through it:
# res[0..16], then wo_page + 8, wo_page + 16, r_page + 8, then the trap count.
#   p1  sd to w_page (the control)
#   p2  ld r_page, an independent store right behind it         (fixed 0, the fill; pre-fix 7, judged a store)
#   p3  a lone sd 0x1234 to r_page                              (fixed 7; pre-fix 0, judged a read: the write lands)
#   p4  ld r_page                                               (whether p3 landed)
#   p5  lr.d of r_page, then nothing                            (fixed 7, an AMO is a write; pre-fix 0)
#   p6  ld wo_page, an independent store right behind it        (fixed 5; pre-fix 0 and the fill: a READ through a
#                                                                W-only entry)
#   p7  a lone sd 0x4321 to wo_page + 8                         (fixed 0; pre-fix 5: a legal store refused as a read)
#   p8  sd wo_page + 16, an independent sd behind it            (store then store: 0 on both trees)
#   p9  sd r_page + 8, an independent sd behind it              (store then store: 7 on both trees)
# Registers the test writes, both spellings: s2/x18 r_page, s4/x20 wo_page, s8/x24 w_page, s9/x25 res (each once, in
# M); s11/x27 the readout capabilities (M, at the end); s10/x26 CAP_PASS's; s1/x9 cause, s3/x19 tval (the handler, and CLR); a7/x17 the trap
# count (the handler); a0-a4/x10-x14 the mint's operands (M only); t0/x5, t5/x30, t6/x31 temporaries.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define REC(k, r) sd r, (8*(k))(s9)
#define CLR       li s1, 0; li s3, 0
#define GAP       nop; nop; nop; nop; nop; nop; nop; nop
#define MINT(creg, ty, lo, hi, pm) \
    CAPCREATE(creg); li a0, ty; lla a2, lo; lla a3, hi; li a4, pm; \
    CAPTYPE(creg, a0); CAPBOUND(creg, a2, a3); CAPPERM(creg, a4)
#define SHOW(off) ld t0, (off)(s11); CAPPRINT(t0)

.option norvc
.section .text.init
.globl _start
_start:
lla t0, trap_handler
csrw mtvec, t0
csrw mcause, x0
li a7, 0

# fills, before CAPENTER
li t0, 0x5a5a5a5a5a5a5a5a
lla t5, r_page
sd t0, 0(t5)
sd t0, 8(t5)
li t0, 0xa5a5a5a5a5a5a5a5
lla t5, wo_page
sd t0, 0(t5)
sd t0, 8(t5)
sd t0, 16(t5)
lla t5, res
li t6, 0
2:
sd x0, 0(t5)
addi t5, t5, 8
addi t6, t6, 1
li t0, 24
bne t6, t0, 2b

lla a0, _start
lla a1, _end_of_text
CAPENTER(a0, a1)

li t0, 0x001fffffffffffff
csrw pmpaddr0, t0
li t0, 0x1f
csrw pmpcfg0, t0
MINT(t5, CAP_TYPE_NONLIN, r_page, wo_page, CAP_PERM_RO)
CCSRRW(x0, CCSR_CPMP(0), t5)
MINT(t5, CAP_TYPE_NONLIN, wo_page, w_page, CAP_PERM_WO)
CCSRRW(x0, CCSR_CPMP(1), t5)
MINT(t5, CAP_TYPE_NONLIN, w_page, w_page_end, CAP_PERM_RW)
CCSRRW(x0, CCSR_CPMP(2), t5)
MINT(t5, CAP_TYPE_NONLIN, s_code, r_page, CAP_PERM_XO)
CCSRRW(x0, CCSR_CPMP(3), t5)
csrw satp, x0
sfence.vma

lla s2, r_page
lla s4, wo_page
lla s8, w_page
lla s9, res
li t0, 0x1800
csrc mstatus, t0
li t0, 0x0800
csrs mstatus, t0               # MPP = S
lla t0, s_code
csrw mepc, t0
li s1, 0
li s3, 0
mret

m_done:
MINT(s11, CAP_TYPE_LIN, res, w_page_end, CAP_PERM_RWX)
SHOW(0)
SHOW(8)
SHOW(16)
SHOW(24)
SHOW(32)
SHOW(40)
SHOW(48)
SHOW(56)
SHOW(64)
SHOW(72)
SHOW(80)
SHOW(88)
SHOW(96)
SHOW(104)
SHOW(112)
SHOW(120)
SHOW(128)
MINT(s11, CAP_TYPE_LIN, wo_page, w_page, CAP_PERM_RWX)
SHOW(8)
SHOW(16)
MINT(s11, CAP_TYPE_LIN, r_page, wo_page, CAP_PERM_RWX)
SHOW(8)
CAPPRINT(a7)
CAP_PASS(s10)

.align 4
trap_handler:
csrr t6, mcause
li t5, 9
beq t6, t5, m_done
addi a7, a7, 1
mv s1, t6
csrr s3, mtval
csrr t6, mepc
addi t6, t6, 4
csrw mepc, t6
mret

.align 12
s_code:
# p1
li t0, 0x77
sd t0, 0(s8)
GAP
REC(0, s1)
CLR
GAP
# p2: a legal read of r_page, an independent store right behind it
ld t0, 0(s2)
REC(1, s1)
GAP
REC(2, s3)
REC(3, t0)
CLR
GAP
# p3: a lone store to r_page
li t0, 0x1234
sd t0, 0(s2)
GAP
REC(4, s1)
REC(5, s3)
CLR
GAP
# p4
ld t0, 0(s2)
GAP
REC(6, t0)
GAP
# p5: lr.d of r_page, then nothing
lr.d t0, (s2)
GAP
REC(7, s1)
REC(8, s3)
CLR
GAP
# p6: a read of wo_page, an independent store right behind it
li t0, 0x77
ld t0, 0(s4)
REC(9, s1)
GAP
REC(10, s3)
REC(11, t0)
CLR
GAP
# p7: a lone store to wo_page + 8
li t0, 0x4321
sd t0, 8(s4)
GAP
REC(12, s1)
REC(13, s3)
CLR
GAP
# p8: a store to wo_page + 16, an independent store right behind it
li t0, 0x99
sd t0, 16(s4)
sd t0, 16(s8)
GAP
REC(14, s1)
CLR
GAP
# p9: a store to r_page + 8, an independent store right behind it
li t0, 0x99
sd t0, 8(s2)
sd t0, 8(s8)
GAP
REC(15, s1)
REC(16, s3)
CLR
GAP
ecall
1: j 1b
s_code_end:

.align 12
r_page:
.zero 4096
wo_page:
.zero 4096
w_page:
.zero 2048
res:
.zero 2048
w_page_end:

.align 4
_end_of_text:

RVTEST_CODE_END
  .data
RVTEST_DATA_BEGIN
  TEST_DATA
RVTEST_DATA_END
