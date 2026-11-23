# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# r79-pmp.S -- R-79 on silicon: does the PMP judge an S-mode load or store with its OWN access type? (2026-11-23)
#-----------------------------------------------------------------------------
#
# R-79 (docs/ref/ISSUES.md): pmp_data_if judged each access with the NEIGHBOURING cycle's access type. A lone store
# was judged as a read and a load followed at once by an independent store as a store. Plain RISC-V only: no
# capability instruction runs in S, so the image runs on any bitstream. It derives from capstone-ariane
# csmint-pmp.S (b5b1-step3), without the CSMINT arms.
# M sets PMP entry 0 = NAPOT 4 KiB over r_page, R only, and entry 1 = NAPOT over everything, RWX, then mrets to S bare
# (satp 0). Each S trap returns to M, which records mcause and mtval and resumes S at mepc + 4. S stores every reading
# to res[]; M prints res[] (the recorder) and then the trap count.
#   p1  sd to w_page (the control: RWX)
#   p2  ld r_page, followed AT ONCE by an independent store (the R-79 load arm: a refused LEGAL read, cause 7)
#   p3  a lone sd 0x1234 to r_page, then nops (the R-79 store arm: the write lands)
#   p4  ld r_page (shows whether p3 landed)
#   p5  sd to r_page followed AT ONCE by an independent sd to w_page (judged as a store either way: 7)
#   p6  lr.d of r_page, then nops (the semantics owner's 2026-11-23 ruling: on the fixed RTL LR takes the store/AMO
#       judgement as the MMU always did, so 7; the twin's LR needs R only: a recorded divergence, this its control)
# Registers the test writes, both spellings (the recorder's scratch is x8, x4, x2, none of which the test uses):
#   s2/x18 r_page, s8/x24 w_page, s9/x25 res (each once, in M); s1/x9 cause, s3/x19 tval (the handler, and CLR);
#   a7/x17 the trap count (the handler); t0/x5, t1/x6, t5/x30, t6/x31 temporaries.

#include "riscv_test.h"
#include "test_macros.h"

#define REC(k, r) sd r, (8*(k))(s9)
#define CLR       li s1, 0; li s3, 0
#define GAP       nop; nop; nop; nop; nop; nop; nop; nop

.option norvc
.section .text.init
.globl _start
_start:
la t0, trap_handler
csrw mtvec, t0
csrw mcause, x0
li a7, 0
li t0, 0x5a5a5a5a5a5a5a5a
lla t1, r_page
sd t0, 0(t1)
sd t0, 8(t1)
sd t0, 16(t1)
lla t1, w_page
sd t0, 0(t1)
sd t0, 8(t1)
# PMP: entry 0 NAPOT 4 KiB over r_page, R only; entry 1 NAPOT over everything, RWX (entry 0 matches first)
lla t0, r_page
srli t0, t0, 2
ori t0, t0, 0x1ff
csrw pmpaddr0, t0
li t0, -1
csrw pmpaddr1, t0
li t0, 0x1f19                  # cfg1 = NAPOT | RWX, cfg0 = NAPOT | R
csrw pmpcfg0, t0
csrw satp, x0
sfence.vma
lla s8, w_page
lla s9, res
lla s2, r_page
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
lla t1, res
li t5, 0
2:
slli t0, t5, 3
add t0, t0, t1
ld t0, 0(t0)
CAPPRINT(t0)
addi t5, t5, 1
li t0, 16
bne t5, t0, 2b
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

.align 4
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
# p5: a store to r_page, an independent store right behind it
li t0, 0x99
sd t0, 8(s2)
sd t0, 8(s8)
GAP
REC(7, s1)
REC(8, s3)
CLR
GAP
ld t0, 8(s2)
GAP
REC(9, t0)
GAP
# p6: lr.d of r_page, then nothing
lr.d t0, (s2)
GAP
REC(10, s1)
REC(11, s3)
CLR
GAP
li t0, 0x77
REC(12, t0); REC(13, t0); REC(14, t0); REC(15, t0)
ecall
1: j 1b

.align 12
r_page:
.zero 4096
w_page:
.zero 4096

.align 4
_end_of_text:

RVTEST_CODE_END
  .data
RVTEST_DATA_BEGIN
  TEST_DATA
.align 6
res:
.zero 128
RVTEST_DATA_END
