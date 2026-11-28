# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# csmint-pmp.S -- B5b-1 step 3: does the PMP judge a lone S-mode store, and CSMINT's probe, as a STORE? (2026-11-23)
#-----------------------------------------------------------------------------
#
# The claim-auditor's item 2(b), read in the source and not yet run: pmp_data_if.sv takes the access type from
# lsu_is_store_i = the CURRENT cycle's st_translation_req (load_store_unit.sv:702), while the request it judges is the
# PREVIOUS cycle's (lsu_valid_i = pmp_translation_valid, the MMU's registered request). A store not followed at once by
# another store would then be checked as a READ. CSMINT's probe is always such a store. The wiring is upstream CVA6's
# (23355d29f, PR #2528), so a confirmed defect is older than CSMINT and reaches every lone store.
# M sets PMP entry 0 = NAPOT over r_page, R only, and entry 1 = NAPOT over everything, RWX; then mret to S, bare (satp 0).
# Each S trap returns to M, which records mcause and mtval and resumes S at mepc + 4. S stores the readings to res[];
# M prints them.
#   p1  sd to w_page (the control: RWX)               cause 0
#   p2  ld from r_page (R allowed)                    cause 0, the fill
#   p3  sd 0x1234 to r_page                           IF THE PMP IS RIGHT: 7, tval r_page.  IF 2(b) HOLDS: 0
#   p4  r_page's first word after p3                  RIGHT: the fill.  2(b): 0x1234
#   p5  csmint with slot = r_page + 16                RIGHT: 7, tval r_page + 16 (the reference: probe_access_full of a
#                                                     store raises the store access fault).  2(b): 0, rd 3 (a mint)
#   p6  r_page + 16's second word after p5            RIGHT: the fill.  2(b): 6 (3 << 1, the granule's hi)
#   p7  csmint with slot on w_page                    0, rd 3 if p5 refused, else 4 (a mint where PMP allows)
#   then the pool's allocations word (1 + the mints), the trap count
# Registers the test writes, both spellings: s4/x20 obj, s5/x21 obj + 256, s6/x22 obj + 16, s7/x23 desc, s8/x24 w_page,
# s9/x25 res, s2/x18 r_page (each once, in M); s1/x9 cause and s3/x19 tval (the handler and CLR); a7/x17 traps (the
# handler); a0/x10 rd; t0-t2/x5-x7 temporaries only.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSMINT(rd, rs1, rs2)  .insn r 0x5B, 0x1, 0x50, rd, rs1, rs2
#define REC(k, r) sd r, (8*(k))(s9)
#define CLR       li s1, 0; li s3, 0
#define DR(i, r)  sd r, (8*(i))(s7)
#define DW(i, v)  li t0, v; sd t0, (8*(i))(s7)
#define VALID DR(0, s4); DR(1, s5); DW(2, 0x10f); DR(3, s6); DW(4, 0); DW(5, 0)

.option norvc
.section .text.init
.globl _start
_start:
li gp, 1
la t0, trap_handler
csrw mtvec, t0
csrw mcause, x0
li a7, 0

# fills
li t0, 0x5a5a5a5a5a5a5a5a
lla t1, r_page
sd t0, 0(t1)
sd t0, 8(t1)
sd t0, 16(t1)
sd t0, 24(t1)
lla t1, w_page
sd t0, 0(t1)
sd t0, 8(t1)

# PMP: entry 0 NAPOT 4 KiB over r_page, R only; entry 1 NAPOT over everything, RWX (entry 0 matches first). The first cut
# made entry 1 TOR, whose bottom is the PREVIOUS entry's address: S could fetch nothing below r_page, and every S
# instruction took an instruction access fault (that run was void, no reading)
lla t0, r_page
srli t0, t0, 2
ori t0, t0, 0x1ff
csrw pmpaddr0, t0
li t0, -1
csrw pmpaddr1, t0
li t0, 0x1f19                  # cfg1 = NAPOT | RWX (0x1f), cfg0 = NAPOT | R (0x19)
csrw pmpcfg0, t0
csrw satp, x0
sfence.vma

lla s4, obj
addi s5, s4, 256
addi s6, s4, 16
lla s7, desc
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
li t2, 0
2:
slli t0, t2, 3
add t0, t0, t1
ld t0, 0(t0)
CAPPRINT(t0)
addi t2, t2, 1
li t0, 14
bne t2, t0, 2b
li t1, 0xbff00000
ld t0, 24(t1); CAPPRINT(t0)
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
# p1: the control store
li t0, 0x77
sd t0, 0(s8)
REC(0, s1)
CLR
# p2: a read of r_page
ld t0, 0(s2)
REC(1, s1); REC(2, t0)
CLR
# p3: a lone store to r_page
li t0, 0x1234
sd t0, 0(s2)
REC(3, s1); REC(4, s3)
CLR
# p4
ld t0, 0(s2); REC(5, t0)
# p5: csmint with the slot on r_page
VALID
addi t1, s2, 16
CSMINT(a0, t1, s7)
REC(6, s1); REC(7, s3)
CLR
# p6
ld t0, 24(s2); REC(8, t0)
# p7: csmint with the slot on w_page
VALID
CSMINT(a0, s8, s7)
REC(9, s1); REC(10, a0)
CLR
li t0, 0x77
REC(11, t0); REC(12, t0); REC(13, t0)
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
.align 8
obj:
.zero 256
.align 6
desc:
.zero 64
.align 6
res:
.zero 128
RVTEST_DATA_END
