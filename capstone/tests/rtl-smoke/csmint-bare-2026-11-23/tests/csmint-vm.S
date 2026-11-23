# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# csmint-vm.S -- B5b-1 step 3: CSMINT in S-mode under Sv39, the probe's and the descriptor loads' page faults (2026-11-23)
#-----------------------------------------------------------------------------
#
# M builds an identity Sv39 map of the image's first 2 MiB in 4 KiB pages (V R W X U A D), with three pages patched:
# s_page (the S code: no U), ro_page (V R U A D: no W) and nod_page (V R W U A: no D, so a store faults; A/D are
# software's on CVA6). VA 0x40000000 has no mapping. The record goes to the physical pool, as in csmint.S (no srevroot).
# M enters S at s_code with sstatus.SUM set. S runs the arms; each trap returns to M, which records mcause and mtval and
# resumes S at mepc + 4. S stores every reading to res[] and ecalls, and M prints res[] (prints come from M).
#   v1  a valid mint, desc and slot on U pages        0, rd 3, slot hi (= 3 << 1)
#   v2  slot on ro_page (no W)                        15, tval slot; ro_page's first word unchanged
#   v3  slot on nod_page (W, no D)                    15, tval slot (the probe checks W && D); unchanged
#   v4  desc at VA 0x40000000 (unmapped)              13, tval 0x40000000
#   v5  slot at VA 0x40000000 (unmapped)              15, tval 0x40000000
#   v6  slot2 on ro_page, slot valid                  15, tval slot2; slot unchanged
#   v7  SUM cleared: desc on a U page from S          13, tval desc
#   v8  SUM set again: a valid mint                   0, rd 4 (nothing above allocated a record)
#   then M prints the pool's header (q0, q1): 0x8000000000010000, next 5, free 0, allocations 3
# Registers the test WRITES, both spellings (the binding pre-registration list; one write each for value holders):
#   s4/x20 obj (once), s5/x21 obj + 256 (once), s6/x22 obj + 16 (once), s7/x23 desc (once), s8/x24 area (once),
#   s9/x25 res (once), s1/x9 cause and s3/x19 tval (the handler, and CLR between arms), a7/x17 trap count (handler),
#   a0/x10 rd, t0-t6/x5-x7,x28-x31 temporaries (never held across an arm), a1-a5/x11-x15 setup only.
# PRE-REGISTERED in csmint-vm.expect before any run (the reference's order and code, as csmint.expect).

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSMINT(rd, rs1, rs2)  .insn r 0x5B, 0x1, 0x50, rd, rs1, rs2
#define PTE_UPG    0xDF          /* V R W X U A D */
#define PTE_SPG    0xCF          /* V R W X A D: S code */
#define PTE_ROPG   0xD3          /* V R U A D */
#define PTE_NODPG  0x57          /* V R W U A */
/* S: store a reading (register r) to res[k] */
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

# ---- the page tables: root[2] -> L1, L1[0] -> L0, L0[i] = 0x80000000 + 4 KiB i
lla t5, pt_l0
li t1, 0x80000000
li t2, 0
1:
srli t3, t1, 12
slli t3, t3, 10
ori t3, t3, PTE_UPG
slli t4, t2, 3
add t4, t4, t5
sd t3, 0(t4)
li t4, 0x1000
add t1, t1, t4
addi t2, t2, 1
li t4, 512
bne t2, t4, 1b
lla t1, pt_l0
srli t1, t1, 12
slli t1, t1, 10
ori t1, t1, 1
lla t4, pt_l1
sd t1, 0(t4)                   # L1[0] -> L0
lla t1, pt_l1
srli t1, t1, 12
slli t1, t1, 10
ori t1, t1, 1
lla t4, pt_root
sd t1, 16(t4)                  # root[2] -> L1
#define PATCH(sym, bits) lla t1, sym; li t3, 0x80000000; sub t3, t1, t3; srli t3, t3, 12; slli t3, t3, 3; add t3, t3, t5; \
  srli t1, t1, 12; slli t1, t1, 10; ori t1, t1, bits; sd t1, 0(t3)
PATCH(s_page, PTE_SPG)
PATCH(ro_page, PTE_ROPG)
PATCH(nod_page, PTE_NODPG)
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

# fills the S arms check: ro_page and nod_page and the slots hold the fill
li t0, 0x5a5a5a5a5a5a5a5a
lla t1, ro_page
sd t0, 0(t1)
sd t0, 8(t1)
lla t1, nod_page
sd t0, 0(t1)
sd t0, 8(t1)
lla t1, area
sd t0, 0(t1)
sd t0, 8(t1)

# ---- enter S at s_code: MPP = S, SUM set
lla s4, obj
addi s5, s4, 256
addi s6, s4, 16
lla s7, desc
lla s8, area
lla s9, res
li t0, 0x1800
csrc mstatus, t0
li t0, 0x0800
csrs mstatus, t0               # MPP = 01 (S)
li t0, 0x40000
csrs mstatus, t0               # SUM
lla t0, s_code
csrw mepc, t0
li s1, 0
li s3, 0
mret

# ---- M: the readings, then the pool's header
m_done:
lla t3, res
li t4, 0
2:
slli t5, t4, 3
add t5, t5, t3
ld t2, 0(t5)
CAPPRINT(t2)
addi t4, t4, 1
li t5, 25
bne t4, t5, 2b
li t3, 0xbff00000
ld t2, 0(t3); CAPPRINT(t2)
ld t2, 8(t3); CAPPRINT(t2)
ld t2, 16(t3); CAPPRINT(t2)
ld t2, 24(t3); CAPPRINT(t2)
CAPPRINT(a7)
CAP_PASS(s10)

.align 4
trap_handler:
csrr t6, mcause
li t5, 9
beq t6, t5, m_done             # S's ecall: the arms are done
addi a7, a7, 1
mv s1, t6
csrr s3, mtval
csrr t6, mepc
addi t6, t6, 4
csrw mepc, t6
mret

# ---------------------------------------------------------------- S code, its own page (no U)
.align 12
s_page:
s_code:
# v1: valid
VALID
CSMINT(a0, s8, s7)
REC(0, s1); REC(1, a0)
ld t0, 8(s8); REC(2, t0)
CLR
# v2: slot on ro_page
lla t1, ro_page
CSMINT(a0, t1, s7)
REC(3, s1); REC(4, s3)
lla t1, ro_page; ld t0, 0(t1); REC(5, t0)
CLR
# v3: slot on nod_page
lla t1, nod_page
CSMINT(a0, t1, s7)
REC(6, s1); REC(7, s3)
lla t1, nod_page; ld t0, 0(t1); REC(8, t0)
CLR
# v4: desc unmapped
li t1, 0x40000000
CSMINT(a0, s8, t1)
REC(9, s1); REC(10, s3)
CLR
# v5: slot unmapped
li t1, 0x40000000
CSMINT(a0, t1, s7)
REC(11, s1); REC(12, s3)
CLR
# v6: slot2 on ro_page (fill slot 0 again first)
li t0, 0x5a5a5a5a5a5a5a5a
sd t0, 0(s8)
sd t0, 8(s8)
lla t1, ro_page
DR(4, t1)
CSMINT(a0, s8, s7)
REC(13, s1); REC(14, s3)
ld t0, 0(s8); REC(15, t0)
CLR
DW(4, 0)
# v7: SUM cleared, desc on a U page (S may write sstatus.SUM)
li t0, 0x40000
csrc sstatus, t0
CSMINT(a0, s8, s7)
li t0, 0x40000
csrs sstatus, t0
REC(16, s1); REC(17, s3)
CLR
# v8: valid again
VALID
CSMINT(a0, s8, s7)
REC(18, s1); REC(19, a0)
ld t0, 8(s8); REC(20, t0)
CLR
#ifdef CM_NONCANON
# v9 / v10 (-DCM_NONCANON, the claim-auditor's 2(a)): a NON-CANONICAL desc, then slot, under Sv39 (bit 47 set over a valid
# address: bits [63:38] not all equal). The reference page-faults (13, then 15, tval the address). Without the
# sequencer's overflow (98fad727a) the MMU translated the aliased page and CSMINT minted.
li t1, 0x0000800000000000
or t1, t1, s7
CSMINT(a0, s8, t1)
REC(21, s1); REC(22, s3)
CLR
VALID
li t1, 0x0000800000000000
or t1, t1, s8
CSMINT(a0, t1, s7)
REC(23, s1); REC(24, s3)
CLR
#else
# res[21..24]: markers, so a short run cannot read as complete
li t0, 0x77
REC(21, t0); REC(22, t0); REC(23, t0); REC(24, t0)
#endif
ecall
1: j 1b

.align 12
ro_page:
.zero 4096
nod_page:
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
area:
.zero 64
.align 6
res:
.zero 256
.align 12
pt_root:
.zero 4096
pt_l1:
.zero 4096
pt_l0:
.zero 4096
pt_end:
RVTEST_DATA_END
