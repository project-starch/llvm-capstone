# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# b5b2-mwrite.S -- B5b-2 step 4's arm list: route.S masked_write (an srevroot write keeps only the aligned bits)
#-----------------------------------------------------------------------------
#
# The reference (capstone-qemu 9289114e06, x-capstone-trap-route; tests/trap-route/route.S case 5): srevroot written with
# the table plus bits below its alignment reads the table and protects U; written with those bits alone it reads 0 and
# U is unprotected. The RTL masks at the write (csr_regfile, srevroot_d = {csr_wdata[63:12], 12'b0}) and keys
# protection on srevroot_q != 0 (cap_data_ctx), so this is a test of existing behaviour, not of step 4's change.
# Table and capabilities: b5b2-sref.S's setup (b5b2-uprot.S's table: x27/s11 = {0,4} over data is STALE in tbl, slot 4
# at generation 2^30; x24/s8 = data as an integer; data[0] = 0x77). Traps go to M; the handler records mcause in x9/s1,
# counts in x28/t3, resumes at mepc + 4; ecall returns to M at x30/t5. U holds its readings in a0 and a1 (a store
# through an integer base would raise 24 in protected U); M stores every reading at x25/s9 and prints them
# at the end, after their count.
# Readings:
#   1 csrr srevroot == tbl after csrw srevroot, tbl | 0x7ff
#   U (protected): 2 cause of ld 0(x27) (stale)   3 cause of ld 0(x24) (untagged base)
#   4 csrr srevroot == 0 after csrw srevroot, 0xfff
#   U (unprotected): 5 cause of ld 0(x27)   6 its value
#   7 the trap count
# PRE-REGISTERED (2026-11-24, before any run), the same on step 2b (b2f27b3b3) and step 4 (7c8550cf8):
#   1 0x19 0x18 1 0 0x77 2
# B5b-2 (2026-11-24, the semantics owner's ruling): uses the capcreate family, to be ported to CSMINT.

# LEGACY SCAFFOLD (B5b-2 step 9, 2026-11-25): this test mints with the custom-3 debug family, which decodes under the
# Sublet format only in simulation builds with CAPSTONE_SIM_DEBUG_MINT. -DCAP_MINT_CSMINT (the bare kits, silicon)
# mints through CSMINT instead, which allocates its own lifetime record, so pool ids shift by one there. New tests
# mint with CSMINT only.
#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define SREVROOT    0x5C1

#ifdef CAP_MINT_CSMINT
/* CSMINT through a descriptor {base, end, perms 0x10f, cursor = base, 0, 0} and a granule private to this use (in
   .data), then an M ldc of the granule (M through a plain base keeps the tag). Clobbers a0, a3, a4, a5. */
#define CS_MINT_INTO(REG, LO, HI) \
  .pushsection .data; .balign 16; 77: .zero 64; .popsection; \
  lla a0, 77b; lla a3, LO; sd a3, 0(a0); lla a4, HI; sd a4, 8(a0); li a5, 0x10f; sd a5, 16(a0); \
  sd a3, 24(a0); sd x0, 32(a0); sd x0, 40(a0); addi a5, a0, 48; .insn r 0x5B, 0x1, 0x50, a3, a5, a0; LDC(REG, a5, 0)
/* the Sublet format stores no type and no rights: a loaded capability decodes NONLIN with every right */
#define MINT(REG, TYPE, LO, HI, PERM) CS_MINT_INTO(REG, LO, HI)
#else
#define MINT(REG, TYPE, LO, HI, PERM) \
  CAPCREATE(REG); li a0, TYPE; lla a3, LO; lla a4, HI; li a5, PERM; \
  CAPTYPE(REG, a0); CAPBOUND(REG, a3, a4); CAPPERM(REG, a5)
#endif

#define CLR  li x9, 0

.option norvc
.section .text.init
.globl _start
_start:
li gp, 1
la t0, trap_handler
csrw mtvec, t0
csrw mcause, x0
csrw medeleg, x0
csrw satp, x0
li x28, 0

# PMP entry 0: NAPOT over everything, RWX
li t0, -1
srli t0, t0, 10
csrw pmpaddr0, t0
li t0, 0x1f
csrw pmpcfg0, t0

# ---------------------------------------------------------------- the table: sublet-ns.S's, PAGED, SL_F4's slot 4
MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)
li t0, 3
slli t0, t0, 62
ori t0, t0, 512
sd t0, 0(s8)                   # capacity 512 | RECYCLE | PAGED
li t0, 6
sd t0, 8(s8)                   # next 6
sd x0, 16(s8)                  # no free list
li t0, 4
sd t0, 24(s8)                  # allocations
li t0, 0x32454741
slli t0, t0, 32
li t1, 0x50444f4e
or t0, t0, t1
sd t0, 32(s8)                  # magic NODPAGE2
sd x0, 40(s8)                  # retired 0
sd x0, 48(s8)                  # growth 0
sd x0, 56(s8)
lla t0, pg_d1
sd t0, 64(s8)                  # root entry 0 -> directory 1
lla t1, pg_d1
lla t0, pg_d2
sd t0, 0(t1)                   # directory 1 [0] -> directory 2
lla t1, pg_d2
lla t0, pg_rp0
sd t0, 0(t1)                   # directory 2 [0] -> record page 0
lla t0, pg_rp1
sd t0, 8(t1)                   # directory 2 [1] -> record page 1
li t0, -1
slli t0, t0, 32
lla t1, pg_rp0
li t2, 512
1:
sd t0, 8(t1)                   # the page-initialisation rule: every record's word 1 retired, both pages
addi t1, t1, 16
addi t2, t2, -1
bnez t2, 1b
lla t1, pg_rp0
li t0, 5
slli t0, t0, 34
sd t0, 32(t1)                  # slot 2: next 5, depth 0
sd x0, 40(t1)                  #         generation 0, prev 0
li t0, 4
slli t0, t0, 34
ori t0, t0, 8
sd t0, 80(t1)                  # slot 5: next 4, depth 1
li t0, 4
sd t0, 88(t1)                  #         prev 2
li t0, 3
slli t0, t0, 34
ori t0, t0, 8
sd t0, 64(t1)                  # slot 4: next 3, depth 1
li t0, 10
li t2, 1
slli t2, t2, 62
or t0, t0, t2
sd t0, 72(t1)                  #         prev 5, generation 2^30 (stale for the generation-0 id x27 carries)
li t0, 8
sd t0, 48(t1)                  # slot 3: next 0, depth 1
li t0, 8
sd t0, 56(t1)                  #         prev 4

# ---------------------------------------------------------------- the capabilities, by cderive in the physical pool
MINT(x18, CAP_TYPE_NONLIN, _start, tbl_end, CAP_PERM_RWX)  # pool node 2: {0,2}
lla a3, data
lla a4, _start
sub a3, a3, a4
CINCOFFSET(t5, x18, a3)
li a1, 4096
CDERIVE(x26, t5, a1)           # pool slot 3: {0,3} over data
CDERIVE(x27, t5, a1)           # pool slot 4: {0,4} over data
li t0, 0x77
sd t0, 0(x26)
sd x0, 8(x26)
lla x24, data                  # the integer address
li t0, 4088
CINCOFFSET(x20, x26, t0)       # x20/s4 = x26's cursor + 4088: ld 8(x20) reads data + 4096, one past the bounds
CINCOFFSET(x21, x27, t0)       # x21/s5 = x27's, the same

lla x29, refbox
li x21, 0
li x22, 0
li x23, 0
li x25, 0
lla x25, res                   # the reading pointer (x25/s9): M prints them all at the end
li x28, 0                      # the trap count, after every setup block

#define REC       sd a5, 0(x25); addi x25, x25, 8
#define EQ(R, V)  sub a5, R, V; seqz a5, a5; REC
#define TO_U(LBL, BACK) li t0, 0x1800; csrc mstatus, t0; lla t0, LBL; csrw mepc, t0; lla x30, BACK; mret

lla t1, tbl
ori t2, t1, 0x7ff
csrw SREVROOT, t2
csrr t2, SREVROOT
EQ(t2, t1)
TO_U(u_prot, m_after_prot)
m_after_prot:
mv a5, a0; REC
mv a5, a1; REC
li t2, 0xfff
csrw SREVROOT, t2
csrr t2, SREVROOT
EQ(t2, x0)
TO_U(u_plain, m_after_plain)
m_after_plain:
mv a5, a0; REC
mv a5, a1; REC
mv a5, x28; REC

lla t1, res
sub t2, x25, t1
srli t2, t2, 3
CAPPRINT(t2)
1:
ld a5, 0(t1)
CAPPRINT(a5)
addi t1, t1, 8
bne t1, x25, 1b
csrw SREVROOT, x0
RVTEST_PASS

.align 4
u_prot:
  # U records in a0 / a1 only: a store through x25, an integer base, would itself raise 24 in protected U
  CLR; ld x22, 0(x27); mv a0, x9
  CLR; ld x22, 0(x24); mv a1, x9
  ecall
.align 4
u_plain:
  li x22, 0
  CLR; ld x22, 0(x27); mv a0, x9
  mv a1, x22
  ecall

.align 4
trap_handler:
  csrr x9, mcause
  li t0, 8
  beq x9, t0, 1f
  li t0, 9
  beq x9, t0, 1f
  addi x28, x28, 1
  csrr x19, mtval
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret
1:
  jr x30

.align 12
img_end:
RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 12
data:
  .zero 8192
.align 12
tbl:
  .zero 4096
pg_d1:
  .zero 4096
pg_d2:
  .zero 4096
pg_rp0:
  .zero 4096
pg_rp1:
  .zero 4096
tbl_end:
.align 6
refbox:
  .zero 64
.align 6
res:
  .zero 512
  TEST_DATA
RVTEST_DATA_END
