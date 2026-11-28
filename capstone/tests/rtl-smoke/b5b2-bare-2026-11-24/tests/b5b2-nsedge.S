# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# b5b2-nsedge.S -- B5b-2 step 7: the namespace must-pass for the consumers outside M (S under rule 1, protected U)
#-----------------------------------------------------------------------------
#
# The plan's step 7: no lifetime verdict from the previous namespace is consumed after a srevroot write. srevroot is
# switched between two accesses, each csrw immediately followed by the access (the write flushes the pipeline, so the
# access is the next instruction to issue). T1 = tbl (b5b2-sref.S's table); T2 = tbl2, T1's five pages copied and
# repointed, with slot 3 at generation 1 and slot 4 at generation 0. So A (x26/s10, id {0, 3}) is LIVE in T1 and STALE
# in T2, and B (x27/s11, id {0, 4}) is STALE in T1 (slot 4 at generation 2^30) and LIVE in T2. B's first access is in
# T1, so a DEAD verdict is cached before the stale-to-live switch.
# va-rc-trap / -zp cannot serve here: they build P1-format tables (va-fetchsplit.S :736-756, a flat RECYCLE header and
# record 2's word 1 = 1 << 60), so under the Sublet format the context entry is refused (mcause 29) before the R-75
# window; B5b-1's and B5b-2's tips read the same short vector. They stay ids55 regression arms.
# Fetch is not covered because under Sublet no fetch check consults the namespace: both PC-capability checks need M
# and capmode (commit_stage.sv:423, pc_cap_check; :518, pcc_chk_o), and capmode is set only by the illegal CAPENTER.
# The reference agrees: protected U checks object authority on its DATA accesses (capstone-qemu 9289114e06
# translate.c:130-132; op_helper.c:1387-1390, capstone_protected_data).
# Traps go to M; the handler records mcause in x9/s1, counts in x28/t3, resumes at mepc + 4. Readings are stored at
# x25/s9 (S too); U holds its own in a0-a3. M prints them all at the end, after their count.
# Readings (S under rule 1: a tagged base is checked against srevroot's table; srevroot 0 makes the access plain):
#   1 T1 ld A        2 T1 -> T2 ld A       3 T2 -> T1 ld A        4 T1 -> 0 ld A (plain)    5 0 -> T1 ld A
#   6 T1 -> T2 ldc A                       7 T2 -> T1 ldc A
#   8 T1 ld B (stale: DEAD cached)         9 T1 -> T2 ld B        10 T2 -> T1 ldc B         11 T1 -> T2 ldc B
#   protected U, each switch made in S right before the sret into U, whose first instruction is the load:
#   12 T1 A    13 T1 -> T2 A (live to stale)    14 T2 -> T1 B (stale)    15 T1 -> T2 B (stale to live)
#   16 the trap count
# PRE-REGISTERED (2026-11-24, written before any run):
#   b5b2-step6 0c9992bc6:   0 0x19 0 0 0 | 0x19 0 | 0x19 0 0x19 0 | 0 0x19 0x19 0 | 6
#   the mutant +NS_SIM_NOFLUSH (sim only: the LSU's lifetime-verdict cache is NOT emptied on a namespace change):
#                           0 0 0 0 0 | 0 0 | 0x19 0x19 0x19 0x19 | 0 0 0x19 0x19 | 6
#     A's ALLOW cached at its cderive (M, the physical pool) is consumed everywhere, so every A reading reads 0; B, ended
#     in the pool first, takes a DEAD verdict in T1 at reading 8, which then answers every later B reading (25). The
#     discriminating readings: live to stale 2, 6, 13 (build 25, mutant 0) and stale to live 9, 11, 15 (build 0,
#     mutant 25). A reading among those six that the mutant leaves unmoved is void.
# FIRST RUN (eb8e32f08, before B was ended in the pool): the build read all 16 as pre-registered; the mutant read 0
#   everywhere, B included. B's ALLOW from its cderive in the pool was never flushed, so under the mutant B never
#   took a DEAD verdict: readings 9, 11, 15 were not moved by the mutant and were VOID as stale-to-live tests, and 8,
#   10, 14 measured a pool-to-T1 leak instead. Fixed by ending B in the pool before the S phase (CREVOKE, which
#   invalidates the cached verdict); the build's readings are unchanged by it (B is stale in T1 either way).
# B5b-2 (2026-11-24, the semantics owner's ruling): uses the capcreate family, to be ported to CSMINT.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define SREVROOT    0x5C1

#define MINT(REG, TYPE, LO, HI, PERM) \
  CAPCREATE(REG); li a0, TYPE; lla a3, LO; lla a4, HI; li a5, PERM; \
  CAPTYPE(REG, a0); CAPBOUND(REG, a3, a4); CAPPERM(REG, a5)

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
#define CAUSE     mv a5, x9; REC
#define TO_S(LBL, BACK) li t0, 0x1800; csrc mstatus, t0; li t0, 0x800; csrs mstatus, t0; lla t0, LBL; csrw mepc, t0; lla x30, BACK; mret

# ---------------------------------------------------------------- T2: T1's five pages copied, repointed, slot 3 stale
lla a1, tbl
lla a2, tbl2
li a3, 2560
1:
ld a4, 0(a1)
sd a4, 0(a2)
addi a1, a1, 8
addi a2, a2, 8
addi a3, a3, -1
bnez a3, 1b
lla a1, tbl2
lla a2, pg2_d1
sd a2, 64(a1)                  # root entry 0 -> T2's directory 1
lla a1, pg2_d1
lla a2, pg2_d2
sd a2, 0(a1)                   # directory 1 [0] -> T2's directory 2
lla a1, pg2_d2
lla a2, pg2_rp0
sd a2, 0(a1)                   # directory 2 [0] -> T2's record page 0
lla a2, pg2_rp1
sd a2, 8(a1)                   # directory 2 [1] -> T2's record page 1
lla a1, pg2_rp0
ld a2, 56(a1)
li a3, 1
slli a3, a3, 32
or a2, a2, a3
sd a2, 56(a1)                  # slot 3 word 1: generation 1 (A carries generation 0: stale in T2)
li a2, 10
sd a2, 72(a1)                  # slot 4 word 1: generation 0, prev 5 (B carries generation 0: live in T2)
CREVOKE(x27, x0)               # B ended in the physical pool (srevroot 0 here): its ALLOW cached at the cderive is
                               # invalidated, so B's first T1 access probes and caches DEAD (see FIRST RUN)
lla a6, tbl                    # T1
lla a7, tbl2                   # T2
csrw SREVROOT, x0

# ---------------------------------------------------------------- S, rule 1
TO_S(s_ns, m_after_s)
m_after_s:

# ---------------------------------------------------------------- protected U, the switch in S right before the sret
li a0, 0x5a5a
li a1, 0x5a5a
li a2, 0x5a5a
li a3, 0x5a5a
TO_S(s_u1, m_after_u)
m_after_u:
mv a5, a0; REC
mv a5, a1; REC
mv a5, a2; REC
mv a5, a3; REC
csrw SREVROOT, x0
mv a5, x28; REC

# ---------------------------------------------------------------- print: the count, then every reading
lla t1, res
sub t2, x25, t1
srli t2, t2, 3
CAPPRINT(t2)
1:
ld a5, 0(t1)
CAPPRINT(a5)
addi t1, t1, 8
bne t1, x25, 1b
RVTEST_PASS

.align 4
s_ns:
  csrw SREVROOT, a6
  CLR; ld x22, 0(x26); CAUSE
  csrw SREVROOT, a7
  CLR; ld x22, 0(x26); CAUSE
  csrw SREVROOT, a6
  CLR; ld x22, 0(x26); CAUSE
  csrw SREVROOT, x0
  CLR; ld x22, 0(x26); CAUSE
  csrw SREVROOT, a6
  CLR; ld x22, 0(x26); CAUSE
  csrw SREVROOT, a7
  CLR; LDC(x23, x26, 0); CAUSE
  csrw SREVROOT, a6
  CLR; LDC(x23, x26, 0); CAUSE
  CLR; ld x22, 0(x27); CAUSE
  csrw SREVROOT, a7
  CLR; ld x22, 0(x27); CAUSE
  csrw SREVROOT, a6
  CLR; LDC(x23, x27, 0); CAUSE
  csrw SREVROOT, a7
  CLR; LDC(x23, x27, 0); CAUSE
  ecall
.align 4
s_u1:
  csrw SREVROOT, a6
  lla t0, u_1
  j s_sret
s_u2:
  csrw SREVROOT, a7
  lla t0, u_2
  j s_sret
s_u3:
  csrw SREVROOT, a6
  lla t0, u_3
  j s_sret
s_u4:
  csrw SREVROOT, a7
  lla t0, u_4
s_sret:
  csrw sepc, t0
  li t0, 0x100
  csrc sstatus, t0             # SPP = U
  sret
.align 4
u_1:
  CLR; ld x22, 0(x26); mv a0, x9
  ecall
u_2:
  CLR; ld x22, 0(x26); mv a1, x9
  ecall
u_3:
  CLR; ld x22, 0(x27); mv a2, x9
  ecall
u_4:
  CLR; ld x22, 0(x27); mv a3, x9
  ecall

.align 4
trap_handler:
  csrr x9, mcause
  li t0, 8
  beq x9, t0, 2f
  li t0, 9
  beq x9, t0, 1f
  addi x28, x28, 1
  csrr x19, mtval
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret
2:
  # an ecall from U: legs 1-3 go on to S's next leg, the fourth ends the U phase
  lla t0, u_legs
  ld t1, 0(t0)
  addi t1, t1, 1
  sd t1, 0(t0)
  li t0, 4
  beq t1, t0, 1f
  li t0, 0x1800
  csrc mstatus, t0
  li t0, 0x800
  csrs mstatus, t0             # MPP = S
  lla t0, s_u2
  li t2, 1
  beq t1, t2, 3f
  lla t0, s_u3
  li t2, 2
  beq t1, t2, 3f
  lla t0, s_u4
3:
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
.align 12
tbl2:
  .zero 4096
pg2_d1:
  .zero 4096
pg2_d2:
  .zero 4096
pg2_rp0:
  .zero 4096
pg2_rp1:
  .zero 4096
.align 6
refbox:
  .zero 64
.align 6
u_legs:
  .zero 8
.align 6
res:
  .zero 512
  TEST_DATA
RVTEST_DATA_END
