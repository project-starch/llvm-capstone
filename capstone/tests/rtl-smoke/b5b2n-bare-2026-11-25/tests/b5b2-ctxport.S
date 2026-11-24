# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# b5b2-ctxport.S -- B5b-2 step 8: three properties of the retired CSRUNV-context tests, ported to run outside a context
#-----------------------------------------------------------------------------
#
# Step 8a made CSRUNV illegal under the Sublet format (capstone-qemu 35de150cef deleted contexts), so sublet-ns.S's arms
# no longer reach their bodies. Three of their properties also exist outside a context (the semantics owner's table,
# 2026-11-25). Here they run in S with srevroot naming sublet-ns.S's paged table: rule 1 checks a tagged base against it,
# the same unit path the context used. Skeleton: sublet-ns.S's table, records, capabilities (cderive in the physical
# pool: x18 = {0,2} the root over the image, x26 = {0,3} and x27 = {0,4} over data, x19 = {0,5} over the page cpg) and
# page tables. There is no frame, no CSRUNV, no supervisor CSR and no CCSRRW. data[0] = 0x77.
# One build per arm:
#   -DCP_F4      F4: slot 4's record holds generation 2^30; S runs  cderive x20 <- x19, 64  (slot 5, whose next is 4: the
#                derive rewrites slot 4 live at the generation it read), then at once  ldc x21, 0(x27).
#   -DCP_F4CTL   its control: the same sequence, slot 4 at generation 0.
#   -DCP_WIDE_LO a capability at slot 2^26 + 0x103 through every widened path: cderive, stc, the raw word 1, ldc reload,
#                cderive from the reloaded one, crevoke; capacity 2^27, next 0x4000103 (sublet-ns-wide's table).
#   -DCP_WIDE_HI the same at slot 0x7FFFFF03: root entry 31, directory indices 0x1FF and 0x1FF, record 3, capacity 2^31
#                (the unit's SLOTS_MAX), so the arm sits at the top edge of every index.
#   -DCP_EDGEM   sublet-ns-edgem's S phase: S names the table, ld from VA 0 faults to M, the ldc behind it misses and
#                probes, M's handler rewrites srevroot as its first instruction; three rounds.
# Prints (M, after S's ecall): the fault cause (0 if none), its pc minus cp_ldc (0 if none), data[8], data[16],
#   data[24], then the trap count (faults only; the final ecall is not counted).
#   An unexpected trap (not 9, 13 or 24-30) ends the run at once: its cause, its mepc (absolute), then the rest.
# FIRST RUN (fcd75ed94) VOID by a port defect, on the base and on 8a alike: the port dropped the li t0, -1 before
#   pmpaddr0, so S fetched under no PMP grant (instruction access fault at s_code), and the handler, sending every
#   fault to cp_after, looped to the cycle limit. Fixed here; the predictions are unchanged.
# PRE-REGISTERED (2026-11-25, written before any run; from sublet-ns.expect's FIXED rows, the reference's): on the base
#   b5b2-ccsrrw2 7997b51a2 and on step 8a alike (the unit is unchanged between them):
#   CP_F4       0x19 0x0 0x0 0x1 0x0 | 1      the ldc reads 25 (slot 4 at generation 2^30 is not the id's 0); the store
#                                              after it never runs
#   CP_F4CTL    0x0 0x0 0x77 0x1 0x0 | 0       the control: slot 4 live, the ldc copies data[0]
#   CP_WIDE_LO  0x0 0x0 0x80000000 0x8000206 0x8000208 | 0   (sublet-ns-wide's: x21's cursor _start; the child's raw
#                                              word 1 = 0x4000103 << 1; the grandchild's = 0x4000104 << 1)
#   CP_WIDE_HI  0x0 0x0 0x80000000 0xfffffe06 0xfffffe08 | 0  (0x7FFFFF03 << 1, 0x7FFFFF04 << 1; generation 0)
#   CP_EDGEM    0x0 0x0 0x0 0x0 0x0 | 3       three load page faults (13); the ldc then re-executes plain (M cleared
#                                              srevroot). The verdict is the trace, on a +R43_TRACE build:
#                                              edgem-r43-check.py exits 0 (a probe dropped while pending), and the .iss
#                                              has EDGE-FLUSHPEND >= 1 and EDGE-PEND == 0 (the trap's flush clears the
#                                              probe before M's csrw). The +R43_TRACE+EDGE_MUT_KEEP_PEND mutant must read
#                                              EDGE-PEND >= 1 (the detector's positive control; run on step 8b).
#   A ported arm that fails on both trees is a port defect; one that fails only on 8a is a finding about 8a.
# B5b-2 (2026-11-25): uses the capcreate family for its setup mints, to be ported to CSMINT with the rest (step 9).
# LEGACY SCAFFOLD (B5b-2 step 9, 2026-11-25): this test mints with the custom-3 debug family, which decodes under the
# Sublet format only in simulation builds with CAPSTONE_SIM_DEBUG_MINT. -DCAP_MINT_CSMINT (the bare kits, silicon)
# mints through CSMINT instead, which allocates its own lifetime record, so pool ids shift by one there. New tests
# mint with CSMINT only.
# REVISION (2026-11-25, before its run; the semantics owner's condition 2): the F4 arms carry a reached-F4 marker.
#   x20 is cleared before the cderive, and at cp_after its type goes to data[16]: 1 = the derive produced a
#   capability, so the race was reached; a pass without it is void. Nothing sits between the cderive and the
#   ldc. The predictions above are updated (data[16] 0x1 in CP_F4 and CP_F4CTL); the others are unchanged.
# KIT BUILD (-DCAP_MINT_CSMINT): CSMINT's root is pool slot 3; s8 is a plain address, the list is re-keyed one up
#   (3 -> 6 -> 5 -> 4, F4's generation on slot 5, next 7), and a key check ends the run with the sentinel
#   0x6b657962616420 if the root is not {0,3}. PRE-REGISTERED, the kit images: as the in-tree vectors, except
#   the wide arms' data[8] (_start is 0x80001000 behind the board prologue). With -DKIT_ROOT_SLOT=4, the
#   sentinel alone. The first kit images (b5b2n, before the re-key) faulted 25 at the cderive (x19 = {0,6}).
#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define SREVROOT    0x5C1
#define PTE_LEAF_RWX  0xDF
#define PTE_LEAF_SRWX 0xCF
#define PTE_PTR       0x01

.option norvc

.section .text.init
.globl _start
_start:

li gp, 1
la t0, trap_handler
csrw mtvec, t0
csrw mcause, x0
li x28, 0                      # the trap count (x28 = t3), zeroed again after every setup block below
lla t0, res
sd x0, 0(t0)
sd x0, 8(t0)

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

# ---------------------------------------------------------------- the table: Sublet, PAGED (cap_rev_table_abi.h)
#ifdef CAP_MINT_CSMINT
lla s8, tbl                    # the kit build: M is integer, the table is written plainly (no second mint)
#else
MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)
#endif
li t0, 3
slli t0, t0, 62
#if defined(CP_WIDE_LO)
li t1, 1
slli t1, t1, 27
or t0, t0, t1
sd t0, 0(s8)                   # capacity 2^27 | RECYCLE | PAGED
li t0, 0x4000103
sd t0, 8(s8)                   # next 2^26 + 0x103: the next allocation lands under root entry 1
#elif defined(CP_WIDE_HI)
li t1, 1
slli t1, t1, 31
or t0, t0, t1
sd t0, 0(s8)                   # capacity 2^31 (SLOTS_MAX) | RECYCLE | PAGED
li t0, 0x7fffff03
sd t0, 8(s8)                   # next 0x7FFFFF03: root entry 31, directories 0x1FF / 0x1FF, record 3
#else
ori t0, t0, 512
sd t0, 0(s8)                   # capacity 512 | RECYCLE | PAGED
#ifdef CAP_MINT_CSMINT
li t0, 7
sd t0, 8(s8)                   # next 7 (the kit's re-key)
#else
li t0, 6
sd t0, 8(s8)                   # next 6
#endif
#endif
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
#if defined(CP_WIDE_LO) || defined(CP_WIDE_HI)
#if defined(CP_WIDE_LO)
lla t0, pg_hd1
sd t0, 72(s8)                  # root entry 1 -> directory 1'
lla t1, pg_hd1
lla t0, pg_hd2
sd t0, 0(t1)                   # directory 1' [0] -> directory 2'
lla t1, pg_hd2
lla t0, pg_hrp
sd t0, 8(t1)                   # directory 2' [1] -> the high record page
#else
lla t0, pg_hd1
sd t0, 312(s8)                 # root entry 31 (64 + 8 * 31) -> directory 1'
lla t1, pg_hd1
li t2, 4088
add t1, t1, t2
lla t0, pg_hd2
sd t0, 0(t1)                   # directory 1' [0x1FF] -> directory 2'
lla t1, pg_hd2
add t1, t1, t2
lla t0, pg_hrp
sd t0, 0(t1)                   # directory 2' [0x1FF] -> the high record page
#endif
li t0, -1
slli t0, t0, 32
lla t1, pg_hrp
li t2, 256
9:
sd t0, 8(t1)                   # the page rule: every word 1 retired
addi t1, t1, 16
addi t2, t2, -1
bnez t2, 9b
#endif
# the page-initialisation rule: every record's word 1 retired, both pages
li t0, -1
slli t0, t0, 32
lla t1, pg_rp0
li t2, 512
1:
sd t0, 8(t1)
addi t1, t1, 16
addi t2, t2, -1
bnez t2, 1b
# the records: word 0 = {next[29:0] << 34 | depth << 3}, word 1 = {generation << 32 | prev << 1 | next[30]}
#ifdef CAP_MINT_CSMINT
lla t1, pg_rp0
# the kit build (CAP_MINT_CSMINT): CSMINT's root takes pool slot 3, so the list is re-keyed one up, 3 -> 6 -> 5 -> 4
li t0, 6
slli t0, t0, 34
sd t0, 48(t1)                  # slot 3: next 6, depth 0 (the root)
sd x0, 56(t1)                  #         generation 0, prev 0
li t0, 5
slli t0, t0, 34
ori t0, t0, 8
sd t0, 96(t1)                  # slot 6: next 5, depth 1
li t0, 6
sd t0, 104(t1)                 #         prev 3
li t0, 4
slli t0, t0, 34
ori t0, t0, 8
sd t0, 80(t1)                  # slot 5: next 4, depth 1
li t0, 12
#ifdef CP_F4
li t2, 1
slli t2, t2, 62                # generation 2^30 at word 1 bit 32
or t0, t0, t2
#endif
sd t0, 88(t1)                  #         prev 6 (F4: generation 2^30)
li t0, 8
sd t0, 64(t1)                  # slot 4: next 0, depth 1
li t0, 10
sd t0, 72(t1)                  #         prev 5
#else
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
#ifdef CP_F4
li t2, 1
slli t2, t2, 62                # generation 2^30 at word 1 bit 32
or t0, t0, t2
#endif
sd t0, 72(t1)                  #         prev 5 (F4: generation 2^30)
li t0, 8
sd t0, 48(t1)                  # slot 3: next 0, depth 1
li t0, 8
sd t0, 56(t1)                  #         prev 4
#endif

# ---------------------------------------------------------------- the capabilities, by cderive in the physical pool
MINT(x18, CAP_TYPE_NONLIN, _start, pt_end, CAP_PERM_RWX)   # pool node 2: {0,2}
#ifdef CAP_MINT_CSMINT
#ifndef KIT_ROOT_SLOT
#define KIT_ROOT_SLOT 3
#endif
# the kit's key check (the semantics owner's condition 1): the root CSMINT returned must be the slot and generation the
# re-keyed tables assume, {0, KIT_ROOT_SLOT}. Word 1 of the minted granule (a5) is generation << 32 | slot << 1 | bounds[26].
# On a mismatch the run ends with the sentinel alone, never a vector. -DKIT_ROOT_SLOT=4 is the deliberate skew.
ld a3, 8(a5)
srli a4, a3, 1
li a0, 0x7fffffff
and a4, a4, a0
li a0, KIT_ROOT_SLOT
bne a4, a0, kit_key_bad
srli a4, a3, 32
bnez a4, kit_key_bad
j kit_key_ok
kit_key_bad:
li a0, 0x6b657962616420        # the sentinel
CAPPRINT(a0)
RVTEST_PASS
kit_key_ok:
#endif
lla a3, data
lla a4, _start
sub a3, a3, a4
CINCOFFSET(t5, x18, a3)
li a1, 4096
CDERIVE(x26, t5, a1)           # pool slot 3: {0,3} over data
CDERIVE(x27, t5, a1)           # pool slot 4: {0,4} over data
lla a3, cpg
lla a4, _start
sub a3, a3, a4
CINCOFFSET(t5, x18, a3)
CDERIVE(x19, t5, a1)           # pool slot 5: {0,5} over the page cpg
li t0, 0x77
sd t0, 0(x26)
sd x0, 8(x26)
sd x0, 16(x26)
sd x0, 24(x26)

# ---------------------------------------------------------------- page tables as va-alloc.S; the S-mode page without U
MINT(s7, CAP_TYPE_NONLIN, pt_root, pt_end, CAP_PERM_RW)
lla t0, pt_l0
lla t1, pt_root
sub a7, t0, t1
lla t0, pt_l1
sub s9, t0, t1
li t1, 0x80000000
li t2, 0
1:
srli t3, t1, 12
slli t3, t3, 10
ori t3, t3, PTE_LEAF_RWX
slli t4, t2, 3
add t4, t4, a7
CINCOFFSET(t0, s7, t4)
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
CINCOFFSET(t0, s7, s9)
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
CINCOFFSET(t0, s7, t4)
sd t3, 0(t0)
addi t2, t2, 1
li t4, 512
bne t2, t4, 2b
lla t1, pt_l1
srli t1, t1, 12
slli t1, t1, 10
ori t1, t1, PTE_PTR
li t4, 16
CINCOFFSET(t0, s7, t4)
sd t1, 0(t0)
# L0[s_code] -> no U
lla t0, s_code
li t1, 0x80000000
sub t0, t0, t1
srli t0, t0, 12
slli t0, t0, 3
add t0, t0, a7
CINCOFFSET(t2, s7, t0)
lla t1, s_code
srli t1, t1, 12
slli t1, t1, 10
ori t1, t1, PTE_LEAF_SRWX
sd t1, 0(t2)
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
li x28, 0                      # the trap count, after every setup block

# ---------------------------------------------------------------- into scalar S (x18 stays the root)
li t0, 0x1800                  # MPP := 01 (S)
csrc mstatus, t0
li t0, 0x0800
csrs mstatus, t0
lla t0, s_code
csrw mepc, t0
mret

.align 12
s_code:
#if defined(CP_EDGEM)
  // sublet-ns-edgem's S phase, unchanged: the edge inside the probe's lifetime
  li t0, 1
  slli t0, t0, 18
  csrs sstatus, t0             # SUM: S reads the U data page
  li a4, 3                     # three rounds: the first warms M's handler in the I$, so later edges come fast
8:
  lla t0, tbl
  csrw 0x5c1, t0               # S's namespace: the paged table
  ld t1, 0(x0)                 # VA 0 unmapped: a load page fault to M
cp_ldc:
  LDC(x21, x26, 0)             # behind it: misses, raises a probe for {0,3}
  csrw 0x5c1, x0               # back to no namespace (after M's handler returns)
  addi a4, a4, -1
  bnez a4, 8b
#else
  lla t0, tbl
  csrw 0x5c1, t0               # S's namespace: the paged table (rule 1 checks a tagged base against it)
  li t0, 64
#if defined(CP_F4) || defined(CP_F4CTL)
  li x20, 0                    # the marker's precondition: x20 holds no capability before the derive
  CDERIVE(x20, x19, t0)        # slot 5: its next, slot 4, is rewritten live (CP_F4: at generation 2^30)
cp_ldc:
  LDC(x21, x27, 0)             # at once, through {0,4}
  sd x21, 8(x26)
#else
  CDERIVE(x20, x18, t0)        # fresh slot: the table's next
  STC(x26, x20, 64)            # stored in the Sublet format at data + 64
  ld t1, 72(x26)
  sd t1, 16(x26)               # its word 1, raw
  LDC(x21, x26, 64)            # reloaded
  CDERIVE(x22, x21, t0)        # derived from the reloaded capability: its id must have round-tripped live
cp_ldc:
  STC(x26, x22, 96)
  ld t1, 104(x26)
  sd t1, 24(x26)               # the grandchild's word 1, raw
  CREVOKE(x21, x0)             # end the reloaded child and its run
  sd x21, 8(x26)               # x21's integer view: the cursor
#endif
cp_after:
#if defined(CP_F4) || defined(CP_F4CTL)
  LCC(t1, x20, x1)
  sd t1, 16(x26)               # the reached-F4 marker: data[16] = x20's type, 1 = the derive gave a capability
#endif
  csrw 0x5c1, x0
#endif
  li a7, 1
  ecall                        # to M: the readings
1: j 1b

.align 12
trap_handler:
#if defined(CP_EDGEM)
  csrw 0x5c1, x0               # FIRST: the namespace edge, while the ldc's probe may still be pending
#endif
  csrr t0, mcause
  li t1, 9
  beq t0, t1, 5f               # S's ecall: print
  addi x28, x28, 1             # a fault: count it
  li t1, 13
  beq t0, t1, 4f
  li t1, 24
  bltu t0, t1, 3f             # not a capability fault (24-30): unexpected
  li t1, 31
  bltu t0, t1, 6f
3:
  lla t1, res                  # an unexpected trap: record it and end the run with the readings
  ld t2, 0(t1)
  bnez t2, 5f
  sd t0, 0(t1)
  csrr t2, mepc
  sd t2, 8(t1)
  j 5f
4:
  csrr t0, mepc                # a load page fault (EDGEM): skip the faulting load; the ldc re-executes after mret
  addi t0, t0, 4
  csrw mepc, t0
  mret
6:
  lla t1, res                  # a capability fault: record cause and pc, then skip to cp_after (the rest never runs)
  ld t2, 0(t1)
  bnez t2, 7f                  # keep the first
  sd t0, 0(t1)
  csrr t2, mepc
  lla t4, cp_ldc
  sub t2, t2, t4
  sd t2, 8(t1)
7:
#if defined(CP_EDGEM)
  csrr t0, mepc
  addi t0, t0, 4
#else
  lla t0, cp_after
#endif
  csrw mepc, t0
  mret
5:
  lla t1, res
  ld t2, 0(t1)
  CAPPRINT(t2)                 # the fault cause, 0 if none
  ld t2, 8(t1)
  CAPPRINT(t2)                 # its pc minus cp_ldc
  lla t1, data
  ld t2, 8(t1)
  CAPPRINT(t2)                 # data[8]
  ld t2, 16(t1)
  CAPPRINT(t2)                 # data[16]
  ld t2, 24(t1)
  CAPPRINT(t2)                 # data[24]
  CAPPRINT(x28)                # the trap count
  RVTEST_PASS

.align 12
cpg:
  .zero 4096

.align 4
_end_of_text:

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 12
data:
  .zero 4096
data_end:
.align 4
res:
  .zero 16
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
pg_hd1:
  .zero 4096
pg_hd2:
  .zero 4096
pg_hrp:
  .zero 4096
tbl_end:
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
  TEST_DATA
RVTEST_DATA_END
