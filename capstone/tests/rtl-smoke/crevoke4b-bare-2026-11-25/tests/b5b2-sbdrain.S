# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# b5b2-sbdrain.S -- does a lifetime operation see the hart's older committed stores to the table? (2026-11-25)
#-----------------------------------------------------------------------------
#
# The semantics owner's question on the claim-auditor's residual (4(b) audit): the reference reads the table as memory
# holds it, with no ordering clause for software writes; the trap-route kernel grows the table with plain stores (the
# record page's words, the directory entry, then CAPACITY, sublet-table.h:81) ordered by SUBLET_WMB, which on its !SMP
# build is a compiler barrier and no fence; then sret resumes the CDERIVE that raised 30. The lifetime unit reads the
# table through its own D$ ports, not through the LSU, so a store still in the committed store queue or the D$ write
# buffer could be missed.
# Table: b5b2-srvcost.S's paged table, but FULL: capacity 256 (one record page), next 256, record page 1 (pg_rp1)
# not published (directory 2 [1] = 0). CSMINT's root {0,3} over page_a in x26/s10. srevroot = tbl. A CDERIVE from the
# root therefore raises 30 (it can grow). The growth, as the kernel does it: (a) record page 1's 256 records written
# retired (the page-initialisation rule; 256 stores), (b) directory 2 [1] := pg_rp1, (c) CAPACITY := 512. A CDERIVE
# that reads the table as memory holds it after (c) allocates slot 256: cause 0, id 0x100. One that reads the old
# CAPACITY raises 30 again; one that reads the new CAPACITY but the old directory entry finds slot 256 unreachable
# (29).
#   SB_NONE   S: cderive (30), (b) (c), cderive at once
#   SB_FILL   S: cderive (30), (a) (b) (c), cderive at once (256 stores ahead of (b) and (c) in the store path)
#   SB_FENCE  as SB_FILL with fence rw, rw before the second cderive (the control: every older store drained)
#   SB_SRET   U: cderive -> 30 to S (medeleg 30); S's handler does (a) (b) (c) and srets; U's cderive retries
#   SB_MINT   S: csmint (30), (a) (b) (c), csmint at once. CSMINT's record request reads the header right after its
#             descriptor loads, while CDERIVE's first request is its check (a lifetime operation that reads no CAPACITY),
#             so this arm's window between the last growth store and the first CAPACITY read is the shorter one.
#   SB_COST   the drain's cost: 16 rounds of {4 stores to distinct lines, cderive of a 16-byte child of the root,
#             crevoke of it} in S on a table with room (the growth done first), timed by rdcycle, against the same 16
#             rounds with the stores replaced by nops (SB_COST0). Prints: the first cause, the cycles, 0, the M traps.
# Prints: the first cderive's cause (SB_SRET: the number of 30s S took), the second's cause, its id (lcc 8), the M trap
# count. The M handler records an exception's cause in s1/x9, counts it in t3/x28 and skips; S's handler uses a2-a5.
# PRE-REGISTERED (2026-11-25, written before any run), in-tree, memory delay 12:
#   SB_MINT on both trees: 0x1e then the second csmint's cause and id RECORDED on 904cd9c6b; with the drain 0x0 and a
#     nonzero id. SB_COST / SB_COST0: cycles recorded on both trees; the drain's cost per round is the difference.
# THIRD RUN (1096cc7bb on 904cd9c6b's core files): SB_SRET 0x1 0x0 0x1 0x0 (one 30, the retry allocated); SB_COST
#   0x1e 0x11ba, SB_COST0 0x1e 0x107d (16 rounds). So 904cd9c6b shows NO exposure in any of the five shapes (none, fill,
#   fence, mint, sret): 30, then the allocation, every time. Not a proof of absence: the window is the store's time in
#   the committed queue before it reaches the D$ (a write-through hit updates the line), against how long a lifetime
#   operation takes to reach its first CAPACITY read (CDERIVE: its check first; CSMINT: its descriptor loads).
#   RTL with the drain (a lifetime operation waits for the committed store queue and the D$ write buffer to empty
#   before its first table read): every arm 0x1e 0x0 0x100 <M traps> (SB_SRET: 0x1 0x0 0x100); M traps 1 in the S
#   arms (the first cderive's 30), 0 in SB_SRET.
#   904cd9c6b (no drain): SB_FENCE as above (the control). SB_NONE, SB_FILL and SB_SRET RECORDED, not predicted: a 0x1e or
#   0x1d second cause is the exposure; 0x0 0x100 means this shape does not expose it on this tree.
# FIRST RUN (5ddb5490f's test on 904cd9c6b's core files): SB_NONE, SB_FILL and SB_FENCE 0x1e 0x0 0x100 0x1; SB_SRET VOID
#   (0x0 0x0 0x0 0x4): U stored its results through integer bases, which protected U refuses with 24 (my slip; the four
#   traps are those stores). The trace shows the S handler entered and the retried cderive retiring. U now keeps its
#   results in registers and M stores them.
# SECOND RUN (8be7a483e on 904cd9c6b's core files): SB_MINT 0x1e 0x0 0x100 0x1 (no exposure on the shorter window);
#   SB_SRET 0x1 0x0 0x0 0x0, its third print uninformative (lcc 8 reads 0 in U; it now prints the type, 1 = allocated,
#   7 = the FAIL); SB_COST / SB_COST0 VOID: this file did not open mcounteren, so both rdcycles in S trapped (the two
#   extra M traps). Both fixed. PRE-REGISTERED for SB_SRET's third print: 0x1 on both trees.
#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define SREVROOT 0x5C1

#define CS_MINT_INTO(REG, LO, HI) \
  .pushsection .data; .balign 16; 77: .zero 64; .popsection; \
  lla a0, 77b; lla a3, LO; sd a3, 0(a0); lla a4, HI; sd a4, 8(a0); li a5, 0x10f; sd a5, 16(a0); \
  sd a3, 24(a0); sd x0, 32(a0); sd x0, 40(a0); addi a5, a0, 48; .insn r 0x5B, 0x1, 0x50, a3, a5, a0; LDC(REG, a5, 0)

/* the growth, as the kernel does it: (a) when FILL, then (b) and (c); clobbers a2-a5 */
.macro GROW fill
.if \fill
  lla a2, pg_rp1
  li a3, -1
  slli a3, a3, 32
  li a4, 256
1:
  sd x0, 0(a2)
  sd a3, 8(a2)
  addi a2, a2, 16
  addi a4, a4, -1
  bnez a4, 1b
.endif
  lla a2, pg_d2
  lla a3, pg_rp1
  sd a3, 8(a2)                 # (b) directory 2 [1] := record page 1
  lla a2, tbl
  li a3, 3
  slli a3, a3, 62
  ori a3, a3, 512
  sd a3, 0(a2)                 # (c) CAPACITY := 512
.endm

.option norvc
.section .text.init
.globl _start
_start:
li gp, 1
lla t0, trap_handler
csrw mtvec, t0
csrw mcause, x0
csrw medeleg, x0
csrw mideleg, x0
csrw mie, x0
csrw satp, x0
li t0, -1
csrw mcounteren, t0            # rdcycle in S (SB_COST)
li t0, -1
srli t0, t0, 10
csrw pmpaddr0, t0
li t0, 0x1f
csrw pmpcfg0, t0

# ---------------------------------------------------------------- the table: b5b2-mtrap.S's
lla s8, tbl
li t0, 3
slli t0, t0, 62
ori t0, t0, 256
sd t0, 0(s8)                   # capacity 256 | RECYCLE | PAGED: one record page
li t0, 256
sd t0, 8(s8)                   # next 256: every slot of page 0 taken, the table full
sd x0, 16(s8)
li t0, 4
sd t0, 24(s8)
li t0, 0x32454741
slli t0, t0, 32
li t1, 0x50444f4e
or t0, t0, t1
sd t0, 32(s8)
sd x0, 40(s8)
sd x0, 48(s8)
sd x0, 56(s8)
lla t0, pg_d1
sd t0, 64(s8)
lla t1, pg_d1
lla t0, pg_d2
sd t0, 0(t1)
lla t1, pg_d2
lla t0, pg_rp0
sd t0, 0(t1)
li t0, -1
slli t0, t0, 32
lla t1, pg_rp0
li t2, 512
1:
sd t0, 8(t1)
addi t1, t1, 16
addi t2, t2, -1
bnez t2, 1b
lla t1, pg_rp0
li t0, 6
slli t0, t0, 34
sd t0, 48(t1)
sd x0, 56(t1)
li t0, 5
slli t0, t0, 34
ori t0, t0, 8
sd t0, 96(t1)
li t0, 6
sd t0, 104(t1)
li t0, 4
slli t0, t0, 34
ori t0, t0, 8
sd t0, 80(t1)
li t0, 12
li t2, 1
slli t2, t2, 62
or t0, t0, t2
sd t0, 88(t1)
li t0, 8
sd t0, 64(t1)
li t0, 10
sd t0, 72(t1)

CS_MINT_INTO(x26, page_a, page_a_end)
ld a3, 8(a5)
srli a4, a3, 1
li a0, 0x7fffffff
and a4, a4, a0
li a0, 3
bne a4, a0, kit_key_bad
srli a4, a3, 32
bnez a4, kit_key_bad
j kit_key_ok
kit_key_bad:
li a0, 0x6b657962616420
CAPPRINT(a0)
RVTEST_PASS
kit_key_ok:
li x28, 0

li s1, 0
lla t0, tbl
csrw SREVROOT, t0
#ifdef SB_SRET
li t0, 1
slli t0, t0, 30
csrw medeleg, t0               # cause 30 to S
lla t0, s_handler
csrw stvec, t0
li t0, 0x1800
csrc mstatus, t0               # MPP = U
lla t0, u_part
#else
li t0, 0x1800
csrc mstatus, t0
li t0, 0x800
csrs mstatus, t0               # MPP = S
lla t0, s_part
#endif
csrw mepc, t0
lla x30, m_after
mret

m_after:
#ifdef SB_SRET
lla t1, results
lla t2, s_count
ld t2, 0(t2)
sd t2, 0(t1)
sd a6, 8(t1)
sd a7, 16(t1)
#endif
lla t1, results
ld t2, 0(t1)
CAPPRINT(t2)
ld t2, 8(t1)
CAPPRINT(t2)
ld t2, 16(t1)
CAPPRINT(t2)
CAPPRINT(x28)
csrw SREVROOT, x0
RVTEST_PASS

#ifdef SB_SRET
.align 4
u_part:
  li a1, 16
  li s1, 0
  CDERIVE(x20, x26, a1)        # 30 to S, which grows the table and srets here: the retry
  mv a6, s1
  li a7, 0; LCC(a7, x20, x1)   # the type: 1 = a tagged child (allocated), 7 = untagged (the no-fault FAIL). LCC 8
                               # (the id) answers only in S, and reads 0 in U
  ecall                        # protected U makes no plain access: M stores a6 and a7 (m_after)
.align 4
s_handler:
  lla a2, s_count
  ld a3, 0(a2)
  addi a3, a3, 1
  sd a3, 0(a2)
  GROW 1
  sret                         # sepc is the cderive: it executes again
#elif defined(SB_MINT)
.align 4
s_part:
  li s1, 0
  CS_MINT_INTO(x20, page_a, page_a_end)   # the table is full: 30 (the record request)
  mv a6, s1
  GROW 1
  li s1, 0
  CS_MINT_INTO(x21, page_a, page_a_end)   # a3 = CSMINT's rd: generation << 31 | slot
  mv a7, a3
  mv a5, s1
  lla a4, results
  sd a6, 0(a4)
  sd a5, 8(a4)
  sd a7, 16(a4)
  ecall
#elif defined(SB_COST) || defined(SB_COST0)
.align 4
s_part:
  li a1, 16
  li s1, 0
  CDERIVE(x20, x26, a1)        # the table is full: 30
  mv a6, s1
  GROW 0                       # room now
  lla s2, page_a               # s2/x18: named nowhere else (t3/x28 is the M counter, s8/x24 the table)
  li s1, 0
  rdcycle a0
  li a7, 16
2:
#ifdef SB_COST
  sd a7, 0(s2)
  sd a7, 512(s2)
  sd a7, 1024(s2)
  sd a7, 1536(s2)              # four stores to distinct lines of page_a
#else
  nop
  nop
  nop
  nop
#endif
  CDERIVE(x21, x26, a1)
  CREVOKE(x21, x0)
  addi a7, a7, -1
  bnez a7, 2b
  rdcycle a2
  sub a5, a2, a0
  lla a4, results
  sd a6, 0(a4)
  sd a5, 8(a4)
  sd x0, 16(a4)
  ecall
#else
.align 4
s_part:
  li a1, 16
  li s1, 0
  CDERIVE(x20, x26, a1)        # the table is full: 30
  mv a6, s1
#if defined(SB_FILL) || defined(SB_FENCE)
  GROW 1
#else
  GROW 0
#endif
#ifdef SB_FENCE
  fence rw, rw
#endif
  li a1, 16
  li s1, 0
  CDERIVE(x21, x26, a1)
  mv a5, s1
  li a7, 0; LCC(a7, x21, x8)
  lla a4, results
  sd a6, 0(a4)
  sd a5, 8(a4)
  sd a7, 16(a4)
  ecall
#endif

.align 4
trap_handler:
  csrr t0, mcause
  li t1, 9
  beq t0, t1, 1f
  li t1, 8
  beq t0, t1, 1f
  mv s1, t0
  addi x28, x28, 1
  csrr t1, mepc
  addi t1, t1, 4
  csrw mepc, t1
  mret
1:
  jr x30

.align 12
img_end:
RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.balign 16
results: .zero 32
s_count: .dword 0
.align 12
page_a:
  .zero 4096
page_a_end:
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
RVTEST_DATA_END
