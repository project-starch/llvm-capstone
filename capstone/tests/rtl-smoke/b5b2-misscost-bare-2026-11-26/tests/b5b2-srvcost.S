# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# b5b2-srvcost.S -- the cost of an srevroot write at switch_mm, in cycles (B5b-2 plan, "Cost to measure in simulation")
#-----------------------------------------------------------------------------
#
# srevroot (0x5C1) is written at every switch_mm. Each write raises flush_o (csr_regfile.sv, CSR_SREVROOT), and a
# write that CHANGES it moves the lifetime namespace, which empties the LSU's revocation cache, the ALLOW record and the
# trackers (ex_stage.sv revnode_ns_flush_o: srevroot_rev_q != srevroot_rev_qq; load_store_unit.sv revnode_ns_flush_i).
# This test reads both parts.
# Setup (M): b5b2-mtrap.S's table (b5b2-sref.S's paged table in its CSMINT keying, srevroot = tbl), CSMINT's root {0,3}
# over page_a in x26; a second table base tbl2 (never used for a lookup, only written to srevroot and written back);
# mcounteren / scounteren open the cycle counter. Translation bare, PMP NAPOT over everything. Traps go to M; the
# handler counts non-ecall traps in x28/t3 and skips them; an ecall returns to M's continuation in x30/t5.
# S: 32 children by cderive from the root, child i over [page_a + 64 i, +64), each stored with stc (plain base) into
#   arr = page_a + 2048 + 16 i. x27/s11 = the root moved to arr. Then the direct loops, each 32 iterations of two
#   writes, unrolled, timed by rdcycle (the loop overhead is the li of the two values, outside the window):
#     D1 csrw sscratch, a; csrw sscratch, b          (control: a CSR write that flushes nothing)
#     D2 csrw srevroot, tbl2; csrw srevroot, tbl     (two namespace changes)
#     D3 csrw srevroot, tbl; csrw srevroot, tbl      (the same value: the pipeline flush, no namespace change)
#     D4 csrw satp, 0; csrw satp, 0                  (control: the other switch_mm CSR, also a flush)
# U (protected, srevroot = tbl), one pass = for each child: ldc it from arr through x27, then ld through it; timed by
#   rdcycle in U. Six entries, M doing between them:
#     P1 the first pass (everything cold)          P2 nothing (warm)
#     P3 csrw srevroot, tbl2; csrw srevroot, tbl   P4 nothing (warm)
#     P5 csrw srevroot, tbl (same value)           P6 csrw satp, 0; sfence.vma (a TLB flush, no namespace change)
# Prints: D1 D2 D3 D4 (cycles for 64 writes each), P1..P6 (cycles per pass of 32 ldc + 32 ld), the trap count.
# PRE-REGISTERED (2026-11-25, written before any run), b5b2-step9 at core = 904cd9c6b, in-tree, memory delay as the
# testlist's (the cycle VALUES are recorded, not predicted):
#   D2 > D1 and D3 > D1 (every srevroot write flushes the pipeline); D2 - D3 small against D3 (the namespace change adds
#   only a pulse in the core itself); D4 of the order of D3.
#   P3 > P2 (the cold revocation cache after a namespace change), P5 = P2 within a few cycles (same value: no namespace
#   flush, ex_stage.sv:1831), P6 >= P2 (DTLB refill only, bare translation: about P2). P4 = P2 within a few cycles.
#   Trap count 0 (every access is live and in bounds).
#   The per-switch cost read from this = (D2 - D1) / 64 per write for the direct part, (P3 - P2) per 32 distinct records
#   touched after the switch for the refill part. Sim memory delay is not silicon cost: the refill part scales with
#   memory latency on silicon.
# FIRST RUN (b8adefc78; repeated with +R43_TRACE, values identical): D1 0x584 D2 0xdb2 D3 0xde2 D4 0xdb2; P1 printed 0x8 (my
#   slip: x9 = s1 is the M handler's mcause; the trace reads 0x14c8 = 5,320), P2-P6 0x25b each; traps 0. VOID for the
#   refill question: every ld replayed once in EVERY pass (R43 "replay" 6 times at each ld pc), because every pass was
#   entered from M, and the table namespace is on in S and U with srevroot != 0 and off in M (csr_regfile.sv
#   vctx_table_o), so every M entry and exit empties the cache (SUP "ns-flush RISE vctx 0->1 / 1->0" around each
#   pass). The direct part (D1-D4) stands.
# SECOND CUT (this text, pre-registered before its run): the passes run U -> S -> U as Linux does. medeleg = 1 << 8 (a U
#   ecall traps to S), stvec = s_trap. s_trap stores the pass's cycles, does the next pass's action in S and srets to
#   u_pass. Eight passes:
#     P1 first   P2 nothing (a syscall round trip)   P3 csrw srevroot, tbl2; csrw srevroot, tbl (switch_mm, and back)
#     P4 nothing   P5 csrw srevroot, tbl (same value)   P6 csrw satp, 0; sfence.vma
#     P7 an ecall from S to M and back (an SBI call's shape; M returns at once)   P8 nothing
#   Prints: D1 D2 D3 D4, P1..P8, the trap count (M, non-ecall).
#   PRE-REGISTERED: P2 = P4 = P5 = P6 = P8 within a few cycles (warm: no namespace change on U <-> S, ex_stage.sv:1831;
#   no replay in them); P3 > P2 and P7 > P2, each by the cost of refilling the cache for the 33 ids the pass touches
#   (the root and 32 children), with about 32-33 replays each in the trace; P7 about P3. Traps 0. P1 recorded.
#   READ (ceffa440e, +R43_TRACE, rtl dirty=no, core = 904cd9c6b): D1 0x584 D2 0xdb2 D3 0xde2 D4 0xdb2 | P1 0x1572
#   P2 0x83 P3 0x25f P4 0x83 P5 0x83 P6 0x83 P7 0x25b P8 0x83 | count 0x8. As pre-registered except the count, which is
#   VOID (my slip: s_trap's temporary t3 is x28, the counter; s_done leaves 8 in it). The trace has exactly two M
#   entries, both ecall (cause 9), and 99 R43 replays = 33 in each of P1, P3 and P7, none in the others. Reading:
#   an srevroot write costs 55.5 cycles against an sscratch write's 22 (D3 / 64 vs D1 / 64), the same as a satp write,
#   whether or not the value changes; a CHANGE (and equally an M round trip) then costs one replay per distinct id
#   touched, (P3 - P2) / 33 = 14.4 cycles each at S12_MEM_DELAY 12.
# -DSC_COLD (2026-11-26, the record-cache decision's per-miss cost, the semantics owner's rule 2): before P3 S evicts the
#   D$ (256 KiB of loads at the line size, outside the timed window) and then changes the namespace as above; before P4
#   it evicts only, the control. P3 and P4 then pay the same cold data accesses, and P3 alone the 33 probes, whose
#   record reads now miss the D$. (P3 - P4) / 33 is the per-miss cost with the record in memory.
#   PRE-REGISTERED (written before any run, on b42c8c644 = R-80 + R-77's PRE fix, memory delay 12):
#     P4 > P2 (the eviction costs the pass its data); P3 > P4; (P3 - P4) / 33 above the warm 14.4 by about one
#     memory round trip per probe, so 20-60 cycles; P2 = P5 = P6 = P8 within a few cycles; traps 0x8 (the count's
#     known slip). The warm per-miss cost on this core is (P3 - P2) / 33 of the plain arm (srvcost, same tree).
#   READ (bd3932800, +R43_TRACE, rtl dirty=no, .sup/mc): plain D1-D4 0x591 0xdb2 0xde2 0xdb2 | P1 0x1572 P2 0x83 P3 0x25d
#     P4 0x83 P5 0x83 P6 0x83 P7 0x25a P8 0x83 | 0x8: warm per miss (P3 - P2) / 33 = 14.4 cycles. SC_COLD: D1-D4 the same |
#     P1 0x159e P2 0x83 P3 0x16a8 P4 0xda2 P5 0x83 P6 0x83 P7 0x9b9 P8 0xb8 | 0x8: cold per miss (P3 - P4) / 33 = 70.0
#     cycles, and independently (P7 - P2) / 33 = 71.5 (P7's probes read records P4's eviction pushed out; P5 and P6 hit the
#     verdict cache, so nothing refetched them). Graded: P4 > P2, P3 > P4 as predicted; the cost MISSED its 20-60 band
#     (70.0); P8 MISSED 'P2 within a few cycles' (0xb8 against 0x83; not traced).
#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define SREVROOT 0x5C1
#define NCH 32

#define CS_MINT_INTO(REG, LO, HI) \
  .pushsection .data; .balign 16; 77: .zero 64; .popsection; \
  lla a0, 77b; lla a3, LO; sd a3, 0(a0); lla a4, HI; sd a4, 8(a0); li a5, 0x10f; sd a5, 16(a0); \
  sd a3, 24(a0); sd x0, 32(a0); sd x0, 40(a0); addi a5, a0, 48; .insn r 0x5B, 0x1, 0x50, a3, a5, a0; LDC(REG, a5, 0)

/* 32 pairs of writes to a CSR from a6 and a7, between two rdcycles; the cycles in REG */
#define PAIRS(REG, CSR) \
  rdcycle a4; \
  .rept 32; csrw CSR, a6; csrw CSR, a7; .endr; \
  rdcycle a5; sub REG, a5, a4

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
csrw mcounteren, t0
csrw scounteren, t0
li t0, -1
srli t0, t0, 10
csrw pmpaddr0, t0
li t0, 0x1f
csrw pmpcfg0, t0

# ---------------------------------------------------------------- the table: b5b2-mtrap.S's
lla s8, tbl
li t0, 3
slli t0, t0, 62
ori t0, t0, 512
sd t0, 0(s8)
li t0, 7
sd t0, 8(s8)
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
lla t0, pg_rp1
sd t0, 8(t1)
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

# ---------------------------------------------------------------- S: the children, then the direct loops
lla t0, tbl
csrw SREVROOT, t0
li t0, 0x1800
csrc mstatus, t0
li t0, 0x800
csrs mstatus, t0
li t0, 0x100
csrw medeleg, t0                # a U ecall traps to S
lla t0, s_trap
csrw stvec, t0
lla t0, s_part
csrw mepc, t0
lla x30, m_after_s
mret
m_after_s:

lla t1, results
CAPPRINT(s7); CAPPRINT(s9); CAPPRINT(s10); CAPPRINT(x31)
.irp k, 0,1,2,3,4,5,6,7
ld t2, 8 * \k(t1)
CAPPRINT(t2)
.endr
csrw SREVROOT, x0
CAPPRINT(x28)
RVTEST_PASS

# ---------------------------------------------------------------- S
.align 4
s_part:
  li a1, 64
  li t1, 0                         # 64 i, the child's offset in page_a
  lla a4, page_a
  addi a4, a4, 2047
  addi a4, a4, 1                   # arr = page_a + 2048, an integer: stc through it is plain in S
  .rept NCH
  CINCOFFSET(t2, x26, t1)
  CDERIVE(a2, t2, a1)              # child i over [page_a + 64 i, +64)
  STC(a4, a2, 0)                   # arr[i] = child i
  addi a4, a4, 16
  addi t1, t1, 64
  .endr
  li t1, 2048
  CINCOFFSET(x27, x26, t1)
  li a6, 0x1234
  li a7, 0x5678
  PAIRS(s7, sscratch)
  lla a6, tbl2
  lla a7, tbl
  PAIRS(s9, SREVROOT)
  lla a6, tbl
  lla a7, tbl
  PAIRS(s10, SREVROOT)
  li a6, 0
  li a7, 0
  PAIRS(x31, satp)
  li s2, 0                         # passes done
  lla s3, results
  lla t3, u_pass
  csrw sepc, t3
  li t3, 0x100
  csrc sstatus, t3                 # SPP = U
  sret

# ---------------------------------------------------------------- S: the syscall side of each pass
.align 4
s_trap:
  sd a2, 0(s3)
  addi s3, s3, 8
  addi s2, s2, 1
  li t3, 8
  beq s2, t3, s_done
  li t3, 2
  bne s2, t3, 1f
#ifdef SC_COLD
  li a3, 0x80100000                # SC_COLD: evict the D$ (32 KiB, 8 ways, 16-byte lines) with 256 KiB of loads,
  li a4, 0x80140000                # 32 fills per set, outside the timed window
9:
  ld t4, 0(a3)
  addi a3, a3, 16
  bltu a3, a4, 9b
#endif
  lla t4, tbl2                     # before P3: switch_mm and back
  csrw SREVROOT, t4
  lla t4, tbl
  csrw SREVROOT, t4
  j s_go
1:
#ifdef SC_COLD
  li t3, 3
  bne s2, t3, 4f
  li a3, 0x80100000                # SC_COLD: evict the D$ (32 KiB, 8 ways, 16-byte lines) with 256 KiB of loads,
  li a4, 0x80140000                # 32 fills per set, outside the timed window
8:
  ld t4, 0(a3)
  addi a3, a3, 16
  bltu a3, a4, 8b
  j s_go                           # before P4: the eviction alone (SC_COLD's control)
4:
#endif
  li t3, 4
  bne s2, t3, 2f
  lla t4, tbl                      # before P5: the same value
  csrw SREVROOT, t4
  j s_go
2:
  li t3, 5
  bne s2, t3, 3f
  csrw satp, x0                    # before P6
  sfence.vma
  j s_go
3:
  li t3, 6
  bne s2, t3, s_go
  li a7, 0x5b1                     # before P7: an ecall to M and back
  ecall
  li a7, 0
s_go:
  lla t3, u_pass
  csrw sepc, t3
  sret
s_done:
  li a7, 0
  ecall

# ---------------------------------------------------------------- U: one pass
.align 4
u_pass:
  rdcycle a0
  .irp i, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
  LDC(t1, x27, 16 * \i)
  ld t2, 0(t1)
  .endr
  rdcycle a1
  sub a2, a1, a0
  ecall

.align 4
trap_handler:
  csrr x9, mcause
  li t0, 9
  bne x9, t0, 2f
  li t0, 0x5b1
  bne a7, t0, 1f
  csrr t0, mepc                    # the SBI-shaped call: return at once
  addi t0, t0, 4
  csrw mepc, t0
  mret
2:
  li t0, 8
  beq x9, t0, 1f
  addi x28, x28, 1
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
.balign 16
results:
  .zero 64
.align 12
tbl2:
  .zero 4096
RVTEST_DATA_END
