#ifndef __CAPSTONE_TEST_BUILTIN__
#define __CAPSTONE_TEST_BUILTIN__
#define cincoffset(rd, rs1, rs2) .insn r 0x5b, 0x1, 0xc, rd, rs1, rs2
#define cincoffsetimm(rd, rs, offset) .insn i 0x5b, 0x2, rd, offset(rs)
#define ldc(rd, rs, offset) .insn i 0x5b, 0x3, rd, offset(rs)
#define stc(rs1, rs2, offset) .insn s 0x5b, 0x4, rs1, offset(rs2)
#define domreturn(rd, rs1, rs2) .insn r 0x5b, 0x1, 0x21, rd, rs1, rs2
#define ccsrrw(rd, ccsr, rs) .insn i 0x5b, 0x7, rd, ccsr(rs)
#define lcc(rd, rs, imm) .insn r 0x5b, 0x1, 0x4, rd, rs, x##imm
#define scc(rd, rs1, rs2) .insn r 0x5b, 0x1, 0x5, rd, rs1, rs2
#define delin(rd) .insn r 0x5b, 0x1, 0x3, rd, x0, x0
#define cscratch 0x004
#endif

.text

__test_reentry:
  ccsrrw(sp, cscratch, x0)
  ldc(gp, sp, -16)
  ldc(t0, sp, -32)
  ccsrrw(x0, cscratch, t0)
  ld t0, -40(sp)
  csrrw x0, mcause, t0
  ld t0, -48(sp)
  csrrw x0, mtval, t0
  j test

.section .text.entry
  .global _start
_start:
__test_entry:
  ccsrrw(sp, cscratch, x0)
  lcc(t1, sp, 4)
  scc(sp, sp, t1)
  delin(sp)
  j test

.text
  .global test
test:
  delin(gp)
  cincoffsetimm(sp, sp, -96)
  stc(ra, sp, 48)
  sd a0, 64(sp)
  stc(a1, sp, 80)

  # Materialize capability globals in place before domain_main: a capability tag
  # cannot live in the static image, so initialized capability globals (pointer
  # tables, string tables) load untagged.  The compiler synthesizes a strong
  # __capstone_cap_init that stores them with tags at runtime; the weak no-op
  # default below is used when a domain has none.  Args a0/a1 are saved on the
  # stack above and reloaded for domain_main below; gp (the data/code root) is
  # callee-preserved.
  .Lpcrel_hi_cap_init:
  auipc t0, %pcrel_hi(__capstone_cap_init)
  addi t0, t0, %pcrel_lo(.Lpcrel_hi_cap_init)
  cincoffset t0, gp, t0
  .insn i 0x5b, 0x5, ra, 0(t0)

  # Call ordinary C compiled by our clang.
  ldc(a0, sp, 80)
  ld a1, 64(sp)
  .Lpcrel_hi_domain_main:
  auipc t0, %pcrel_hi(domain_main)
  addi t0, t0, %pcrel_lo(.Lpcrel_hi_domain_main)
  cincoffset t0, gp, t0
  .insn i 0x5b, 0x5, ra, 0(t0)

  lla t2, __test_reentry
  ldc(t1, sp, 48)
  lcc(t0, sp, 4)
  scc(sp, sp, t0)
  stc(gp, sp, -16)
  li ra, 0
  li gp, 0
  li tp, 0
  li t0, 0
  li s0, 0
  li s1, 0
  li a0, 0
  li a1, 0
  li a2, 0
  li a3, 0
  li a4, 0
  li a5, 0
  li a6, 0
  li a7, 0
  li s2, 0
  li s3, 0
  li s4, 0
  li s5, 0
  li s6, 0
  li s7, 0
  li s8, 0
  li s9, 0
  li s10, 0
  li s11, 0
  li t3, 0
  li t4, 0
  li t5, 0
  li t6, 0
  ccsrrw(t0, cscratch, x0)
  stc(t0, sp, -32)
  csrrw t0, mcause, x0
  sd t0, -40(sp)
  csrrw t0, mtval, x0
  sd t0, -48(sp)
  li t0, 0
  ccsrrw(x0, cscratch, sp)
  li sp, 0
  domreturn(t1, t2, x0)

.Ltest_ret:
  j .Ltest_ret

# Weak no-op default for __capstone_cap_init: linked when a domain has no
# capability globals.  When the compiler emits capability-global initialization,
# its strong __capstone_cap_init overrides this (weak < strong at link time).
.text
  .weak __capstone_cap_init
  .type __capstone_cap_init, @function
__capstone_cap_init:
  cjalr zero, 0(ra)

  .align 4
  .section .gct
  .global __capstone_gct_start
  .hidden __capstone_gct_start
__capstone_gct_start:
