/* Stage-3 gp-captable entry glue (proof: 1 global `acc`, 32 bytes, .bss).
 *
 * Builds the per-global capability table from the DATA region (sp/cscratch) with
 * sizes BAKED as immediates (the RTL-safe pattern -- no runtime descriptor read,
 * which would hit the same execute-cap data-read the code-split gp did). Then
 * calls domain_main compiled with -capstone-gp-captable, which reaches acc via
 * `ldc gp[0]`. Generalises start-gpfree-captable.S to the call-graph app shape of
 * start-gpfree-cscratch.S. A build-time generator will later emit the BUILD block
 * from the .capstone_gp_table descriptor for arbitrary apps; here N=1, size[0]=32.
 */
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
#define split(rd, rs1, rs2) .insn r 0x5b, 0x1, 0x6, rd, rs1, rs2
#define delin(rd) .insn r 0x5b, 0x1, 0x3, rd, x0, x0
#define cscratch 0x004
#endif

.text

/* Build gp = N-entry cap-table over the DATA region, with per-global zeroed
   storage. Baked for this app: N=1, size[0]=32 (acc). Clobbers t1,t2. */
.macro BUILD_GP_CAPTABLE
  lcc(t1, sp, 4)                  /* t1 = sp.END */
  addi t1, t1, -16                /* reserve N*16 = 16 bytes for the cap-table */
  split(gp, sp, t1)               /* gp = [t1, END) = cap table; sp = [base, t1) */
  delin(gp)
  /* global 0: acc, 32 bytes */
  addi t1, t1, -32                /* reserve size[0] = 32 */
  split(t2, sp, t1)               /* t2 = [t1, gp.base) = acc storage (DATA cap) */
  delin(t2)
  sd x0, 0(t2)                    /* zero the 32 bytes */
  sd x0, 8(t2)
  sd x0, 16(t2)
  sd x0, 24(t2)
  stc(t2, gp, 0)                  /* cap-table[0] = acc cap */
  scc(sp, sp, t1)                 /* sp.cursor = top of remaining stack */
  delin(sp)
.endm

__test_reentry:
  ccsrrw(sp, cscratch, x0)
  BUILD_GP_CAPTABLE               /* re-derive gp + rebuild table on reentry */
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
  ccsrrw(sp, cscratch, x0)        /* sp = dom_data (cscratch) DATA region */
  BUILD_GP_CAPTABLE
  j test

.text
  .global test
test:
  cincoffsetimm(sp, sp, -96)
  stc(ra, sp, 48)                 /* save the entry return cap for domreturn */
  sd a0, 64(sp)
  stc(a1, sp, 80)                 /* a1 = shared region cap (res) */

  ldc(a0, sp, 80)                 /* a0 = res */
  ld a1, 64(sp)                   /* a1 = func */
  call domain_main

  lla t2, __test_reentry
  ldc(t1, sp, 48)                 /* t1 = entry return cap */
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

  .align 4
  .section .gct
  .global __capstone_gct_start
  .hidden __capstone_gct_start
__capstone_gct_start:
