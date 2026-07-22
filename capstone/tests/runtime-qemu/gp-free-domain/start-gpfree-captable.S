/* gp-free domain entry glue — capstone-c-style DATA-authority gp (silicon fix probe).
 *
 * WHY: on captype-fixed CVA6 the previous approach (gp = __split of the *code*
 * image, delivered via a cscratch memory slot) wedges the hart the instant the
 * domain does any global access (proven: bare `scc gp; delin; lw` faults). The
 * capstone-c reference NEVER data-accesses the code image: it derives gp from
 * `sp` (the cscratch DATA region), builds a per-global cap-table, and each global
 * lives in DATA memory with its own data capability (codegen.rs:264-298).
 *
 * This glue mirrors that model to validate the mechanism on silicon with a single
 * global (value 105 = 0x69):
 *   - sp = cscratch (dom_data, a DATA cap — the same authority the stack uses);
 *   - carve a 1-entry cap-table off the top of sp;  gp = that table;
 *   - carve one data word off sp for the global, store 105 through that DATA cap,
 *     put the cap in the table (gp[0]);
 *   - remainder of sp = stack.
 * The workload reads the global back through the cap-table (ldc gp[0]; lw) and
 * writes 0x2110C000 | (105 & 0xff) = 0x2110C069 to the shared region cap (res),
 * then domreturns. NO code-image split, NO gp memory round-trip.
 *
 * The monitor is UNCHANGED: it still stashes its (unused) code-split gp and
 * delivers dom_data via cscratch; we ignore the stashed gp. The image is padded
 * >0x1000 (link-gpfree.ld + a pad global) so the monitor's fixed base+0x1000
 * SPLIT stays non-degenerate and create_domain returns.
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

/* cap field indices (LccField): Cursor=2, Base=3, End=4 */

.text

/* Build gp (cap-table over the DATA region) + one global. Clobbers t1,t2,t3.
   On exit: gp = 1-entry cap-table (non-linear), sp = remaining stack, and
   gp[0] holds a data cap over a word containing 105. */
.macro BUILD_GP_CAPTABLE
  lcc(t1, sp, 4)                  /* t1 = sp.END */
  addi t1, t1, -16                /* reserve one 16B cap-table slot at the top */
  split(gp, sp, t1)               /* gp = [t1, END) = cap table; sp = [base, t1) */
  delin(gp)
  addi t1, t1, -16                /* reserve one 16B data word for the global */
  split(t2, sp, t1)               /* t2 = [t1, gp.base) DATA cap for the global */
  delin(t2)
  li t3, 105                      /* the global's value (0x69) */
  sw t3, 0(t2)                    /* store it through the DATA cap (data authority) */
  stc(t2, gp, 0)                  /* cap-table[0] = t2 */
  scc(sp, sp, t1)                 /* sp.cursor = t1 (top of remaining stack) */
  delin(sp)
.endm

__test_reentry:
  ccsrrw(sp, cscratch, x0)
  BUILD_GP_CAPTABLE               /* re-derive gp fresh on reentry (like capstone-c) */
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

  /* workload: read the global via the cap-table, compute result, write to res */
  ldc(t2, gp, 0)                  /* t2 = global's data cap from cap-table */
  lw t3, 0(t2)                    /* t3 = 105 */
  lui t4, 0x2110c
  or  t3, t3, t4                  /* 0x2110c069 */
  ldc(t2, sp, 80)                 /* t2 = res (region cap) */
  sw t3, 0(t2)                    /* *res = 0x2110c069 */

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
