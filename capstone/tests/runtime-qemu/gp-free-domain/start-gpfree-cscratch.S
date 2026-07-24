/* gp-free domain entry glue with REAL cscratch gp delivery (silicon-ready).
 *
 * The monitor (create_domain) mints an image-covering data cap for gp and stores
 * it at the top-16 slot of the domain's stack (cscratch) region. This glue loads
 * it (`ldc gp, END-16`) on first entry and delin's it; domain_main uses `scc gp`
 * for globals. No cincoffset gp, no cjalr, no cursor-0, no QEMU fabrication --
 * gp comes entirely from the monitor via cscratch (the board owner's confirmed
 * channel; same as capstone-c). Reentry reloads the saved (non-linear) gp.
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
#define delin(rd) .insn r 0x5b, 0x1, 0x3, rd, x0, x0
#define cscratch 0x004
#endif

.text

__test_reentry:
  ccsrrw(sp, cscratch, x0)
  ldc(gp, sp, -16)                # reload saved (already non-linear) gp
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
  ccsrrw(sp, cscratch, x0)        # sp = stack (cscratch = dom_data) region
  lcc(t1, sp, 4)                  # t1 = END
  scc(sp, sp, t1)                 # sp.cursor = END
  ldc(gp, sp, -16)                # gp = monitor-delivered image cap @ END-16
  delin(gp)                       # make gp non-linear for scc-based global loads
  delin(sp)
  j test

.text
  .global test
test:
  # gp is already loaded + non-linear (both entry paths); no delin(gp) here.
  cincoffsetimm(sp, sp, -96)
  stc(ra, sp, 48)
  sd a0, 64(sp)
  stc(a1, sp, 80)

  # (No capability-global initializer loop: integer app, empty table.)

  # Call domain_main with a PLAIN call within PCC -- gp-free, no cjalr.
  ldc(a0, sp, 80)
  ld a1, 64(sp)
  call domain_main

  lla t2, __test_reentry
  ldc(t1, sp, 48)
  lcc(t0, sp, 4)
  scc(sp, sp, t0)
  stc(gp, sp, -16)                # save (non-linear) gp for reentry @ END-16
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
