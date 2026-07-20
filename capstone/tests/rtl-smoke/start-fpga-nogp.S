/* gp-free domain entry glue for the FPGA/RTL borrow-cost measurement.
 *
 * WHY THIS EXISTS (2026-07-20): the stock my_first_domain/start.S reaches
 * domain_main and every global via `gp` -- but our LLVM Capstone backend uses a
 * gp = PCC(cursor 0) convention that only works because our QEMU fork FABRICATES
 * that gp at every use site (capstone-qemu 7aca0540 / 39130dc1). Jason confirmed
 * the RTL never does this and gp=PCC(cursor 0) is not representable. So on silicon
 * `delin(gp)` stalls with gp=0. See history 20-07-2026 + the NDSS plan gp section.
 *
 * This entry glue removes gp entirely. It pairs with a GLOBAL-FREE domain
 * (borrow_cost_fpga_nogp.c): no module statics => our compiler emits no
 * `cincoffset gp`. Two facts from the reference make this work on real silicon:
 *   - plain `call` reaches domain_main within PCC (the reference monitor's own
 *     sbi_capstone.S uses `call handle_exception` &c -- no gp, no cjalr);
 *   - `split` carves a LINEAR scratch capability off the top of the delivered
 *     stack (exactly smode.dom.S's gp-carve) for measure_borrow to mrev/revoke.
 *
 * Entry hands domain_main three args: a0 = delivered region cap, a1 = func code,
 * a2 = the linear scratch cap. No gp is ever referenced.
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
/* linear scratch carved off the stack top for measure_borrow (bytes) */
#define CAP_SCRATCH_BYTES 256
#endif

.text

__test_reentry:
  ccsrrw(sp, cscratch, x0)
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
  ccsrrw(sp, cscratch, x0)        # sp = full (linear) stack capability
  lcc(t1, sp, 4)                  # t1 = stack END bound
  addi t3, t1, -CAP_SCRATCH_BYTES # t3 = END - scratch
  split(a2, sp, t3)               # a2 = [END-scratch, END] LINEAR; sp = [base, END-scratch]
  scc(sp, sp, t3)                 # sp.cursor = END-scratch (new stack top)
  delin(sp)                       # sp non-linear for ordinary locals
  # a2 = linear scratch cap, carried unclobbered into `test`
  j test

.text
  .global test
test:
  # NB: no delin(gp) and no capability-global-init loop -- the paired domain is
  # global-free, so the compiler emits no gp use and the init table is empty.
  cincoffsetimm(sp, sp, -96)
  stc(ra, sp, 48)
  sd a0, 64(sp)                   # func code (scalar) arrives in a0
  stc(a1, sp, 80)                 # delivered region capability arrives in a1
  stc(a2, sp, 32)                 # linear scratch capability

  # domain_main(void *region, unsigned func, void *lin_scratch)
  ldc(a0, sp, 80)                 # a0 = region cap
  ld a1, 64(sp)                   # a1 = func code
  ldc(a2, sp, 32)                 # a2 = linear scratch cap

  # Call domain_main WITHOUT gp, WITHOUT any capability jump. This is the
  # reference monitor's plain call/ret-within-PCC ABI (sbi_capstone.S uses plain
  # `call handle_exception` &c): a plain `call` keeps execution inside PCC and
  # sets ra to a SCALAR return address -- no gp, no code-capability formation.
  #
  # The one catch: our clang defaults to a capability return for domain_main
  # (`cjalr zero, 0(ra)`), which would require ra to be a capability. On a cscall
  # domain entry there is no gp-free way to form a code capability (PCC is not
  # readable and CTVEC arrives as an untagged scalar), so we instead retarget
  # that single return to a plain `ret` (jalr zero, 0(ra)) at the .s stage in
  # build-borrow-cost-fpga-nogp.sh -- see the comment there. Plain call + plain
  # ret both stay within PCC, exactly like the reference monitor.
  call domain_main                # plain call within PCC (scalar ra); no gp/cjalr

  lla t2, __test_reentry
  ldc(t1, sp, 48)                 # t1 = saved return (sealed-return) cap
  lcc(t0, sp, 4)                  # t0 = stack END
  scc(sp, sp, t0)                 # sp.cursor = END
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
