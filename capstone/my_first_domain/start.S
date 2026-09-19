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
#define split(rd, rs1, rs2) .insn r 0x5b, 0x1, 0x6, rd, rs1, rs2
#define ctvec 0x000
#define cscratch 0x004
#endif

#include "../runtime/include/capstone/domain-fault.h"

.text

__test_reentry:
#ifdef CAPSTONE_DOMAIN_FAULT_RECOVERY
  ccsrrw(t0, cscratch, x0)
  ldc(sp, t0, 48)
  ldc(gp, t0, 32)
  ccsrrw(x0, cscratch, t0)
#else
  ccsrrw(sp, cscratch, x0)
  ldc(gp, sp, -16)
  ldc(t0, sp, -32)
  ccsrrw(x0, cscratch, t0)
  ld t0, -40(sp)
  csrrw x0, mcause, t0
  ld t0, -48(sp)
  csrrw x0, mtval, t0
#endif
  j test

.section .text.entry
  .global _start
_start:
__test_entry:
  ccsrrw(sp, cscratch, x0)
#ifdef CAPSTONE_DOMAIN_FAULT_RECOVERY
  # The upper 256 bytes are recovery state, OUTSIDE the application's stack
  # bounds. cscratch owns the route to it even if application SP/GP is damaged.
  # Slots: 0 sealed return; 16 CALL result pointer; 32 gp; 48 stack top; 64 DPI.
  lcc(t1, sp, 4)
  addi t1, t1, -256
  split(t2, sp, t1)
  delin(t2)
  ccsrrw(x0, cscratch, t2)
#endif
  lcc(t1, sp, 4)
  scc(sp, sp, t1)
  delin(sp)
#ifdef CAPSTONE_DOMAIN_FAULT_RECOVERY
  stc(sp, t2, 48)
  li t2, 0
#endif
  j test

.text
  .global test
test:
  delin(gp)
  cincoffsetimm(sp, sp, -96)
#ifdef CAPSTONE_DOMAIN_FAULT_RECOVERY
  ccsrrw(t2, cscratch, x0)
  stc(ra, t2, 0)
  li ra, 0
  stc(gp, t2, 32)
  sd a0, 64(t2)
  # REGION_SHARE passes a region, not a result pointer. Never write through it
  # on a fault. A share fault quarantines the domain; its next CALL reports it.
  bnez a0, .Lfault_no_result
  stc(a1, t2, 16)
.Lfault_no_result:
  ccsrrw(x0, cscratch, t2)
  li t2, 0
  lla t0, .Ldomain_fault
  cincoffset(t0, gp, t0)
  # Execute-only is the QEMU local-recovery opt-in. Legacy monitor RWX trap
  # vectors assume S/U-mode state and must not receive internal C-mode faults.
  .insn r 0x5b, 0x1, 0x2, t0, t0, x1
  ccsrrw(x0, ctvec, t0)
  li t0, 0
#else
  stc(ra, sp, 48)
#endif
  sd a0, 64(sp)
  stc(a1, sp, 80)

  # Run capability-global initializers before domain_main. A capability tag
  # cannot live in the static image, so initialized capability globals (pointer/
  # string/function tables) load untagged; the compiler emits one initializer per
  # such module plus a PC-relative entry in the .capstone_cap_init table (each
  # entry = initfn - entry). The domain is loaded at a runtime base and processes
  # no load-time relocations, so we resolve each initializer's runtime address as
  # entry_runtime + offset and derive a code capability via cincoffset gp. The
  # current/end table addresses are kept in the frame across calls (initializers
  # clobber t/a regs but preserve gp/sp); a0/a1 are reloaded for domain_main
  # below. Empty table (start == end) is a no-op.
  .Lpcrel_init_start:
  auipc t0, %pcrel_hi(__capstone_cap_init_start)
  addi t0, t0, %pcrel_lo(.Lpcrel_init_start)
  sd t0, 0(sp)
  .Lpcrel_init_end:
  auipc t1, %pcrel_hi(__capstone_cap_init_end)
  addi t1, t1, %pcrel_lo(.Lpcrel_init_end)
  sd t1, 8(sp)
.Lcap_init_loop:
  ld t0, 0(sp)
  ld t1, 8(sp)
  bgeu t0, t1, .Lcap_init_done
  cincoffset t2, gp, t0          # cap to current table entry
  ld t3, 0(t2)                   # t3 = initfn - entry (link-time relative)
  add t3, t0, t3                 # target_runtime = entry_runtime + offset
  cincoffset t2, gp, t3          # code capability to the initializer
  .insn i 0x5b, 0x5, ra, 0(t2)   # call it
  ld t0, 0(sp)
  addi t0, t0, 8                 # next 8-byte entry
  sd t0, 0(sp)
  j .Lcap_init_loop
.Lcap_init_done:

  # Call ordinary C compiled by our clang.
  ldc(a0, sp, 80)
  ld a1, 64(sp)
  .Lpcrel_hi_domain_main:
  auipc t0, %pcrel_hi(domain_main)
  addi t0, t0, %pcrel_lo(.Lpcrel_hi_domain_main)
  cincoffset t0, gp, t0
  .insn i 0x5b, 0x5, ra, 0(t0)

  lla t2, __test_reentry
#ifdef CAPSTONE_DOMAIN_FAULT_RECOVERY
  ccsrrw(t0, cscratch, x0)
  ldc(t1, t0, 0)
  stc(x0, t0, 0)
  ccsrrw(x0, cscratch, t0)
  j .Ldomain_scrub_return
#else
  ldc(t1, sp, 48)
  lcc(t0, sp, 4)
  scc(sp, sp, t0)
  stc(gp, sp, -16)
#endif
.Ldomain_scrub_return:
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
#ifndef CAPSTONE_DOMAIN_FAULT_RECOVERY
  ccsrrw(t0, cscratch, x0)
  stc(t0, sp, -32)
  csrrw t0, mcause, x0
  sd t0, -40(sp)
  csrrw t0, mtval, x0
  sd t0, -48(sp)
  li t0, 0
  ccsrrw(x0, cscratch, sp)
#endif
  li sp, 0
  domreturn(t1, t2, x0)

#ifdef CAPSTONE_DOMAIN_FAULT_RECOVERY
.Ldomain_fault:
  # No C calls and no use of the faulting stack or global pointer. A second
  # fault in recovery retains QEMU's fail-stop behavior instead of recursing.
  ccsrrw(x0, ctvec, x0)
  ccsrrw(sp, cscratch, x0)
  ld t0, 64(sp)
  bnez t0, .Ldomain_fault_return
  ldc(t0, sp, 16)
  li t3, CAPSTONE_DOMAIN_FAULT_RETVAL
  sd t3, 0(t0)
.Ldomain_fault_return:
  ldc(t1, sp, 0)
  stc(x0, sp, 0)
  ccsrrw(x0, cscratch, sp)
  lla t2, .Ldomain_faulted_reentry
  j .Ldomain_scrub_return

.Ldomain_faulted_reentry:
  # Fresh caller, but a permanently quarantined client. Neither global
  # initializers nor domain_main may execute after a fault.
  bnez a0, .Ldomain_faulted_return
  li t0, CAPSTONE_DOMAIN_FAULT_RETVAL
  sd t0, 0(a1)
.Ldomain_faulted_return:
  .insn r 0x5b, 0x1, 0xa, t1, ra, x0
  li ra, 0
  lla t2, .Ldomain_faulted_reentry
  j .Ldomain_scrub_return
#endif

.Ltest_ret:
  j .Ltest_ret

  .align 4
  .section .gct
  .global __capstone_gct_start
  .hidden __capstone_gct_start
__capstone_gct_start:
