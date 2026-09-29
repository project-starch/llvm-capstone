/* Domain entry glue for the musl application runtime (the delegated runtime,
 * docs/plans/delegation-abi.md).
 *
 * This is my_first_domain/start.S with the fault-recovery block it has under
 * CAPSTONE_DOMAIN_FAULT_RECOVERY, plus a RESUMABLE yield and the entries of
 * minted contexts. The shared start.S is left byte-identical on purpose --
 * every existing domain and every silicon ladder rung depends on it, and a
 * change there would put a working measurement at risk to buy nothing.
 *
 * WHY A SEPARATE YIELD IS NEEDED AT ALL
 *
 * The shared glue returns with `domreturn(t1, __test_reentry, x0)`, so the next
 * CALL into the domain RESTARTS `test`: cap-globals are re-initialised and
 * domain_main is entered again from the top. That is right for the SQLite
 * shape, where each entry is a separate errand (stash region #1, stash region
 * #2, run the database). It is useless for a syscall: `write()` must return to
 * the instruction after itself, with the C stack intact.
 *
 * The ISA already provides this. RETURN sets `pc.cursor` to x[rs1] and only
 * then swaps the C-effective registers out to the domain's context block, so
 * x[rs1] is where THIS domain resumes on its next CALL, and the whole register
 * file -- sp included -- is restored by the hardware swap. Passing a label
 * inside the yield routine instead of __test_reentry is the entire difference.
 *
 * The bare HostCall wire probes (tests/runtime-qemu/hostcall-*-probe) get
 * resumption a different way: they run an S-mode payload inside a nested
 * domain (create_dom(sbi.dom, x.smode)) and yield with an SBI DOM_RETURN
 * ecall. That path is not pure-capability, which is why it cannot carry musl.
 *
 * NO SCALAR STATUS IS RETURNED. The glue zeroes every register before
 * returning (capability-leak hygiene), so a domain's `call_dom` result is
 * always 0. What the launcher reads is the result slot of the entry it made
 * (recovery-block slot 16, CAPSTONE_DOMAIN_FAULT_RETVAL after a fault) and the
 * delegated entry block (delegate.h), never a return value.
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
#define split(rd, rs1, rs2) .insn r 0x5b, 0x1, 0x6, rd, rs1, rs2
#define ctvec 0x000
#define cscratch 0x004
#endif

#include "../../../runtime/include/capstone/domain-fault.h"
#include "../../../runtime/include/capstone/context.h"

.text

__test_reentry:
  ccsrrw(t0, cscratch, x0)
  ldc(sp, t0, 48)
  ldc(gp, t0, 32)
  ccsrrw(x0, cscratch, t0)
  j test

.section .text.entry
  .global _start
_start:
__test_entry:
  ccsrrw(sp, cscratch, x0)
  /* Keep recovery state outside the application's stack. cscratch always
     holds this context, including across the yields of delegated calls. Slots match the
     generic entry, with suspended SP added at 80. */
  lcc(t1, sp, 4)
  addi t1, t1, -256
  split(t2, sp, t1)
  delin(t2)
  ccsrrw(x0, cscratch, t2)
  /* Context arena (docs/plans/delegation-threads.md). An application built
     with CONTEXT_BYTES gets the lowest bytes of its data region as ONE LINEAR
     capability, parked in __capstone_context_arena; the runtime carves the
     thread areas it seals out of it (context.c). This is the only point at
     which the data region is still linear. With 0 bytes, which is every other
     application, the region below is split exactly as before. */
  delin(gp)
  .Lpcrel_arena_bytes:
  auipc t3, %pcrel_hi(__capstone_context_arena_bytes)
  addi t3, t3, %pcrel_lo(.Lpcrel_arena_bytes)
  cincoffset t3, gp, t3
  ld t4, 0(t3)
  beqz t4, .Lno_context_arena
  lcc(t1, sp, 3)
  add t1, t1, t4
  split(t5, sp, t1)              /* sp: the arena, t5: the stack above it */
  .Lpcrel_arena_slot:
  auipc t3, %pcrel_hi(__capstone_context_arena)
  addi t3, t3, %pcrel_lo(.Lpcrel_arena_slot)
  cincoffset t3, gp, t3
  stc(sp, t3, 0)                 /* moves the linear arena out of sp */
  .insn r 0x5b, 0x1, 0xa, sp, t5, x0   /* movc sp, t5 */
.Lno_context_arena:
  li t3, 0
  li t4, 0
  lcc(t1, sp, 4)
  scc(sp, sp, t1)
  delin(sp)
  stc(sp, t2, 48)
  li t2, 0
  j test

.text
  .global test
test:
  delin(gp)
  cincoffsetimm(sp, sp, -96)
  ccsrrw(t2, cscratch, x0)
  stc(ra, t2, 0)
  li ra, 0
  stc(gp, t2, 32)
  sd a0, 64(t2)
  bnez a0, .Lmusl_no_result
  stc(a1, t2, 16)
.Lmusl_no_result:
  ccsrrw(x0, cscratch, t2)
  li t2, 0
  lla t0, .Lmusl_fault
  cincoffset(t0, gp, t0)
  .insn r 0x5b, 0x1, 0x2, t0, t0, x1
  ccsrrw(x0, ctvec, t0)
  li t0, 0
  sd a0, 64(sp)
  stc(a1, sp, 80)

  /* The sealed-return capability `ra` holds on EVERY entry went into the
     recovery block above (slot 0), with the result slot (16) and gp (32):
     the yield and the fault handler need them from arbitrary depth inside
     libc, and cscratch finds that block from anywhere. */

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
  ccsrrw(t0, cscratch, x0)
  ldc(t1, t0, 0)
  stc(x0, t0, 0)
  ccsrrw(x0, cscratch, t0)
.Lmusl_scrub_return:
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
  li sp, 0
  domreturn(t1, t2, x0)

.Lmusl_fault:
  ccsrrw(x0, ctvec, x0)
  ccsrrw(sp, cscratch, x0)
  ld t0, 64(sp)
  bnez t0, .Lmusl_fault_return
  ldc(t0, sp, 16)
  li t3, CAPSTONE_DOMAIN_FAULT_RETVAL
  sd t3, 0(t0)
.Lmusl_fault_return:
  ldc(t1, sp, 0)
  stc(x0, sp, 0)
  ccsrrw(x0, cscratch, sp)
  lla t2, .Lmusl_faulted_reentry
  j .Lmusl_scrub_return

.Lmusl_faulted_reentry:
  bnez a0, .Lmusl_faulted_return
  li t0, CAPSTONE_DOMAIN_FAULT_RETVAL
  sd t0, 0(a1)
.Lmusl_faulted_return:
  .insn r 0x5b, 0x1, 0xa, t1, ra, x0
  li ra, 0
  lla t2, .Lmusl_faulted_reentry
  j .Lmusl_scrub_return

/* ---- Minted contexts (docs/plans/delegation-threads.md) ------------------
 *
 * A context the runtime mints (context.c) enters here on its first CALL. Its
 * cscratch is its own start block, laid out like the recovery block above:
 * slots 0..80 are the entry glue's (return capability, result slot, gp, sp,
 * request, suspended sp), so the yield and the fault handler above serve a
 * minted context unchanged. Slots 96..128 carry tp, the start function and
 * its argument; 160 is the completion word and 168 the return value
 * (include/capstone/context.h). The capability-global initializers never run
 * here: they ran once, on the application's first entry. */
  .global __capstone_context_entry
  .balign 16
__capstone_context_entry:
  ccsrrw(t0, cscratch, x0)
  stc(ra, t0, 0)
  li ra, 0
  delin(gp)
  stc(gp, t0, 32)
  sd a0, 64(t0)
  bnez a0, .Lcontext_no_result
  stc(a1, t0, 16)
.Lcontext_no_result:
  lla t1, .Lmusl_fault
  cincoffset(t1, gp, t1)
  .insn r 0x5b, 0x1, 0x2, t1, t1, x1
  ccsrrw(x0, ctvec, t1)
  ldc(sp, t0, 48)
  ldc(tp, t0, 96)
  ldc(t1, t0, 112)
  ldc(a0, t0, 128)
  ccsrrw(x0, cscratch, t0)
  li t0, 0
  .insn i 0x5b, 0x5, ra, 0(t1)
  /* The start function returned: its value is the context's. */

/* void __capstone_context_exit(unsigned long value), never returns.
 *
 * Everything the final switch needs is loaded into registers BEFORE the
 * completion word is published; after that store nothing touches the stack,
 * the TLS block or the start block, no libc function runs and the syscall
 * dispatcher is never entered, so no signal is delivered here. A joiner that
 * sees done == 1 may revoke the thread area at once. */
  .global __capstone_context_exit
__capstone_context_exit:
  ccsrrw(t0, cscratch, x0)
  ldc(t1, t0, 0)
  stc(x0, t0, 0)
  ld t3, 64(t0)
  ldc(t2, t0, 16)
  sd a0, 168(t0)
  fence rw, w
  j .Lcontext_exit_done

/* void __capstone_context_exit_clear(unsigned long value, volatile int *clear),
 * never returns: as __capstone_context_exit, and before the completion word it
 * stores 0 into *clear, the word Linux clears when a thread made with
 * CLONE_CHILD_CLEARTID ends (musl's thread-list lock). A thread that joined
 * may free this context's stack and TLS from that store on, so it comes after
 * the last access to them; the launcher wakes the word's waiters once the
 * context has returned for good (CONTEXT_EXITING). */
  .global __capstone_context_exit_clear
__capstone_context_exit_clear:
  ccsrrw(t0, cscratch, x0)
  ldc(t1, t0, 0)
  stc(x0, t0, 0)
  ld t3, 64(t0)
  ldc(t2, t0, 16)
  sd a0, 168(t0)
  fence rw, w
  sw x0, 0(a1)
  fence w, w
.Lcontext_exit_done:
  li t4, 1
  sd t4, 160(t0)
  /* No access to the thread area below this line. */
  ccsrrw(x0, cscratch, t0)
  li t0, 0
  bnez t3, .Lcontext_exit_switch
  li t4, CAPSTONE_CONTEXT_EXITED
  sd t4, 0(t2)
.Lcontext_exit_switch:
  li t2, 0
  li t3, 0
  li t4, 0
  li sp, 0
  li gp, 0
  li tp, 0
  li s0, 0
  li s1, 0
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
  li a0, 0
  li a1, 0
  li a2, 0
  li a3, 0
  li a4, 0
  li a5, 0
  li a6, 0
  li a7, 0
  li t5, 0
  li t6, 0
  lla t2, .Lcontext_reentry
  domreturn(t1, t2, x0)

/* Every later entry into an exited context lands here. It uses only the
 * return capability and result slot this entry delivered, reports EXITED
 * again and returns: no stack, no TLS, no start block. */
.Lcontext_reentry:
  bnez a0, .Lcontext_reentry_switch
  li t4, CAPSTONE_CONTEXT_EXITED
  sd t4, 0(a1)
.Lcontext_reentry_switch:
  li a0, 0
  li a1, 0
  li t4, 0
  lla t2, .Lcontext_reentry
  domreturn(ra, t2, x0)

/* void __capstone_context_seal(capstone_cap_slot *region, void *entry,
 *                              void *start_block, capstone_cap_slot *out,
 *                              unsigned long mstatus, unsigned long mie)
 *
 * Fill a LINEAR seal region with the synchronous context a first CALL loads
 * (pc, ctvec, cscratch, then the control words) and seal it into *out. The
 * runtime passes the words create_domain writes; a probe may pass others
 * (A12). ctvec stays null: the entry installs the runtime's own. */
  .global __capstone_context_seal
__capstone_context_seal:
  ldc(t0, a0, 0)
  stc(x0, a0, 0)
  stc(a1, t0, 0)
  stc(x0, t0, 16)
  stc(a2, t0, 32)
  sd a4, 48(t0)
  sd x0, 56(t0)
  sd x0, 64(t0)
  sd x0, 72(t0)
  sd a5, 80(t0)
  .insn r 0x5b, 0x1, 0x7, t0, t0, x0   /* seal t0 in place */
  stc(t0, a3, 0)
  li t1, 0
  ret

/* long __capstone_context_offer(capstone_cap_slot *seal, unsigned long ticket)
 *
 * Move a minted seal, with its ticket, into the invocation descriptor the
 * current call entry lent this context (recovery-block slot 16). The monitor
 * takes it out when the call returns, before it revokes the loan; the launcher
 * then asks it to ADOPT the offer under the ticket. -1 when the current entry
 * carried no descriptor (a region share); the seal then stays where it is. */
  .global __capstone_context_offer
__capstone_context_offer:
  ccsrrw(t0, cscratch, x0)
  ld t1, 64(t0)
  bnez t1, .Lcontext_offer_none
  ldc(t2, t0, 16)
  ccsrrw(x0, cscratch, t0)
  ldc(t3, a0, 0)
  stc(x0, a0, 0)
  sd a1, 16(t2)
  stc(t3, t2, 32)
  li t0, 0
  li t2, 0
  li t3, 0
  li a0, 0
  ret
.Lcontext_offer_none:
  ccsrrw(x0, cscratch, t0)
  li t0, 0
  li a0, -1
  ret

/* void __capstone_context_call(capstone_cap_slot *seal, unsigned long request,
 *                              unsigned long *result)
 *
 * Enter a minted context directly with CALL (a nested, unsupervised entry
 * from this context) and return when it yields or exits; the seal comes back
 * into *seal. The CALL exchanges only pc, ctvec, cscratch and the control
 * words, so everything this caller needs is saved on its stack and sp
 * travels in the recovery block's suspended-sp slot, as in the yield. */
  .global __capstone_context_call
__capstone_context_call:
  cincoffsetimm(sp, sp, -256)
  stc(ra, sp, 0)
  stc(gp, sp, 16)
  stc(s0, sp, 32)
  stc(s1, sp, 48)
  stc(s2, sp, 64)
  stc(s3, sp, 80)
  stc(s4, sp, 96)
  stc(s5, sp, 112)
  stc(s6, sp, 128)
  stc(s7, sp, 144)
  stc(s8, sp, 160)
  stc(s9, sp, 176)
  stc(s10, sp, 192)
  stc(s11, sp, 208)
  stc(tp, sp, 224)
  stc(a0, sp, 240)
  ccsrrw(t0, cscratch, x0)
  stc(sp, t0, 80)
  ccsrrw(x0, cscratch, t0)
  ldc(t3, a0, 0)
  stc(x0, a0, 0)
  mv a0, a1
  .insn r 0x5b, 0x1, 0xa, a1, a2, x0   /* movc a1, a2 */
  /* Named so that a probe can match a fault at exactly this CALL. */
  .global __capstone_context_call_insn
__capstone_context_call_insn:
  .insn r 0x5b, 0x1, 0x20, t3, t3, x0  /* call t3 */
  ccsrrw(t0, cscratch, x0)
  ldc(sp, t0, 80)
  ccsrrw(x0, cscratch, t0)
  li t0, 0
  ldc(a0, sp, 240)
  stc(t3, a0, 0)
  ldc(ra, sp, 0)
  ldc(gp, sp, 16)
  ldc(s0, sp, 32)
  ldc(s1, sp, 48)
  ldc(s2, sp, 64)
  ldc(s3, sp, 80)
  ldc(s4, sp, 96)
  ldc(s5, sp, 112)
  ldc(s6, sp, 128)
  ldc(s7, sp, 144)
  ldc(s8, sp, 160)
  ldc(s9, sp, 176)
  ldc(s10, sp, 192)
  ldc(s11, sp, 208)
  ldc(tp, sp, 224)
  cincoffsetimm(sp, sp, 256)
  ret

.Ltest_ret:
  j .Ltest_ret

/* void __capstone_yield(void);
 *
 * Hand control back to the launcher and resume HERE when it calls again.
 *
 * SP DOES NOT SURVIVE THE BOUNDARY. The first version of this routine assumed
 * the hardware register-file swap restored the whole C state, saved only `ra`,
 * and MEASURED (QEMU, 2026-08-14) exactly one thing on resume:
 *
 *   [CAPSTONE] Cap mem access requires capability: pc = ..., rs1 = x2, imm = 0
 *   [CAPSTONE] domain halted by capability fault: cause = 24
 *
 * `rs1 = x2` is sp, at the first sp-relative `ldc` after resume: sp came back
 * as a plain integer, not a capability. `gp` did survive -- the gp-relative
 * access two instructions earlier did not fault -- which is consistent with the
 * monitor handing gp to the domain on every entry.
 *
 * cscratch is the designated carrier for a capability across a domain
 * boundary, and in this glue it always holds the context's recovery block (see
 * _start). So: sp travels in that block (slot 80), the return capability comes
 * from its slot 0, and everything else travels on the stack that sp then finds
 * again. The resuming CALL supplies a new return capability and result slot,
 * which the resume path installs before it touches the stack.
 *
 * Callee-saved registers are saved even though `gp` suggests some registers do
 * survive. Which ones is unverified, one wrong guess costs a ten-minute boot to
 * discover, and 14 stores are free by comparison.
 *
 * Deliberately does NOT scrub the register file. The shared glue scrubs because
 * it is finished and must not leave capabilities behind; here the registers ARE
 * the suspended computation.
 */
  .global __capstone_yield
__capstone_yield:
  cincoffsetimm(sp, sp, -256)
  stc(ra, sp, 0)
  stc(gp, sp, 16)
  stc(s0, sp, 32)
  stc(s1, sp, 48)
  stc(s2, sp, 64)
  stc(s3, sp, 80)
  stc(s4, sp, 96)
  stc(s5, sp, 112)
  stc(s6, sp, 128)
  stc(s7, sp, 144)
  stc(s8, sp, 160)
  stc(s9, sp, 176)
  stc(s10, sp, 192)
  stc(s11, sp, 208)
  /* THE THREAD POINTER IS PART OF THE SUSPENDED COMPUTATION, and it is the one
     register here that is neither callee-saved by the C ABI nor handed back by
     the monitor. musl keeps errno, the locale and the cancellation state behind
     it, so a domain that loses it across a yield works until the first syscall
     that FAILS and then faults in __errno_location instead of returning -1.
     Measured 2026-09-16: write(1,...) completed through this yield, and the
     write(7,...) after it halted with cause = 24 at `cincoffsetimm a0, a0, -0x124`,
     the pointer arithmetic in __pthread_self, with tp reading 0. Slot 224 was
     free; the frame is 256 bytes and 0..208 were in use. */
  stc(tp, sp, 224)

  ccsrrw(t0, cscratch, x0)
  stc(sp, t0, 80)
  ldc(t1, t0, 0)
  stc(x0, t0, 0)
  ccsrrw(x0, cscratch, t0)
  lla t2, .Lyield_resume
  domreturn(t1, t2, x0)

.Lyield_resume:
  /* The resuming CALL supplies a NEW result pointer and return capability.
     Install both before touching the suspended application's stack. */
  ccsrrw(t0, cscratch, x0)
  stc(ra, t0, 0)
  li ra, 0
  stc(a1, t0, 16)
  sd a0, 64(t0)
  ldc(sp, t0, 80)
  ldc(gp, t0, 32)
  ccsrrw(x0, cscratch, t0)

  ldc(ra, sp, 0)
  ldc(s0, sp, 32)
  ldc(s1, sp, 48)
  ldc(s2, sp, 64)
  ldc(s3, sp, 80)
  ldc(s4, sp, 96)
  ldc(s5, sp, 112)
  ldc(s6, sp, 128)
  ldc(s7, sp, 144)
  ldc(s8, sp, 160)
  ldc(s9, sp, 176)
  ldc(s10, sp, 192)
  ldc(s11, sp, 208)
  ldc(tp, sp, 224)
  cincoffsetimm(sp, sp, 256)
  ret

  .align 4
  .section .gct
  .global __capstone_gct_start
  .hidden __capstone_gct_start
__capstone_gct_start:

  .section .data
/* The linear context arena _start splits off (see there); empty without
   CONTEXT_BYTES. */
  .align 4
  .global __capstone_context_arena
__capstone_context_arena:
  .zero 16

/* musl's cancellation-point bounds (src/thread/pthread_cancel.c compares a
 * handler's interrupted pc against them). A delegated call is not an
 * instruction range, so these are three consecutive addresses that nothing
 * executes: signals.c shows a cancellation handler __cp_begin while a
 * cancellation point's call has no result, __cp_end after, and reads back
 * __cp_cancel as the request to abandon the call. */
  .section .rodata.capstone_cancel_points,"a",@progbits
  .global __cp_begin
  .global __cp_end
  .global __cp_cancel
__cp_begin:
  .byte 0
__cp_end:
  .byte 0
__cp_cancel:
  .byte 0
