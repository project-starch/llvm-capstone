/* Domain entry glue for pure-capability domains that make HOSTCALLS.
 *
 * This is my_first_domain/start.S plus exactly one capability: a RESUMABLE
 * yield. The shared start.S is left byte-identical on purpose -- every existing
 * domain and every silicon ladder rung depends on it, and a change there would
 * put a working measurement at risk to buy nothing.
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
 * The existing HostCall v0 probes get resumption a different way: they run an
 * S-mode payload inside a nested domain (create_dom(sbi.dom, x.smode)) and
 * yield with an SBI DOM_RETURN ecall. That path is not pure-capability, which
 * is why it cannot carry musl.
 *
 * NO SCALAR STATUS IS RETURNED. The shared glue zeroes every register before
 * returning (capability-leak hygiene), so a domain's `call_dom` result is
 * always 0 and SQLite already carries its real status in the shared region.
 * The yield keeps that property: the host distinguishes "pending hostcall"
 * from "finished" by reading metadata->phase, not from a return value.
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

  /* PUBLISH dom_data, so the libc can put its heap there instead of in .bss.
   *
   * sp still holds exactly what cscratch gave us: the capability for the
   * domain's second region. Both entry paths reach here with it -- _start sets
   * the cursor to the end first, __test_reentry takes it straight from cscratch
   * -- and the cursor does not matter, because the reader asks for base and end.
   *
   * Same three-instruction idiom as __capstone_dom_ret below, deliberately: that
   * one is proven on every entry of every domain that uses this glue, and the
   * gp-relative address computation it relies on is not something to re-derive
   * from scratch for a second symbol.
   *
   * The slot is a WEAK definition in libc-ext/malloc.c, so a domain that links
   * this glue without that libc still assembles and links. */
  .Lpcrel_domdata_entry:
  auipc t0, %pcrel_hi(__capstone_dom_data)
  addi t0, t0, %pcrel_lo(.Lpcrel_domdata_entry)
  cincoffset t0, gp, t0
  stc(sp, t0, 0)

  cincoffsetimm(sp, sp, -96)
  stc(ra, sp, 48)
  sd a0, 64(sp)
  stc(a1, sp, 80)

  /* THREAD POINTER. musl reaches errno as `tp - 0x124`
     (__errno_location: `mv a0, tp; addi a0, a0, -0x124`), so an unset tp makes
     the first failing syscall fault in __syscall_ret instead of returning -1.
     Measured that exact way: helper_cscincoffset on an untagged base, with the
     real error hidden behind it.
     A domain has no threads, so tp does not need a real TLS image -- it needs to
     be a capability with enough room below it for errno_val. The block is
     1024 bytes and tp points at its middle, which covers the 0x124 reach with
     margin. gp covers the whole image, so deriving from gp is in bounds. */
  .Lpcrel_tls:
  auipc t0, %pcrel_hi(__capstone_tls)
  addi t0, t0, %pcrel_lo(.Lpcrel_tls)
  addi t0, t0, 512
  cincoffset tp, gp, t0

  /* Stash the sealed-return capability where the yield can find it.
     `ra` holds it on EVERY entry, which is why the shared glue can do
     `stc(ra, sp, 48)` here and still work across SQLite's three entries. The
     yield needs it from arbitrary depth inside libc, so it also goes into a
     global slot. */
  .Lpcrel_domret_entry:
  auipc t0, %pcrel_hi(__capstone_dom_ret)
  addi t0, t0, %pcrel_lo(.Lpcrel_domret_entry)
  cincoffset t0, gp, t0
  stc(ra, t0, 0)

  /* INITIALISE THE pthread STRUCT tp POINTS AT, before any C runs.
   *
   * tp on its own is only addressable space; musl expects a struct __pthread
   * below it with at least `locale` populated, and reads it as a capability.
   * Zeroed, that read comes back untagged and faults one dereference later --
   * in strerror, which is the first thing in a domain that ever asked.
   *
   * AFTER the domret stash above, not before it: this is a call, so it clobbers
   * ra, and ra is exactly what that stash stores. Putting it earlier assembles,
   * links and boots, and breaks the yield -- the domain would return to whatever
   * this call left in ra. Before the cap-init loop below, so an initializer may
   * already use the libc.
   *
   * Called from the glue rather than from each program because every domain
   * using THIS glue links libc-capstone.a by construction: the file exists to
   * carry musl. Same cjalr encoding as the domain_main call further down.
   *
   * ra is dead from here to domain_main, which reloads what it needs. */
  .Lpcrel_init_tp:
  auipc t0, %pcrel_hi(__capstone_init_tp)
  addi t0, t0, %pcrel_lo(.Lpcrel_init_tp)
  cincoffset t0, gp, t0
  .insn i 0x5b, 0x5, ra, 0(t0)

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

/* void __capstone_yield(void);
 *
 * Hand control back to the host and resume HERE when the host calls again.
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
 * The shared glue already encodes the remedy and it was there to be read: it
 * stashes sp into `cscratch` before returning and `__test_reentry` recovers it
 * with `ccsrrw(sp, cscratch, x0)`. cscratch is the designated carrier for a
 * capability across a domain boundary. So: sp travels in cscratch, and
 * everything else travels on the stack that sp then finds again.
 *
 * Callee-saved registers are saved even though `gp` suggests some registers do
 * survive. Which ones is unverified, one wrong guess costs a ten-minute boot to
 * discover, and 14 stores are free by comparison.
 *
 * cscratch needs no save/restore: `_start` reads it with `ccsrrw(sp, cscratch,
 * x0)`, which leaves null behind, and the resume path here does the same, so
 * the register is returned to the state the rest of the glue expects.
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
  stc(tp, sp, 224)

  .Lpcrel_domret_load:
  auipc t3, %pcrel_hi(__capstone_dom_ret)
  addi t3, t3, %pcrel_lo(.Lpcrel_domret_load)
  cincoffset t3, gp, t3
  ldc(t1, t3, 0)
  lla t2, .Lyield_resume
  ccsrrw(x0, cscratch, sp)
  domreturn(t1, t2, x0)

.Lyield_resume:
  ccsrrw(sp, cscratch, x0)
  ldc(gp, sp, 16)

  /* `ra` holds the FRESH sealed-return capability the resuming CALL left; the
     next yield needs it. Stash before restoring the C return address. */
  .Lpcrel_domret_store:
  auipc t3, %pcrel_hi(__capstone_dom_ret)
  addi t3, t3, %pcrel_lo(.Lpcrel_domret_store)
  cincoffset t3, gp, t3
  stc(ra, t3, 0)

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

/* One capability-sized slot. Not in .capstone_cap_init: it is written at entry
   from a register, never materialised from the static image. */
  .section .data
  .align 4
  .global __capstone_dom_ret
__capstone_dom_ret:
  .zero 16

/* Backing store for the thread pointer. Not a TLS image: just addressable,
   writable space so musl's errno has somewhere to live in a domain that has no
   threads. */
  .align 4
  .global __capstone_tls
__capstone_tls:
  .zero 1024

/* THIS GLUE REQUIRES TWO SYMBOLS IT DOES NOT DEFINE:
 *
 *   __capstone_dom_data   a 16-byte slot the entry stores dom_data into
 *   __capstone_init_tp    called once, to populate the struct tp points at
 *
 * libc-capstone.a provides both. A domain that links NO libc -- yield-probe
 * links start-musl.o and its own domain and no archive at all -- links
 * runtime/no-libc-stubs.c instead, and gets a link error naming these two if it
 * forgets.
 *
 * A WEAK DEFINITION HERE WOULD BE WORSE THAN THE LINK ERROR, and was tried on
 * 2026-08-20. It links everywhere, and it is silently wrong: a weak definition
 * SATISFIES the reference, so the linker never extracts tls.o from the archive,
 * and the real __capstone_init_tp is dropped. Measured, not reasoned --
 * `llvm-nm` on the mruby domain showed `W __capstone_init_tp` with size 0, the
 * stub. __capstone_dom_data survived only because malloc.o is pulled anyway.
 * The domain then boots, runs, and faults in strerror much later.
 *
 * A loud link error naming both symbols is the honest failure here. */
