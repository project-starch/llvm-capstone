/* Generic gp-captable entry glue, DESCRIPTOR-DRIVEN (silicon ladder).
 *
 * Drop-in replacement for start-gp-captable-generic.S + the per-app
 * gp_captable_build.inc that gen-gp-captable-glue.py emits. Same entry/reentry
 * contract, same domain_main call, same domreturn unwind. The difference is how
 * the cap table is built:
 *
 *   generic: a GENERATED, unrolled prologue -- one li/sd pair per 8 bytes of every
 *            initialized global, plus a split/delin/stc per global, emitted twice
 *            (entry and reentry). O(total global bytes) of .text.
 *   interp:  this file -- a FIXED loop over `.capstone_gp_initdesc`, which the
 *            compiler emits and the monitor copies into the front of dom_data.
 *            O(1) .text regardless of global count or size.
 *
 * WHY IT MATTERS. The generated form has hard ceilings that all bite at SQLite
 * scale: 2040 B per initialized global (12-bit store offset), 127 globals total
 * (12-bit cap-table immediates), no `.L`-private symbol may use the bulk-copy path
 * (the generator must `lla` it from a separate TU), and ~25 bytes of .text per byte
 * of initialized data. SQLite has ~1,095 globals, 910 of them private. Reading the
 * offsets from a descriptor instead of baking them into generated code removes all
 * four at once, because the COMPILER can name a private symbol and emit
 * `sym - __gpfree_globals_base` as an ordinary link-time difference.
 *
 * WHY A RUNTIME DESCRIPTOR READ IS LEGAL HERE. The long-standing rule was "silicon
 * forbids a runtime descriptor read" -- true while the only copy lived in the code
 * image, which the domain holds via an EXECUTE-authority capability (the 2026-07-22
 * root cause: a code cap cannot load data on captype-fixed CVA6). It no longer
 * applies: the monitor now copies the initialized-globals bytes into dom_data, which
 * the domain holds with DATA authority. This reads the descriptor from THERE, never
 * from the image.
 *
 * DESCRIPTOR LAYOUT (see CapstoneAsmPrinter::emitGpCaptableInitDesc). The linker
 * script places it FIRST in the globals region, so its blob offset is exactly 0 and
 * this code needs no address arithmetic to find it:
 *   0:  u64 built_flag     (0 in the image; reserved for the reentry fast path)
 *   8:  u64 count
 *   16: u64 gp_slot[2]     (16 B, reserved for parking gp on reentry)
 *   32: per global, 24 B:  u64 size ; u64 align ; i64 blob_off
 *                          blob_off == -1 means zero-initialized (.bss).
 *
 * REGISTERS. s1 = blob (data cap over dom_data, cursor at base), s2 = moving
 * cap-table pointer, s3 = loop index, s4 = count, t1 = carve address (integer),
 * t2 = storage cap, t0/t3/t4/t5/t6 = scratch. s-registers are dead after this runs
 * (domain_main is an ordinary C function that saves its own), and gp persists by
 * ABI. `sp` is touched ONLY by split/scc -- never cincoffset -- because
 * `cincoffset rd, rs1, rs2` with rd != rs1 CONSUMES a linear rs1 (issue C-4b);
 * the delin below makes that safe regardless.
 */
/* Diagnostic staging for C-13 (see ISSUES). 1 = minimal glue, proven PASSING on
   silicon; 2 = minimal + the early delin(sp) and sp-derived blob view. */
#ifndef INTERP_DIAG_STAGE
#define INTERP_DIAG_STAGE 1
#endif

/* C-13. Is `sp` still LINEAR after the entry block above?
   The RTL's DELIN accepts CAP_TYPE_LINEAR ONLY and raises UNEXPECTED_CAP_TYPE
   otherwise -- it is NOT idempotent, unlike our patched QEMU helper_csdelin, which
   returns early for an already-NONLIN cap and hid this for the entire bring-up.
   SPLIT preserves cap_type (modify_cap_start/_end/_revnode all copy it through), so
   once sp is delin'd at entry, EVERY capability split from it is already NONLIN and
   each later delin is both redundant and fatal on silicon.
   So: delin exactly once. When the entry delin runs, all the later ones are gated off. */
#if defined(INTERP_FAKE_COUNT) && INTERP_DIAG_STAGE < 2
#define INTERP_SP_LINEAR 1         /* entry delin skipped -> sp and its splits stay LIN */
#else
#define INTERP_SP_LINEAR 0         /* entry delin ran -> everything derived is NONLIN   */
#endif

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

/* Build the per-global capability table from .capstone_gp_initdesc.
   Precondition: sp = dom_data, cursor at its base. Clobbers t0-t6, s1-s4.
   Postcondition: gp = cap table, sp = remaining stack with cursor at its top. */

/* Run capability-global initializers (.capstone_cap_init) -- S1.5.
 *
 * A capability tag cannot live in the static image, so a global initialized to a
 * POINTER (a function-pointer method table, an array of `const char *`, `&other[N]`)
 * loads untagged. The compiler synthesizes one initializer function per module and
 * records it in .capstone_cap_init as a PC-relative `initfn - entry` diff; running
 * them is what turns those words into real capabilities.
 *
 * The generated silicon glue never did this -- no ladder rung has such a global, so
 * nothing caught it. SQLite has 54 of them (VFS method tables, sqlite3StdType,
 * sqlite3aLTb/aEQb/aGTb, ...), so it is a hard prerequisite, not an optimisation.
 *
 * TWO DIFFERENCES from my_first_domain/start.S, both forced by this ABI:
 *  - It reads the table from the BLOB, not the image. Under gp-captable the image is
 *    reachable only through an EXECUTE-authority capability, and the 2026-07-22 root
 *    cause established that a code cap cannot load data on captype-fixed CVA6. The
 *    monitor's copy of the globals region is in dom_data, which we hold as DATA.
 *    The table's blob offset is the link-time constant
 *    (__capstone_cap_init_start - __gpfree_globals_base); both `lla`s are PC-relative,
 *    so their difference is base-independent.
 *  - It calls with a plain `jalr`, not a capability call. Under the gp-free ABI calls
 *    are ordinary jumps within PCC; `gp` here is bounded to the cap table and must not
 *    be used to derive a code capability.
 *
 * Empty table (start == end) is a no-op. Clobbers t0-t6, a4-a7; preserves gp, sp,
 * a0, a1 (initializers are ordinary C and save what they use).
 */
.macro RUN_CAP_INIT
  /* Skipped on a reentry for the same reason the table build is: re-running the
     initializers would reset every pointer-valued global to its image value,
     silently discarding whatever the previous entry stored there. */
#if defined(INTERP_FAKE_COUNT) && INTERP_DIAG_STAGE == 5
  /* STAGE 5: the same value stage 4 obtained with `lla`, obtained instead as a BAKED
     IMMEDIATE. If stage 4 fails and stage 5 passes, the fault is specifically auipc in
     domain code, and baking link-time constants is a proven workaround -- which is the
     whole fix direction for the copy path and cap-init. Same instruction count either
     way, so this is not a timing artefact. */
  li    t3, 0x11000
  j     79f
#endif
#if defined(INTERP_FAKE_COUNT) && INTERP_DIAG_STAGE == 4
  /* STAGE 4: one `lla` and nothing else. beebs_prime's cap-init table is EMPTY, so in
     stage 3 the only instructions that actually executed were two llas and a bgeu --
     the jalr never ran. If stage 4 fails, the culprit is `lla` (auipc) in domain glue,
     not the indirect call. That would also explain R-9: the large-RO copy path emits
     `lla <sym>` while the zero-init/unrolled paths emit none. */
  lla   t3, __capstone_cap_init_start
  j     79f
#endif
/* Stages 1, 2 and 6 all skip cap-init. `< 3` alone was a BUG: stage 6 is meant to be
   stage 2 plus ONE load, but `6 < 3` is false, so it silently un-skipped this whole
   block and re-introduced two `lla`/`auipc` pairs and a store into the blob -- exactly
   the confound behind the retracted lla/auipc root cause (1d83cd9260af, retracted by
   d579630011ee). Any stage that is "stage 2 plus something" must skip cap-init too.
   Verify by diffing disassembly after changing this; do not trust the #if. */
/* INTERP_SKIP_CAPINIT is a REAL-BUILD knob (independent of INTERP_FAKE_COUNT), the
   companion to INTERP_SKIP_GPPARK. Cap-init is the OTHER thing the real path does that
   board-verified stage 6 does not. For a rung whose .capstone_cap_init table is empty
   (every ladder rung), skipping it is semantically a no-op, so a pass/fail difference
   isolates the two `lla`/`auipc` pairs and the bgeu -- and nothing else. */
#if defined(INTERP_FAKE_COUNT) && (INTERP_DIAG_STAGE < 3 || INTERP_DIAG_STAGE >= 6)
  j     79f                          /* diagnostic: skip cap-init entirely             */
#endif
#if !defined(INTERP_FAKE_COUNT) && defined(INTERP_SKIP_CAPINIT)
  j     79f                          /* real build, cap-init skipped for bisection     */
#endif
  beqz  s5, 79f                      /* fast path: table already built, globals live   */
  lla   t3, __capstone_cap_init_start
  lla   t4, __capstone_cap_init_end
  bgeu  t3, t4, 78f                  /* empty table -> still record that we built      */
  lla   t5, __gpfree_globals_base
  sub   t6, t3, t5                   /* table's offset into the blob                  */
  /* The initializers are ordinary C, so they save/restore callee-saved registers
     with sd/ld -- 8-byte integer accesses that DROP A CAPABILITY TAG. Anything
     capability-valued must therefore be parked with stc/ldc across the call, not
     left in a register. Keeping the blob in s1 across the call is what faulted on
     the first attempt: `Cap mem access OOB ... rs1 = x9`, s1 having come back
     untagged. sp and gp are safe because the ABI keeps them intact. */
  cincoffsetimm(sp, sp, -32)
  stc(s1, sp, 0)                     /* park the blob cap across the calls            */
  cincoffset(a6, s1, t6)
  stc(a6, sp, 16)                    /* park the entry pointer too                    */
  mv    a7, t3                       /* a7 = runtime IMAGE address of the entry       */
70:
  ldc(a6, sp, 16)
#ifdef INTERP_CAPINIT_IMAGE_TABLE
  /* DIAGNOSTIC: read the entry's delta from the IMAGE (a7) rather than from the monitor's
     copy of it in dom_data (a6). Both should hold the same bytes -- the table lives in the
     globals region, so it is copied -- and a difference means the blob copy or the offset
     into it is wrong, which would send this `jalr` to an arbitrary address. Observed:
     execution reaching sqlite3_aggregate_context on what should be a capability-stash
     entry, i.e. control transferred somewhere it was never meant to go.

     RESULT (2026-07-31): this DOES NOT WORK and the knob is kept only to record why.
     `lla` yields a plain INTEGER address under the gp-free ABI, and a bare `ld` through an
     integer base is rejected:
         Cap mem access requires capability: pc = ..., rs1 = x17, imm = 0
         domain halted by capability fault: cause = 24
     Reading the table through the blob capability (a6) is therefore not an accident of the
     implementation, it is the only legal way to reach those bytes from the glue. Do not
     "simplify" it back to an image-relative load. */
  ld    t0, 0(a7)
#else
  ld    t0, 0(a6)                    /* t0 = initfn - entry (link-time relative)      */
#endif
  add   t0, a7, t0                   /* -> initfn runtime address                     */
#ifdef INTERP_PEEK_CAPINIT_TARGET
  /* DIAGNOSTIC: publish the computed call target instead of calling it. The target is
     (image address of the table entry) + (delta read from the monitor's COPY of that entry
     in dom_data). Everything about that chain checks out statically -- the ELF entry
     resolves to __capstone_cap_init, and the table sits at blob offset 0x12700 well inside
     the 78768-byte copied region -- so the remaining question is what the glue actually
     reads at run time. The host prints this as `obs=`. Compare against
     pcc_base + (0x141b70 - 0x10000); anything else means the delta or the blob base is
     wrong, and the `jalr` is transferring control to an arbitrary function. */
  /* 112, not 80: test: parks res at 80(sp), then RUN_CAP_INIT drops sp by 32 before this
     loop, so the slot is 32 higher from in here. Reading 80(sp) lands on the parked RETURN
     capability and faults. */
  /* csdebugprint is a QEMU-ONLY debug op (funct7 0x43 on opcode 0x5b); the FPGA decoder
     has nothing there. It prints the named register straight to the emulator console,
     which beats routing a value through the shared region -- the host only reports `obs`
     on its own failure paths and would never surface it. QEMU-only, so this knob must
     never be set for a board build. */
  .insn r 0x5b, 0x1, 0x43, x0, t0, x0
  ldc(a2, sp, 112)                   /* a2 = res capability parked by test:            */
  sw    t0, 0(a2)
  /* Unwind exactly as the normal path does before leaving. Branching straight to 78f skips
     BOTH the blob-cap restore and the `sp += 32`, leaving sp 32 bytes low for the rest of
     the domain -- which shows up as an unrelated fault in whatever runs next (strlen, in
     the first attempt). A diagnostic that changes the frame is measuring itself. */
  ldc(s1, sp, 0)
  cincoffsetimm(sp, sp, 32)
  j     78f
#endif
  jalr  ra, 0(t0)                    /* plain call: gp-free ABI, within PCC           */
  ldc(a6, sp, 16)                    /* reload: the callee may have clobbered it      */
  cincoffsetimm(a6, a6, 8)
  stc(a6, sp, 16)
  addi  a7, a7, 8
  lla   t4, __capstone_cap_init_end
  bltu  a7, t4, 70b
  ldc(s1, sp, 0)                     /* restore the blob cap                          */
  cincoffsetimm(sp, sp, 32)
78:
  /* Persist "built" ONLY here, once the table AND the initializers are done. Setting
     it inside BUILD would make the very first entry skip its own cap-init, since
     RUN_CAP_INIT runs later (in `test:`, after the return capability is parked). The
     in-register s5 carries "did build" across that gap; the in-memory flag is what a
     LATER entry sees. */
  li    t3, 1
  sd    t3, 0(s1)
79:
.endm

.macro BUILD_GP_CAPTABLE_INTERP first_entry=1
#if defined(INTERP_FAKE_COUNT) && INTERP_DIAG_STAGE < 2
  /* Diagnostic build: no descriptor is read, so no blob view is needed -- which also
     lets us drop the early delin(sp). That matters because R-2 is literally "delin in
     domain code wedges the board", and the generated prologue (which WORKS on silicon)
     delins sp only at the END. Nothing here touches sp with cincoffset, so sp can stay
     linear: split and scc do not consume rs1. */
#else
  /* STAGE 2 restores exactly this block and nothing else, so a board run attributes
     the failure to the early delin(sp) + sp-derived view rather than to cap-init. */
  /* FIRST ENTRY ONLY. sp arrives LINEAR from the monitor and must be delin'd once, or
     the cincoffset below consumes it (C-4b). But on a REENTRY sp comes back from cscratch
     ALREADY NONLIN -- the exit path stashes the delin'd sp -- and DELIN is LINEAR-ONLY on
     silicon: capstone_dyn_unit.anvil:447-448 raises UNEXPECTED_CAP_TYPE for any cap_type
     that is not CAP_TYPE_LINEAR. Our QEMU returns early for NONLIN and treats delin as
     idempotent (op_helper.c:905-910), so the second delin is invisible under emulation and
     fatal on hardware -- QEMU-green / FPGA-dead by construction.
     This is R-2's shape ("delin in domain code wedges the board"). C-13 removed the
     TRAILING delin but left this one, safe only under the assumption stated at :520 --
     "the reentry fast path, which the ladder rungs never take". SQLite is the first domain
     to take it: it dies on its SECOND entry, before the built_flag is even read.
     NOTE: QEMU CANNOT validate this fix -- delin is idempotent there, so both versions
     pass. Only the board can. */
  .if \first_entry
  delin(sp)                          /* C-4b: sp arrives LIN; cincoffset would eat it */
  .endif
  cincoffset(s1, sp, x0)
  lcc(t3, sp, 3)                     /* t3 = sp.base                                   */
  scc(s1, s1, t3)                    /* s1 = blob view, cursor forced to the base      */
#endif

  /* S2 -- IDEMPOTENT ENTRY. On the FIRST entry sp is the whole dom_data with its
     cursor at the base; on REENTRY it is whatever the exit path stashed in cscratch,
     i.e. the already-carved stack region -- which still starts at the same base, so
     the blob is reachable either way once the cursor is forced there.

     Without this, both entry points rebuild the table and re-run every initializer,
     so globals DO NOT SURVIVE a domreturn. That is a correctness gate rather than a
     saving: sqlite_capstone_domain.c handles two CAPSTONE_DPI_REGION_SHARE entries
     that stash the host capabilities in globals before the run entry, so a rebuilding
     reentry would reset shared_region_count to 0, drop both capabilities, and run
     SQLite with no host channel.

     The flag lives at blob[0] and is 0 in the image, so a freshly created domain
     always builds. It is safe to rely on that zero specifically because the blob is a
     deterministic COPY of image bytes -- dom_data itself is __get_free_pages memory
     and is not zeroed. gp is parked at blob[16] with stc, so it keeps its tag. */
#ifdef INTERP_FAKE_COUNT
#if INTERP_DIAG_STAGE >= 7
  /* STAGE 7+: USE the real built_flag (do not force the build path). Stage 6 proved a
     load from 0(s1) succeeds but DISCARDED the value; stage 7 is the first stage whose
     control flow depends on it. If 7 fails where 6 passed, the blob's built_flag is not
     the 0 the image holds -- i.e. the monitor copy did not land where the glue looks. */
  ld    t4, 0(s1)
#else
#if INTERP_DIAG_STAGE >= 6
  /* STAGE 6 -- CAN THE DOMAIN READ THE MONITOR-COPIED BLOB AT ALL?
     Stage 2 (entry delin present, delin fix applied) plus ONE load through the
     sp-derived blob view -- verified by DISASSEMBLY DIFF, not by reading the #if.
     (The first version of this claim was false: the cap-init skip at the top of
     RUN_CAP_INIT was gated `< 3`, so stage 6 also un-skipped two lla/auipc pairs and
     a store. Fixed; re-diff after any change here.) The value is discarded and we
     always build, so nothing downstream depends on what comes back -- this isolates the
     READ itself and nothing else.
     Why this is the right next test: the monitor's WRITE into dom_data is proven, but the
     domain reading it back is the one assumption in this design never checked on silicon,
     and it is the only thing the REAL interp path does that stage 2 does not. Real interp
     still failed WITH the delin fix, so the remaining fault is either here or downstream
     of it.
     stage 6 FAILS -> the domain cannot read the blob; the descriptor-driven design needs
                      the descriptor delivered another way (e.g. baked immediates).
     stage 6 PASSES -> the read works and the remaining fault is in the descriptor's
                      CONTENTS or in the record loop that consumes them. */
  ld    t4, 0(s1)                    /* the load under test                            */
#endif
  li    t4, 0                        /* diagnostic: always build, ignore what was read */
#endif
#else
  ld    t4, 0(s1)                    /* built_flag                                     */
#endif
  beqz  t4, 5f
  li    s5, 0                        /* s5 = 0: took the fast path, did NOT build      */
  ldc(gp, s1, 16)                    /* already built: reload gp and skip everything   */
  j     99f
5:
  li    s5, 1                        /* s5 = 1: building now                           */
  /* INTERP_FAKE_COUNT=N replaces the runtime descriptor READ with an immediate, to
     test on silicon whether the domain can read the monitor-copied blob at all. The
     monitor's WRITE into dom_data is proven; the domain's READ back out is the one
     assumption in this design never checked on hardware, and for a rung with a single
     .bss global it is the ONLY thing interp does that the generated prologue does not.
     Diagnostic only -- a real build must read the descriptor. */
/* STAGE 10 splits stage 8's failure in two. Board-measured: stage 7 (which LOADS and
   USES blob offset 0) PASSES, stage 8 (which adds a load of offset 8) FAILS. Two very
   different causes are consistent with that, and they need different fixes:
     (i)  the LOAD at +8 itself faults, or
     (ii) the load succeeds but the VALUE is garbage, so `count` is huge and the
          subsequent `slli`/`sub`/`split` carves at a wild bound and faults.
   Stage 10 performs exactly stage 8's load and then THROWS THE VALUE AWAY, using the
   immediate count instead. So it isolates the access from the value:
     stage 10 FAILS  -> the load at +8 faults (a hardware/access-path problem)
     stage 10 PASSES -> the load is fine and the CONTENT at +8 is wrong, i.e. the
                        monitor's 16-byte capability-granular copy does not lay plain
                        scalars down contiguously (the descriptor's own image bytes are
                        verified correct: built_flag=0, count=1).
   t3 is dead here -- its last use was `scc(s1, s1, t3)` in the entry block. */
#if defined(INTERP_FAKE_COUNT) && INTERP_DIAG_STAGE == 11
  /* STAGE 11 -- OBSERVE the blob instead of inferring it. Reads one word, keeps it in s6,
     and builds with a fake count so the domain still runs; the zero-fill path below
     publishes s6 as the global's value, which the harness prints as retval.
     s6 is untouched in between (the builder uses t0-t6, s1-s4).

     INTERP_PEEK_OFF selects WHICH word, so one boot can map the whole descriptor header
     instead of arguing about one ambiguous zero. Expected values for a correct copy,
     from the image (verified statically):
        +0  built_flag   0     <- ZERO, so indistinguishable from an empty page: useless alone
        +8  count        1
        +32 rec0.size    8     <- nonzero, unambiguous
        +40 rec0.align   8     <- nonzero, unambiguous
        +48 rec0.blob_off -1   <- 0xFFFFFFFF as a 32-bit retval: unmistakable
     A correct copy makes +32/+40/+48 clearly nonzero. All-zero across every offset means
     the blob is simply not there, and the fix belongs at the copy's DESTINATION.

     INTERP_PEEK_MODE=1 instead publishes sp.BASE -- the address the glue believes the
     blob starts at. Non-comparable to the monitor's own value directly, but a wildly
     implausible base (0, or outside the domain's allocation) would be decisive. */
#ifndef INTERP_PEEK_OFF
#define INTERP_PEEK_OFF 8
#endif
#if defined(INTERP_PEEK_MODE) && INTERP_PEEK_MODE == 1
  lcc(s6, sp, 3)                     /* publish sp.BASE (low 32 bits reach retval)      */
#elif defined(INTERP_PEEK_MODE) && INTERP_PEEK_MODE == 2
  lcc(s6, sp, 4)                     /* publish sp.END                                  */
#elif defined(INTERP_PEEK_MODE) && INTERP_PEEK_MODE == 3
  /* PUBLISH THE REGION SIZE -- the single most diagnostic number available to the
     domain, and immune to not knowing the physical base. If sp really is dom_data then
     end-base must be exactly tot_size - code_size - DOMAIN_DATA_SIZE
     = 131072 - 4192 - 1536 = 125344 for these probe domains.
       125344    -> sp IS dom_data; the glue looks in the right place, so the monitor
                    wrote the blob somewhere else
       something else -> sp is NOT the region the monitor copied into, and the
                    destination mismatch is on the glue/handoff side */
  lcc(s6, sp, 4)
  lcc(t3, sp, 3)
  sub   s6, s6, t3
#else
  ld    s6, INTERP_PEEK_OFF(s1)
#endif
  li    s4, INTERP_FAKE_COUNT
#elif defined(INTERP_FAKE_COUNT) && INTERP_DIAG_STAGE == 10
  ld    t3, 8(s1)                    /* STAGE 10: the load under test, value DISCARDED */
  li    s4, INTERP_FAKE_COUNT
#elif defined(INTERP_FAKE_COUNT) && INTERP_DIAG_STAGE < 8
  li    s4, INTERP_FAKE_COUNT
#else
  ld    s4, 8(s1)                    /* count                                          */
#endif
  beqz  s4, 99f                      /* no globals: nothing to build                   */

  lcc(t1, sp, 4)                     /* t1 = sp.END                                    */
  slli  t3, s4, 4                    /* count * 16                                     */
  sub   t1, t1, t3
  split(gp, sp, t1)                  /* gp = [t1,END) table; sp = [base,t1)            */
#if INTERP_SP_LINEAR
  delin(gp)                          /* only needed when sp was still LIN (see C-13)   */
#endif
  cincoffset(s2, gp, x0)             /* s2 = moving table pointer                      */

/* INTERP_BUILD_LIMIT -- build only the first N entries while the TABLE stays full size.
   Clamped here, AFTER the table split, precisely so the cap-table geometry is identical
   to an unclamped build and the only variable is how many carve iterations run.

   This is the bisect for the two-global failure. A domain that reads ONLY slot 0
   (gpn2use0) still has a 2-entry table and a 2-record descriptor; with the limit at 1
   the glue performs one split and one stc instead of two. So:
     limit=1 PASSES, unclamped FAILS -> the SECOND split/store is what breaks it
     both FAIL                       -> the damage is already done by the table split
                                        or by something outside the loop entirely
   Diagnostic only; never set in a shipping build. */
#ifdef INTERP_BUILD_LIMIT
  li    t3, INTERP_BUILD_LIMIT
  bge   t3, s4, 96f
  mv    s4, t3
96:
#endif

/* STAGE 9+: derive the record pointer and read records for real. */
#if !defined(INTERP_FAKE_COUNT) || INTERP_DIAG_STAGE == 9
  cincoffsetimm(t0, s1, 32)          /* t0 = &record[0]                                */
#endif
  li    s3, 0
10:
#if defined(INTERP_FAKE_COUNT) && INTERP_DIAG_STAGE != 9
  li    t3, 16                       /* pretend: 16 B, zero-init -- reads no descriptor */
  li    t5, -1
#else
  ld    t3, 0(t0)                    /* size                                           */
  ld    t5, 16(t0)                   /* blob_off (-1 => zero-init)                     */
#endif

  /* stor = align_up(size,16), minimum 16 so a zero-size global still gets a
     representable, non-degenerate storage capability rather than an empty split. */
  addi  t4, t3, 15
  andi  t4, t4, -16
  bnez  t4, 11f
  li    t4, 16
11:
  /* BOUNDS REPRESENTABILITY. A capability's bounds are encoded with a granule of
       1 << (max(0, floor(log2 len) - 12) + 3)
     so 16-byte alignment is exact only while that granule is <= 16, i.e. len < 8192.
     Above it the hardware CANNOT represent a 16-aligned base/length and must round the
     encoded bounds outward -- which silently moves the object the cap-table entry points
     at. This is the same rule that caps merged string containers at 4096 bytes.

     Measured 2026-07-31 on the SQLite domain: of 179 globals exactly ONE trips it --
     index 170, sqlite_heap, 262144 bytes, granule 512 -- and it is precisely the object
     whose contents strlen reads. QEMU tolerated it; on silicon strlen ran off the end of
     the heap and scanned 31,342,951 bytes.

     So align BOTH ends of every carve to that global's own granule: round the running top
     DOWN and the storage size UP. Costs at most granule-1 bytes of padding per global, and
     only for globals >= 8192 bytes, of which this domain has one. */
/* INTERP_NO_GRANULE_ALIGN=1 disables this, as a CONTROL. The granule alignment below was
   added 2026-07-31 and touches a4/a5/a6 and t1/t4 in the middle of the carve loop; when
   string CONTENTS later turned out to be wrong on silicon (strlen of a 21-char literal
   returning 1), the first thing to rule out was whether this edit caused it. Keep the knob
   so that control can be re-run cheaply rather than reasoned about. */
/* CORRECTION 2026-07-31 (later the same day): the revert note below is WRONG about WHICH
   end is misaligned, and the block does in fact fix every carve. Replaying this exact loop
   over the real descriptor of a built probe domain (same granule shift loop, same
   stor = max(16, align_up(size,16)), top descending) gives:

       granule-align OFF -> 1 carve not representable:
           idx 170  size=262144  stor=262144  granule=512  base%g=64  len%g=0
       granule-align ON  -> 0 carves not representable

   identical for every plausible region top tried, so it does not depend on where dom_data
   lands. len%g is ZERO: the LENGTH was always fine and the BASE is what is misaligned.
   align_up(262144,512) == 262144, so this block's stor adjustment is a no-op for that
   object and its `top = align_down(top, granule)` is precisely the missing piece.

   Still off by default until confirmed on silicon, because the last decision here was taken
   from an unverified claim and reversing it on another one would repeat the mistake. */
/* DISABLED BY DEFAULT 2026-07-31, same day it was added. Enable with INTERP_GRANULE_ALIGN=1
   only to re-measure it.
   It was added to make carve bounds exactly representable, and it does NOT achieve that: it
   masks `t1` BEFORE `sub t1,t1,t4`, but the carve's upper end is the PREVIOUS iteration's
   `t1`, so the BASE ends up granule-aligned and the LENGTH does not. Audited on the real
   descriptor: idx 170 (sqlite_heap, 262,144 B, granule 512) gets a carve length of 262,384,
   and 262,384 % 512 = 240 -- still unrepresentable, so the top must round outward.
   Worse, it introduced a NEW defect: the zero-fill writes `t4` = 262,144 bytes into a
   262,384-byte carve, so the top 240 bytes of the memsys5 arena capability are now whatever
   the page allocator left there. They were zeroed before this block existed.
   And it fixed nothing measurable: 14 domains carrying this exact code pass SHA5->SHA6
   twice, and simulation over the real descriptors shows the residual misalignment is LARGER
   in a domain that passes (304 B) than in the one that wedges (240 B), i.e. it is
   non-discriminating.
   A correct version must align the LENGTH (mask after the subtraction, or align every carve
   to one global maximum granule), not the base. Until then, off. */
#if defined(INTERP_GRANULE_ALIGN)
  li    a4, 8                        /* a4 = granule                                    */
  mv    a5, t4
12:
  li    a6, 8192
  blt   a5, a6, 13f
  slli  a4, a4, 1
  srli  a5, a5, 1
  j     12b
13:
  addi  a5, a4, -1                   /* a5 = granule - 1                                */
  not   a6, a5                       /* a6 = ~(granule - 1)                             */
  add   t4, t4, a5
  and   t4, t4, a6                   /* stor = align_up(stor, granule)                  */
  and   t1, t1, a6                   /* top  = align_down(top, granule)                 */
#endif
  sub   t1, t1, t4
  split(t2, sp, t1)                  /* t2 = storage cap                               */
#if INTERP_SP_LINEAR
  delin(t2)                          /* only needed when sp was still LIN (see C-13)   */
#endif

  /* dst walker; t6 counts bytes still to write into storage. */
  cincoffset(t6, t2, x0)
  bltz  t5, 30f                      /* blob_off < 0 -> pure zero fill                 */

  /* ---- copy `size` bytes from blob+blob_off, 8 at a time, then a byte tail ---- */
  cincoffset(a4, s1, t5)             /* a4 = src = blob + blob_off                     */
  mv    a5, t3                       /* a5 = bytes left to copy                        */
  /* UNALIGNED SOURCE -> BYTE COPY. The fast path below issues `ld` (8 bytes) from
     blob+blob_off, and blob_off is NOT 8-byte aligned for most globals: measured on the
     SQLite domain, 67 of 176 records have blob_off%8 != 0 (e.g. idx=1 blob_off=4375,
     idx=4 blob_off=4385). The destination is always fine -- carves are >=16-aligned -- so
     only the source matters, and blob itself is aligned, so blob_off&7 IS the misalignment.

     QEMU services an unaligned `ld` transparently, which is why this never showed under
     emulation. CVA6 does not, and the result is not a clean trap: on the board a 21-char
     literal read back as byte 0 correct, bytes 1-4 ZERO, bytes 5-8 correct (strlen
     returned 1 instead of 21). That corrupts every merged string container, which is why
     sqlite3RegisterBuiltinFunctions -- the first code to hash function NAMES -- was the
     bisected wedge point.

     The byte loop at 21 already copies a5 bytes correctly; it was only ever used as the
     size%8 tail. Sending the whole object there when the source is unaligned costs a few
     ms once per entry and is unconditionally correct. */
  andi  a6, t5, 7
  bnez  a6, 21f
20:
  li    a6, 8
  blt   a5, a6, 21f
  ld    a7, 0(a4)
  sd    a7, 0(t6)
  cincoffsetimm(a4, a4, 8)
  cincoffsetimm(t6, t6, 8)
  addi  a5, a5, -8
  j     20b
21:
  beqz  a5, 22f                      /* byte tail: the size%8 != 0 case the unrolled   */
  lb    a7, 0(a4)                    /* path could not express at all                  */
  sb    a7, 0(t6)
  cincoffsetimm(a4, a4, 1)
  cincoffsetimm(t6, t6, 1)
  addi  a5, a5, -1
  j     21b
22:
  sub   t4, t4, t3                   /* remaining pad = stor - size                    */

  /* ---- zero the tail (pad after a copy, or the whole thing for .bss) ---- */
30:
  beqz  t4, 40f
  /* R-16 FIX (2026-08-04, board-measured): always pad BYTE-WISE.
     The 8-byte pad store below starts at t6 = t2 + size. For any global whose size is not a
     multiple of 8 that address is MISALIGNED, and this core mishandles misaligned 8-byte
     accesses (see the note at the copy loop) -- it writes the aligned window containing t2 and
     destroys the bytes just copied. That is why every `char[2]` global read back 0 while
     `char[8]` globals and zero-init .bss globals were fine.
     Measured: with this jump, three char[2] globals return 321 (all correct) where the
     unmodified glue returns 0. The cost is a byte loop over at most stor-size (< 16) bytes per
     global, once per domain entry.
     A faster fix exists -- zero the whole carve first from t6 = t2 with t4 = stor, both
     16-aligned, then copy over it -- but it restructures this block, so it wants review rather
     than an end-of-session edit.

     COST NOTE (2026-08-04). For the COPY case the byte loop runs at most stor-size < 16 bytes,
     which is what was measured. For the .bss case it does NOT: `bltz t5, 30f` arrives here with
     t4 = stor, the WHOLE object, so every zero-initialized global is filled one byte at a time.
     SQLite's sqlite_heap alone is 262,144 bytes of .bss. Correct, but it makes domain entry
     cost proportional to total .bss in single-byte stores -- which also pollutes any
     entry/boundary-cost measurement taken through this glue.

     INTERP_PAD_BYTEWISE=0 restores the pre-fix path as a CONTROL. It reintroduces the
     misaligned-store corruption for globals whose size % 8 != 0, so it is for A-B testing
     only, never for a correctness run. */
#ifndef INTERP_PAD_BYTEWISE
#define INTERP_PAD_BYTEWISE 1
#endif
#if INTERP_PAD_BYTEWISE
  j     32f
#endif
31:
  li    a6, 8
  blt   t4, a6, 32f
#if defined(INTERP_FAKE_COUNT) && INTERP_DIAG_STAGE == 11
  sd    s6, 0(t6)                    /* STAGE 11: publish the peeked blob word          */
#else
  sd    x0, 0(t6)
#endif
  cincoffsetimm(t6, t6, 8)
  addi  t4, t4, -8
  j     31b
32:
  beqz  t4, 40f
  sb    x0, 0(t6)
  cincoffsetimm(t6, t6, 1)
  addi  t4, t4, -1
  j     32b

40:
  stc(t2, s2, 0)                     /* cap-table[i] = storage                         */
  cincoffsetimm(s2, s2, 16)
#if !defined(INTERP_FAKE_COUNT) || INTERP_DIAG_STAGE == 9
  cincoffsetimm(t0, t0, 24)          /* next record                                    */
#endif
  addi  s3, s3, 1
  blt   s3, s4, 10b

  scc(sp, sp, t1)                    /* sp.cursor = top of the remaining stack         */

  /* C-13 ROOT CAUSE: delin sp EXACTLY ONCE.
     The RTL's DELIN (capstone-ariane core/anvil_build/capstone_dyn_unit.anvil) raises
     UNEXPECTED_CAP_TYPE unless the operand is CAP_TYPE_LINEAR -- it is NOT idempotent.
     Our QEMU helper_csdelin (op_helper.c:900) was patched to return early when the cap
     is already NONLIN, so a double delin is silent under emulation and fatal on silicon.
     Every real (non-diagnostic) build takes the #else branch above and delins sp at
     ENTRY to derive the blob view, which leaves sp NONLIN here -- so this trailing delin
     was a second delin on a non-linear capability and wedged the board. It is also
     redundant: its only purpose is to make the stack non-linear for the app, which the
     entry delin already did.
     Measured: stage 1 (no entry delin) 4/4 PASS, stage 2 (entry delin) 3/3 FAIL.
     SPLIT and SCC both accept NONLIN, so DELIN is the only instruction that cared. */
#if INTERP_SP_LINEAR
  delin(sp)                          /* entry delin was skipped: sp is still LIN here  */
#endif
/* INTERP_SKIP_GPPARK is a REAL-BUILD diagnostic knob (independent of
   INTERP_FAKE_COUNT). This `stc` is a CAPABILITY store into the monitor-copied blob --
   the only capability-width write the glue makes into dom_data, and one of just two
   things the real path does that the board-verified stage 6 does not. Skipping it
   costs only the reentry fast path, which the ladder rungs never take (one entry). */
#if !defined(INTERP_FAKE_COUNT) && !defined(INTERP_SKIP_GPPARK)
  stc(gp, s1, 16)                    /* park gp for the reentry fast path              */
#endif
99:
.endm

__test_reentry:
  ccsrrw(sp, cscratch, x0)
#ifdef INTERP_PEEK_SP
  /* Print sp at BOTH entry points. The re-carve-on-reentry fix depends on this being the
     SAME region each time; if the first entry's carve loop consumed sp by repeated `split`
     and cscratch holds the post-carve remainder, the second carve starts smaller, every
     global moves, and the domain reads storage that holds nothing. QEMU-only debug op. */
  .insn r 0x5b, 0x1, 0x43, x0, sp, x0
#endif
  BUILD_GP_CAPTABLE_INTERP 0
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
  ccsrrw(sp, cscratch, x0)           /* sp = dom_data (cscratch) DATA region */
#ifdef INTERP_PEEK_SP
  .insn r 0x5b, 0x1, 0x43, x0, sp, x0
#endif
  BUILD_GP_CAPTABLE_INTERP
#ifdef INTERP_PEEK_GP
  /* Print gp itself (the cap-table base) so a faulting address can be tested for being
     INSIDE THE TABLE. A 16-byte capability is exactly one table entry, so a pointer bounded
     to 16 bytes is the signature of code that used the ADDRESS OF A SLOT where it should
     have `ldc`-ed the slot to obtain the object's capability. QEMU-only debug op. */
  .insn r 0x5b, 0x1, 0x43, x0, gp, x0
#endif
#ifdef INTERP_PEEK_SLOT
  /* Print the capability the builder actually installed in gp[INTERP_PEEK_SLOT], right
     after the table is built and before anything uses it. The DESCRIPTOR for a slot can be
     verified statically (and is: slot 903 records size 464, matching the symbol), but
     nothing until now has checked what the glue BUILT from it. A holder whose runtime
     bounds are wrong would let a cap-init store land outside its global -- which is the
     only remaining way a store can corrupt the frame and send a later `ret` into an
     unrelated function. QEMU-only (csdebugprint); never set for a board build. */
  li    t5, (INTERP_PEEK_SLOT) * 16
  cincoffset(t6, gp, t5)
  ldc(t6, t6, 0)
  .insn r 0x5b, 0x1, 0x43, x0, t6, x0
#endif
  j test

.text
  .global test
test:
  cincoffsetimm(sp, sp, -96)
  stc(ra, sp, 48)                    /* save the entry return cap for domreturn */
  sd a0, 64(sp)
  stc(a1, sp, 80)                    /* a1 = shared region cap (res) */

#if defined(INTERP_DOMAIN_MTVEC)
  /* INSTALL THE TRAP VECTOR **BEFORE** CAP-INIT, not just before domain_main.
     Measured 2026-08-05: with the install sited immediately before `call domain_main`, a
     SQLite domain at stage 11 wedged with the trap latch reading 0x98 -- seen=1, mcause=24,
     a capability fault -- and the handler-ON and handler-OFF images wedged IDENTICALLY, same
     latch, same debug mux. The handler cannot have run. The install was simply too late:
     RUN_CAP_INIT is straight-line capability stores over every global (179 carves for
     SQLite), it runs BEFORE this point, and with mtvec still 0 a fault there vectors to pc=0,
     lands outside PCC and re-faults forever -- silently, which is the whole failure mode this
     gate exists to break.

     Siting it here costs nothing and strictly dominates: mtvec is a CSR, so RUN_CAP_INIT
     clobbering t0 afterwards is irrelevant, and cscratch is provably free across cap-init
     (written only at the entry points and read again at the exit unwind), so the frame-cap
     stash survives it. */
  lla   t0, .Ldomain_trap
  csrw  mtvec, t0
  ccsrrw(x0, cscratch, sp)
#endif

  /* Cap-init runs HERE, not at the entry points, because it CALLS: `jalr ra`
     clobbers ra, and ra holds the entry return capability that domreturn needs.
     Running it before `stc(ra, sp, 48)` destroyed that capability and the domain
     faulted at its own exit (cause 24 at the domreturn) with the compute already
     finished and correct -- a failure that looks like a miscompute but is not.
     my_first_domain/start.S orders it the same way, for the same reason. */
  RUN_CAP_INIT

  ldc(a0, sp, 80)                    /* a0 = res */
  ld a1, 64(sp)                      /* a1 = func */

#if defined(INTERP_RETURN_PRECALL)
  /* BISECT THE ENTRY GLUE WITHOUT TOUCHING THE CAP-TABLE BUILD.
     Returns a sentinel from HERE -- after the carve loop and after RUN_CAP_INIT, immediately
     before domain_main -- so the question "did the glue complete?" gets a yes/no answer:
        returns 0x9E11        -> carve loop AND cap-init both completed; the fault is inside
                                 domain_main (or in reaching it)
        wedges                -> the fault is in the glue itself
     That is the one split the evidence cannot currently make: SQLite wedges at stage 0, which
     runs no SQLite work, so the fault is somewhere in carve / cap-init / domain_main entry.

     WHY IT IS SITED HERE AND GATED ON func. The glue runs THREE times per domain -- once per
     region share and once for call_domain (sbi_capstone.c: the share path re-enters via
     __domcallsaves). A probe that returns unconditionally would break share #1, and a probe
     that alters BUILD_GP_CAPTABLE_INTERP kills the domain there too: that is exactly how
     INTERP_BUILD_LIMIT and INTERP_FAKE_COUNT/INTERP_DIAG_STAGE were shown to destroy the thing
     they measure (SHA6 absent, dead before the code under test ran). So:
       * nothing above this point changes -- the entry path and the table build are byte-identical
       * share entries (func == CAPSTONE_DPI_REGION_SHARE == 1) fall through untouched, so both
         shares still complete and SHA6 still appears twice
       * only the REAL call is diverted
     INTERP_SKIP_CAPINIT is the companion knob and the only other one that leaves the build
     alone; the pair brackets cap-init. */
  li    t0, 1                        /* CAPSTONE_DPI_REGION_SHARE */
  beq   a1, t0, 98f                  /* a share entry: behave exactly as before */
  li    t0, 0x9E11                   /* sentinel: "glue reached precall" */
  sw    t0, 0(a0)                    /* res is a plain word the host prints as obs= */
  j     .Ldomain_returned
98:
#endif

#ifdef INTERP_PEEK_PRECALL
  /* Print the frame state immediately before entering domain_main. With string merging on,
     the domain dies at its first entry with cause = 1, pc = 0 -- an instruction fetch at
     null with no preceding memory fault -- and cap-init's own body has been cleared (call
     target exact, ra never written). That leaves the glue's post-cap-init unwind: if the
     blob-cap restore or the `sp += 32` left sp or the parked return capability wrong, the
     next control transfer is what fails. Printing sp, the parked return cap and the two
     arguments says which, instead of inferring it. QEMU-only debug op (funct7 0x43). */
  .insn r 0x5b, 0x1, 0x43, x0, sp, x0
  ldc(t0, sp, 48)
  .insn r 0x5b, 0x1, 0x43, x0, t0, x0
  .insn r 0x5b, 0x1, 0x43, x0, a0, x0
  .insn r 0x5b, 0x1, 0x43, x0, a1, x0
#endif
#if defined(INTERP_FENCE_BEFORE_MAIN)
  /* MEMORY ORDERING BETWEEN THE GLUE'S WRITES AND THE DOMAIN'S READS.
     The carve loop and __capstone_cap_init between them STORE every global in the domain --
     blob copies, zero fills, and one capability store per cap-init leaf -- and domain_main
     is the first code to LOAD them. There is no fence anywhere in this file, so nothing
     orders those stores against those loads.

     Evidence this matters, measured on silicon 2026-08-01:
       * stage 66 walks the SAME element TWICE through the SAME pointer and is
         DETERMINISTIC at rc=2 (0b10) across three runs in one boot: the FIRST walk overruns
         and the SECOND terminates correctly. Straight-line code, identical loops, same
         memory -- so the bytes genuinely read back differently the first time.
       * stage 63 (four identical arrays) is NON-DETERMINISTIC across runs in a single boot:
         0x0E then 0x0F. A deterministic logic bug cannot vary run to run.
     First-read-stale plus run-to-run variation is a coherency signature, not a logic bug.

     `fence rw,rw` is the minimal correct barrier here. Gated rather than unconditional
     because the last two changes to this file were made on unverified reasoning and one had
     to be reverted; this gets measured on the deterministic probe (stage 66: 2 -> 3 would
     confirm) before it becomes the default. */
  fence rw, rw
#endif
  /* ROUTE A rationale (the install itself now happens BEFORE cap-init, above):
     the monitor never writes dom_seal[1] (sbi_capstone.c:760,782-784) and slot 1 IS
     {ctvec,mtvec} (csr_regfile.sv:399), so a domain runs with mtvec=0: any fault vectors to
     pc=0, lands outside PCC and re-faults forever, silently. That is why a fault and a hung
     core are indistinguishable, and why every wedge so far has been unrepeatable (a wedge
     ends the board session).
     Board-confirmed prerequisite: a domain CAN write mtvec -- stage 75 wrote 0x40 and read
     0x40 back (2026-08-01). If it could not, the csrw would raise ILLEGAL INSTRUCTION, which
     is cause 2 and is excluded from the trap latch, i.e. it would look exactly like a hang.
     Converting a wedge into a RETURN is the whole point: it makes wedging probes repeatable
     within one boot.
     Gated: OFF unless built with INTERP_DOMAIN_MTVEC=1, so ordinary builds are untouched. */
  call domain_main

#if defined(INTERP_RETURN_POSTCALL)
  /* SECOND BISECTION POINT, same technique as INTERP_RETURN_PRECALL and the natural next split.
     The precall probe proved the glue completes (obs 0x9E11 in 4 s, control green), so what
     remains is: the call transfer into domain_main, its stage-0 body, and its return. This
     writes its sentinel AFTER the call returns:
        returns 0x9E22  -> domain_main RETURNED; the fault is in the glue's post-call unwind
        wedges          -> domain_main itself never came back
     Overwriting whatever domain_main wrote is fine: this build is a diagnostic, never a
     measurement. a0/a1 are caller-saved and the call has clobbered them, so both are reloaded
     from the frame rather than assumed. Gated on func so region shares stay untouched, and
     sited below the entry path so the table build stays byte-identical to the baseline. */
  ld    t1, 64(sp)                   /* func, reloaded: a1 is clobbered by the call */
  li    t0, 1                        /* CAPSTONE_DPI_REGION_SHARE */
  beq   t1, t0, 97f
  ldc(a0, sp, 80)                    /* res, reloaded for the same reason */
  li    t0, 0x9E22
  sw    t0, 0(a0)
97:
#endif
#if defined(INTERP_DOMAIN_MTVEC)
  /* Normal return: put cscratch back to zero so .Ldomain_returned sees exactly the state it
     saw before this gate existed. The trap path below arrives at the same state by swapping
     the stashed cap OUT, so both paths converge with identical register and CSR state. */
  ccsrrw(x0, cscratch, x0)
#endif

.Ldomain_returned:
  lla t2, __test_reentry
  ldc(t1, sp, 48)                    /* t1 = entry return cap */
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

#if defined(INTERP_DOMAIN_MTVEC)
  /* Placed AFTER the infinite loop above, so it is never reached by fall-through -- it is
     only ever entered by the trap vector. mtvec requires a 4-byte-aligned target and mode 0
     (direct), which `lla` of an aligned label gives. */
  .balign 4
.Ldomain_trap:
  /* FAULT-PROOF BY CONSTRUCTION: two instructions, neither of which touches memory.
     `ccsrrw` swaps the stashed glue frame capability back into sp and leaves cscratch zero
     -- byte-identical CSR/register state to the normal-return path above -- so the shared
     unwind at .Ldomain_returned is reached with a valid sp and its `ldc(t1, sp, 48)` resolves.

     Deliberately NOT recovered here: mepc/mcause/mtval are left in their CSRs and captured by
     .Ldomain_returned's existing `csrrw t0, mcause, x0` / mtval sequence, so this gate adds no
     new store that could itself fault. What it buys is the discriminator that has been missing:
     if a build with this handler RETURNS, the domain faulted; if it still hangs, no trap was
     ever taken and the core is genuinely wedged. Those two have been indistinguishable until
     now, and every "it does not fault" claim made without this has had to be retracted.

     There is no `mret` here on purpose. mret would need mstatus.mpp = M, which only a real
     trap sets; the monitor's dom_seal[3] has MPP=0, so a synthetic mret would drop the domain
     to U-mode. domreturn is the exit this glue already uses and it needs no mode change.

     IT NOW ALSO SAYS WHY, because "returns" alone is not a result. Measured 2026-08-12: with
     this handler installed, a SQLite domain RETURNED in 6 s having written neither its result
     word nor a single byte of the payload region -- so it faulted somewhere, and nothing on
     the board could say where. The handler had converted every fault into an indistinguishable
     silent return, which is a strictly worse instrument than the wedge it replaced: a wedge at
     least pinned the core for a debugger. mcause and mepc were sitting in CSRs and simply were
     not reported.

     `res` is only 32 bits wide -- the monitor passes `&res` of a local `unsigned`
     (sbi_capstone.c call_domain) -- so the two values are packed into one word:

         bits 31..28  0xF          marker: this run TRAPPED, the value is not a result
         bits 27..22  mcause & 0x3F
         bits 21..0   (mepc - _start) >> 2, the faulting instruction as an image offset

     A word that does not begin with 0xF is an ordinary result and this path never ran. The
     offset is relative to `_start` rather than absolute because the loader places the image at
     a base that moves with the initramfs, and an absolute mepc has twice been mis-mapped by
     hand; `_start` is in the image, so the subtraction happens where the base is known.

     THIS TRADES SOME FAULT-PROOFING FOR INFORMATION, deliberately. The two-instruction version
     touched no memory and so could not fault again; this one reads the frame and writes through
     the region capability. If the fault was itself caused by a bad sp or a bad region cap, the
     handler re-faults and the core wedges -- which is exactly what happened before this gate
     existed, so the downside is bounded by the status quo. Everything up to the store is
     register-only, so a nested fault can only come from the final two instructions. */
  ccsrrw(sp, cscratch, x0)
  csrr  t0, mcause
  csrr  t1, mepc
  lla   t2, _start
  sub   t1, t1, t2
  srli  t1, t1, 2
  li    t3, 0x3FFFFF
  and   t1, t1, t3
  andi  t0, t0, 0x3F
  slli  t0, t0, 22
  or    t0, t0, t1
  li    t3, 0xF0000000
  or    t0, t0, t3
  ldc(a0, sp, 80)                    /* the shared region cap saved at entry */
  sw    t0, 0(a0)
  j .Ldomain_returned
#endif

  .align 4
  .section .gct
  .global __capstone_gct_start
  .hidden __capstone_gct_start
__capstone_gct_start:
