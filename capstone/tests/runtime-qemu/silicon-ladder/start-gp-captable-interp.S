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

/* ---- capability bounds representability (C-13, glue side) -------------------
   CVA6's compress_bounds encodes a base exactly only when it is a multiple of

       granule(L) = 1 << (max(0, floor(log2 L) - 12) + 3)

   An unaligned base is truncated DOWNWARD, so the capability silently overlaps
   its neighbour -- the same failure that put dom_data.base 96 bytes low and cost
   the C-13 hunt. That was fixed in the monitor's split geometry; the glue's own
   per-global carve had the same hole.

   Lengths below 16384 have a granule of 8 or 16 and are already covered by the
   16-byte carve, so this path is DEAD for every rung built to date and wakes up
   only for a global >= 16 KiB or a cap table of >= 1024 entries. SQLite is the
   first domain to have either (a 256 KiB global, and 1059 entries). */

/* \dst = granule(\len). Valid only for \len >= 16384; callers branch around it
   otherwise. Clobbers \dst and \tmp. */
.macro REPR_GRAN dst, tmp, len
  li    \dst, 3                      /* e+3, e = 0                                */
  srli  \tmp, \len, 13               /* >= 2 here, so the loop runs at least once */
53:
  addi  \dst, \dst, 1
  srli  \tmp, \tmp, 1
  bnez  \tmp, 53b
  li    \tmp, 1
  sll   \dst, \tmp, \dst
.endm

/* Carve \len bytes off the top of sp into \cap, keeping BOTH endpoints granule-
   aligned so the result is exactly representable. \top is the running split
   point and is updated. Clobbers \s0, \s1, \s2 (and \len, which is rounded up --
   the caller's pad-zeroing loop then covers the extra bytes). */
.macro CARVE cap, top, len, s0, s1, s2
  li    \s0, 16384
  bltu  \len, \s0, 52f               /* granule <= 16: 16-byte carve already suffices */
  REPR_GRAN \s0, \s1, \len
  addi  \s1, \s0, -1                 /* granule - 1                                   */
  not   \s2, \s1                     /* ~(granule - 1)                                */
  and   \s0, \top, \s2
  beq   \s0, \top, 51f
  split(\cap, sp, \s0)               /* drop a pad so the TOP is granule-aligned      */
  mv    \top, \s0
51:
  add   \len, \len, \s1
  and   \len, \len, \s2              /* round the length so the BASE is aligned too   */
52:
  sub   \top, \top, \len
  split(\cap, sp, \top)
.endm

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
  ld    t0, 0(a6)                    /* t0 = initfn - entry (link-time relative)      */
  add   t0, a7, t0                   /* -> initfn runtime address                     */
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

.macro BUILD_GP_CAPTABLE_INTERP
#if defined(INTERP_FAKE_COUNT) && INTERP_DIAG_STAGE < 2
  /* Diagnostic build: no descriptor is read, so no blob view is needed -- which also
     lets us drop the early delin(sp). That matters because R-2 is literally "delin in
     domain code wedges the board", and the generated prologue (which WORKS on silicon)
     delins sp only at the END. Nothing here touches sp with cincoffset, so sp can stay
     linear: split and scc do not consume rs1. */
#else
  /* STAGE 2 restores exactly this block and nothing else, so a board run attributes
     the failure to the early delin(sp) + sp-derived view rather than to cap-init. */
  delin(sp)                          /* C-4b: sp arrives LIN; cincoffset would eat it */
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
  CARVE gp, t1, t3, a5, a6, a7       /* gp = [t1,END) table; sp = [base,t1)            */
#if INTERP_SP_LINEAR
  delin(gp)                          /* only needed when sp was still LIN (see C-13)   */
#endif
  cincoffset(s2, gp, x0)             /* s2 = moving table pointer                      */

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
  CARVE t2, t1, t4, a5, a6, a7       /* t2 = storage cap (granule-aligned if large)    */
#if INTERP_SP_LINEAR
  delin(t2)                          /* only needed when sp was still LIN (see C-13)   */
#endif

  /* dst walker; t6 counts bytes still to write into storage. */
  cincoffset(t6, t2, x0)
  bltz  t5, 30f                      /* blob_off < 0 -> pure zero fill                 */

  /* ---- copy `size` bytes from blob+blob_off, 8 at a time, then a byte tail ---- */
  cincoffset(a4, s1, t5)             /* a4 = src = blob + blob_off                     */
  mv    a5, t3                       /* a5 = bytes left to copy                        */
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
  BUILD_GP_CAPTABLE_INTERP
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
  BUILD_GP_CAPTABLE_INTERP
  j test

.text
  .global test
test:
  cincoffsetimm(sp, sp, -96)
  stc(ra, sp, 48)                    /* save the entry return cap for domreturn */
  sd a0, 64(sp)
  stc(a1, sp, 80)                    /* a1 = shared region cap (res) */

  /* Cap-init runs HERE, not at the entry points, because it CALLS: `jalr ra`
     clobbers ra, and ra holds the entry return capability that domreturn needs.
     Running it before `stc(ra, sp, 48)` destroyed that capability and the domain
     faulted at its own exit (cause 24 at the domreturn) with the compute already
     finished and correct -- a failure that looks like a miscompute but is not.
     my_first_domain/start.S orders it the same way, for the same reason. */
  RUN_CAP_INIT

  ldc(a0, sp, 80)                    /* a0 = res */
  ld a1, 64(sp)                      /* a1 = func */
  call domain_main

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

  .align 4
  .section .gct
  .global __capstone_gct_start
  .hidden __capstone_gct_start
__capstone_gct_start:
