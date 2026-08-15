/*
 * Compact, self-contained, freestanding string/memory routines shared by the
 * Capstone PureCap libc-frontier BEEBS benchmarks (fasta, and later dtoa/trio).
 *
 * This is the "pure computation" slice of libc — memcpy/memmove/memset/strlen/
 * strcmp/strcpy touch no OS and make no syscalls, so they are implemented
 * locally (the design-sanctioned approach for fine-grained helpers; HostCall is
 * reserved for true OS/IO boundary crossings, and there is no hosted libc on the
 * bare-metal domain path).  It is the string counterpart to the shared
 * double-precision libm in beebs_softfloat_libm.c.
 *
 * Compiled and linked like the libm object; -ffunction-sections/--gc-sections in
 * the per-benchmark build drops whatever a given benchmark does not reference, so
 * pulling in this whole file costs nothing in the final domain image.
 *
 * Build the native self-test with:
 *   cc -DBEEBS_FREESTANDING_STRING_TEST -O2 beebs_freestanding_string.c \
 *      -o /tmp/str_test && /tmp/str_test
 */

typedef __SIZE_TYPE__ bsize_t;

/*
 * BEEBS_MEMCPY_OPTNONE -- compile memcpy, and only memcpy, at -O0 while the rest of this
 * file keeps the optimisation level the build gives it.
 *
 * WHY THIS EXISTS. The two string primitives SQLite depends on are broken at OPPOSITE
 * optimisation levels on silicon, and until now they shared one file-wide `-O` flag, so
 * every SQLite build ran with one defect or the other (issues S-04 and the `-O0` strlen
 * defect):
 *
 *   -O0 memcpy  works                 -O0 strlen  WRONG on silicon: it re-loads the string
 *   -O1 memcpy  WRONG on silicon      capability with `ldc` from a stack slot every
 *               (S-04, below)         iteration and sporadically returns 1 (stage 13
 *                                     returned 15, then 26, then hung, vs 36 on QEMU)
 *
 * `SQLITE_SUPPORT_OPT_LEVEL=-O0` cured S-04 by reintroducing the strlen defect. Scoping the
 * level to the one function that needs it is what lets a build have NEITHER.
 *
 * This is a WORKAROUND for a silicon defect, not a fix, and it is deliberately not a code
 * change: the -O1 code is CORRECT. Verified by disassembling the linked domain
 * (`memcpy` at 0x14ca1c) -- for the failing case (n=7, dst and src both 16-byte aligned)
 * the -O1 form branches over the head loop (`beqz a5`), does not enter the capability loop
 * (`bgeu a2, a4` with a4=16 > n=7), and issues seven `sb` stores from the tail loop. The
 * stores are ISSUED and do not stick. So the earlier theory that "-O1 skips the byte tail
 * loop" is REFUTED; what differs between the working and failing forms is only which
 * capability register holds the destination base (-O1 uses the incoming argument a0
 * directly, -O0 round-trips it through a stack slot).
 *
 * CONFIRMED ON SILICON 2026-08-10 by a matched pair in one boot, control green (k800 = 4):
 * two images differing in memcpy and NOTHING else (compared per symbol, raw encodings) gave
 * stage 164 = 0x74 with -O1 memcpy and 0x70 with this attribute -- one bit apart, and it is
 * exactly the "memcpy does not stick" bit. The failing arm doubles as the positive control,
 * so the clean result is a real negative rather than a dead test.
 *
 * Applies to memcpy ONLY. The other writers have their own knob
 * (BEEBS_STRING_WRITERS_OPTNONE, below) so the two stay independently measurable.
 *
 * Default OFF. Enabled only by build-sqlite-silicon.sh, exactly like
 * BEEBS_STRING_LINEAR_SAFE, so the silicon-ladder rungs keep the geometry their published
 * numbers were taken with. Remove when the silicon defect behind S-04 is fixed.
 */
#if defined(BEEBS_MEMCPY_OPTNONE) && BEEBS_MEMCPY_OPTNONE
#define BEEBS_MEMCPY_ATTR __attribute__((optnone, noinline))
#else
#define BEEBS_MEMCPY_ATTR
#endif

/*
 * BEEBS_STRING_WRITERS_OPTNONE -- the same treatment for the OTHER primitives that WRITE
 * memory: memmove, memset and strcpy. Separate knob from BEEBS_MEMCPY_OPTNONE on purpose, so
 * the two can be measured independently rather than moving four functions at once.
 *
 * The rationale is a rule, not a list: S-04 was a STORE that did not commit, in a primitive
 * compiled at -O1 whose destination capability arrived as an argument and was used straight
 * out of a0. memmove, memset and strcpy have exactly that shape. strlen and strcmp do NOT --
 * they only read, and they must stay at -O1 because the -O0 form of strlen is itself a
 * documented silicon defect. So the split is WRITERS at -O0, READERS at -O1.
 *
 * Whether the other writers are actually affected is measured by stage 167 in
 * sqlite_capstone_domain.c; do not enable this on the theory alone.
 */
#if defined(BEEBS_STRING_WRITERS_OPTNONE) && BEEBS_STRING_WRITERS_OPTNONE
#define BEEBS_WRITER_ATTR __attribute__((optnone, noinline))
#else
#define BEEBS_WRITER_ATTR
#endif

typedef unsigned long long bu64_t;      /* exactly one half of a 128-bit capability word */

#define BEEBS_CHUNK_COPY(D, S) ((void)(*(void **)(D) = *(void *const *)(S)))

/*
 * Capability-preserving memcpy/memmove.
 *
 * A byte loop copies address bits but drops the out-of-band tag of any stored
 * capability, so a copied pointer comes back untagged and the next dereference
 * faults (SQLite gap 3). The fix is to copy the pointer-aligned middle one
 * capability at a time (a `void*` load/store lowers to ldc/stc, which preserve
 * the tag), with a byte head/tail for the unaligned ends. This is only correct
 * once untagged ldc/stc is bit-exact over the full 128-bit word — otherwise the
 * high half of every plain-data chunk is zeroed (gap 4). That QEMU fix landed
 * (capstone-qemu: fix/untagged-ldc-stc-128bit-preservation), so this path is
 * now safe: it preserves tags for capabilities AND the full 16 bytes of plain
 * data. Using sizeof(void*) as the grain keeps the native self-test correct on
 * the host (8-byte pointers) too — on the Capstone target it is 16, i.e. one
 * capability, which is exactly the tag granularity.
 */

#if defined(BEEBS_MEMCPY_TAGCHECK) && BEEBS_MEMCPY_TAGCHECK

/* BEEBS_MEMCPY_TAGCHECK -- is memcpy HANDED an untagged destination, or does it lose the tag
 * mid-loop?
 *
 * Measured on silicon 2026-08-14: mcause 25 UNEXPECTED_OPERAND at memcpy+0x2a8, on
 * `cincoffset a1, a2, a1` where a2 had just been reloaded by `ldc a2, 0x0(a2)` from the
 * destination pointer's stack slot -- untagged. Three ladder rungs refuted every simple
 * explanation (a spilled capability keeps its tag, keeps its bounds, and survives byte stores
 * made through it, 16/16 each on this hardware), so the difference is here, in this function.
 *
 * The FIRST thing to establish is which side of the call boundary it is on, which is what the
 * entry type answers: NOT_CAP (7) on ENTRY means the caller handed over a bad pointer and memcpy
 * is innocent; anything else means the loop lost it, and the offset says where.
 *
 * IT ALSO CONVERTS THE WEDGE INTO A WRONG ANSWER, by stopping the copy and returning. A wedge
 * takes the core and the host never writes the payload out, which is why every observe-only probe
 * on this path reported nothing. An incomplete copy makes SQLite compute a wrong result; that is
 * the intended trade.
 *
 * NO GLOBALS LIVE HERE. This file is one of the "no-globals support objects" the build compiles
 * separately, and the first version of this instrument put six counters in it. That produced a
 * domain that faulted under QEMU on a wild scalar load -- adding globals to an object the
 * gp-captable ABI does not generate a cap table for is not a diagnostic, it is a second bug. The
 * counters live in the domain TU; this side only CALLS across, which is safe.
 *
 * The type query is TOTAL (selector 1), so it can be issued on a NOT_CAP without faulting -- an
 * instrument that raised on exactly the input it exists to describe would report nothing. */
/* `where` says which construct lost it, which is the whole point of adding the chunk loop:
   1 = untagged ALREADY on entry (the caller is the subject, memcpy is innocent)
   2 = the capability-grained CHUNK loop
   3 = the BYTE TAIL loop -- the construct the board fault sits in
   0xAB = the self-test's injected sentinel.
   Without it a hit in the byte loop and a hit in the chunk loop are indistinguishable, and those
   two point at completely different code. */
extern void capstone_mcp_note(unsigned long where, unsigned long entry_ty, unsigned long off,
                              unsigned long n, unsigned long al);

#define BEEBS_MCP_TYPE(out_, cap_) \
  __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x1" : "=r"(out_) : "r"(cap_))
#endif

BEEBS_MEMCPY_ATTR void *memcpy(void *dst, const void *src, bsize_t n) {
  unsigned char *d = (unsigned char *)dst;
  const unsigned char *s = (const unsigned char *)src;
  const bsize_t ps = sizeof(void *);
  bsize_t i = 0;
  bsize_t da = ((bsize_t)d) & (ps - 1u);
  bsize_t sa = ((bsize_t)s) & (ps - 1u);
#if defined(BEEBS_MEMCPY_TAGCHECK) && BEEBS_MEMCPY_TAGCHECK
  unsigned long mcp_ety_;
  BEEBS_MCP_TYPE(mcp_ety_, d);
#if defined(BEEBS_MCP_SELFTEST) && BEEBS_MCP_SELFTEST
  /* POSITIVE CONTROL, end to end: query a value that is NOT a capability and push the result
     through the same note/report path. `hits=0` on a board run is otherwise indistinguishable
     from "the query cannot report 7" or "the reporter was compiled out", both of which have
     happened on this probe already. Requires hits>=1 and ety=7 to appear. */
  { unsigned long mcp_junk_ = 0x5A5Aul, mcp_tj_;
    BEEBS_MCP_TYPE(mcp_tj_, mcp_junk_);
    capstone_mcp_note(0xABul, mcp_tj_, 0xABul, (unsigned long)n, 0xCDul); }
#endif
  if (mcp_ety_ == 7ul) {
    /* Untagged ALREADY -- the caller is the subject, not this loop. Record and bail before the
       first dereference, which would fault and take the report with it. */
    capstone_mcp_note(1ul, mcp_ety_, 0ul, (unsigned long)n,
                      (((unsigned long)da) << 4) | (unsigned long)sa);
    return dst;
  }
#endif
  /* Only when src and dst share alignment can the middle be copied as whole
     capabilities; otherwise fall through to the byte loop for everything. */
#ifndef BEEBS_MEMCPY_BYTES_ONLY
  if (da == sa) {
    bsize_t head = da ? (ps - da) : 0u;
    if (head > n)
      head = n;
#if defined(BEEBS_MEMCPY_TAGCHECK) && BEEBS_MEMCPY_TAGCHECK
  /* Same check, same escape, in memcpy's HEAD loop. Extended here because a fault in a SIBLING primitive is
     indistinguishable from "the check never fired" if only memcpy carries it, and that ambiguity
     cost a board slot. `where` codes 4-7 name which one. */
  {
    unsigned long mcp_t_;
    BEEBS_MCP_TYPE(mcp_t_, d);
    if (mcp_t_ == 7ul) {
      capstone_mcp_note(8ul, mcp_t_, 0ul, (unsigned long)n, 0ul);
      return dst;
    }
  }
#endif
    for (; i < head; i++)
      d[i] = s[i];
    for (; i + ps <= n; i += ps) {
#if defined(BEEBS_MEMCPY_TAGCHECK) && BEEBS_MEMCPY_TAGCHECK
      /* The CHUNK loop, checked separately from the byte tail. The board fault is in the tail,
         but a tail check that never fires while the run still wedges leaves two readings open --
         "the faulting call is the chunk loop" and "the tag dies between the check and the use" --
         and only this arm separates them. */
      {
        unsigned long mcp_ct_;
        BEEBS_MCP_TYPE(mcp_ct_, d + i);
        if (mcp_ct_ == 7ul) {
          capstone_mcp_note(2ul, mcp_ety_, (unsigned long)i, (unsigned long)n,
                            (((unsigned long)da) << 4) | (unsigned long)sa);
          return dst;
        }
      }
#endif
      BEEBS_CHUNK_COPY(d + i, s + i);
    }
  }
#else
  /* BEEBS_MEMCPY_BYTES_ONLY: skip the aligned capability-copy path entirely and use the byte
     loop below for everything. Default OFF; this is a diagnostic/workaround knob, not a fix.
     Reason (issue S-04, measured on silicon 2026-08-10, and STILL PRESENT on caplifive_r20.bit
     so it is NOT R-20): a 7-byte memcpy with dst and src both 16-byte aligned leaves the
     destination completely unchanged -- no bytes copied, and nothing past the range disturbed,
     which is the signature of the copy loop being SKIPPED rather than going wrong. An explicit
     byte loop written at the call site copies the same 7 bytes to the same address correctly.
     The consequence in SQLite is severe and silent: findCollSeqEntry's key copy never lands, so
     the collation hash cannot find its own entry and SQLite reports SQLITE_NOMEM from
     sqlite3_open with nothing having failed to allocate.
     (void)da; (void)sa; keeps the alignment locals used so -Wunused stays quiet. */
  (void)da;
  (void)sa;
#endif
  for (; i < n; i++) {
#if defined(BEEBS_MEMCPY_TAGCHECK) && BEEBS_MEMCPY_TAGCHECK
    /* Checked EVERY iteration, not once: the fault is a reload from the stack slot inside this
       loop, so a single check before the loop would pass and prove nothing about the iteration
       that dies. */
    {
      unsigned long mcp_ty_;
      BEEBS_MCP_TYPE(mcp_ty_, d);
      if (mcp_ty_ == 7ul) {
        capstone_mcp_note(3ul, mcp_ety_, (unsigned long)i, (unsigned long)n,
                          (((unsigned long)da) << 4) | (unsigned long)sa);
        return dst;   /* stop before the faulting use -- a wrong answer beats a wedge */
      }
    }
#endif
    d[i] = s[i];
  }
  return dst;
}

BEEBS_WRITER_ATTR void *memmove(void *dst, const void *src, bsize_t n) {
  unsigned char *d = (unsigned char *)dst;
#if defined(BEEBS_MEMCPY_TAGCHECK) && BEEBS_MEMCPY_TAGCHECK
  /* Same check, same escape, in memmove. Extended here because a fault in a SIBLING primitive is
     indistinguishable from "the check never fired" if only memcpy carries it, and that ambiguity
     cost a board slot. `where` codes 4-7 name which one. */
  {
    unsigned long mcp_t_;
    BEEBS_MCP_TYPE(mcp_t_, dst);
    if (mcp_t_ == 7ul) {
      capstone_mcp_note(4ul, mcp_t_, 0ul, (unsigned long)n, 0ul);
      return dst;
    }
  }
#endif
  const unsigned char *s = (const unsigned char *)src;
  const bsize_t ps = sizeof(void *);
  if (d == s || n == 0)
    return dst;
  if (d < s) {
    /* Forward copy is safe when dst is below src; use the same capability-
       preserving aligned fast path as memcpy. */
    bsize_t i = 0;
    bsize_t da = ((bsize_t)d) & (ps - 1u);
    bsize_t sa = ((bsize_t)s) & (ps - 1u);
    if (da == sa) {
      bsize_t head = da ? (ps - da) : 0u;
      if (head > n)
        head = n;
      for (; i < head; i++)
        d[i] = s[i];
      for (; i + ps <= n; i += ps)
        BEEBS_CHUNK_COPY(d + i, s + i);
    }
    for (; i < n; i++)
      d[i] = s[i];
  } else {
    /* Overlapping copy toward higher addresses: byte loop backward. This drops
       tags for capabilities in an overlapping backward move (rare); such a
       pointer faults loudly on next use rather than corrupting silently. */
    for (bsize_t i = n; i != 0; i--)
      d[i - 1] = s[i - 1];
  }
  return dst;
}

BEEBS_WRITER_ATTR void *memset(void *dst, int c, bsize_t n) {
  unsigned char *d = (unsigned char *)dst;
#if defined(BEEBS_MEMCPY_TAGCHECK) && BEEBS_MEMCPY_TAGCHECK
  /* Same check, same escape, in memset. Extended here because a fault in a SIBLING primitive is
     indistinguishable from "the check never fired" if only memcpy carries it, and that ambiguity
     cost a board slot. `where` codes 4-7 name which one. */
  {
    unsigned long mcp_t_;
    BEEBS_MCP_TYPE(mcp_t_, dst);
    if (mcp_t_ == 7ul) {
      capstone_mcp_note(5ul, mcp_t_, 0ul, (unsigned long)n, 0ul);
      return dst;
    }
  }
#endif
  for (bsize_t i = 0; i < n; i++)
    d[i] = (unsigned char)c;
  return dst;
}

int memcmp(const void *a, const void *b, bsize_t n) {
  const unsigned char *x = (const unsigned char *)a;
  const unsigned char *y = (const unsigned char *)b;
  for (bsize_t i = 0; i < n; i++)
    if (x[i] != y[i])
      return (int)x[i] - (int)y[i];
  return 0;
}

/* BEEBS_STRING_LINEAR_SAFE — index, never walk, when the argument may be a LINEAR
 * capability.
 *
 * The pointer-walking form below is the natural one and is what every BEEBS rung uses,
 * but it is hostile to linear capabilities, and SQLite on silicon is the first thing here
 * ever to call strlen on the board (zero strlen references across all 20 ladder domains).
 * Walking compiles to a loop that COPIES the cursor:
 *
 *     movc          a1, a2        <- keeps the pre-increment pointer for `p - s`
 *     lbu           a3, 0x0(a2)
 *     cincoffsetimm a2, a2, 0x1
 *     bnez          a3, ...
 *
 * `movc` is a MOVE: capstone_flu_unit.anvil:6-27 writes cnull to the SOURCE whenever
 * rd != rs1 and the source is not CAP_TYPE_NONLIN. So on a linear argument the first
 * iteration destroys the very pointer it is walking. The -O0 form avoids the in-loop
 * `movc` but still ends every iteration on `cincoffsetimm` of the live pointer, and both
 * builds freeze at that instruction on hardware.
 *
 * Indexing avoids the whole shape. `s[i]` lowers to CINCOFFSET (reg-reg) into a scratch
 * register plus the load; CINCOFFSET returns rs1 UNCHANGED (capstone_flu_unit.anvil:29-46,
 * `create_result_pack(..., rs1, rd)`), so `s` is never consumed and never incremented.
 * The length is the counter, so there is no `p - s` and hence no trailing `lcc` pair.
 *
 * Opt-in rather than default: the ladder rungs' measured geometry backs a published table
 * and must not change silently. Enabled only by the SQLite silicon build.
 */
/* BEEBS_STRING_DEBUG_BOUNDS — print the ARGUMENT's capability bounds on every strlen call.
 *
 * QEMU-ONLY: csdebugprint is funct7 0x43 on opcode 0x5b and the FPGA decoder has nothing
 * there, so a board build must never set this. Output goes to the emulator console as
 * `Print = Cap(type, perms, cursor, base, end)` (op_helper.c:1439), i.e. it shows the
 * bounds directly rather than making them be inferred.
 *
 * Why it exists: on silicon strlen scanned 31,342,951 characters before the core wedged,
 * against a total domain image of 1.37 MB. Either the string capability really does carry
 * region-sized bounds — in which case this prints them and the bug is found offline — or
 * it does not and the 31 M figure needs a different explanation. Cheap either way, and it
 * needs no board time.
 */
#ifdef BEEBS_STRING_DEBUG_BOUNDS
#define BEEBS_PRINT_CAP(p) __asm__ volatile(".insn r 0x5b, 0x1, 0x43, x0, %0, x0" :: "r"(p))
#else
#define BEEBS_PRINT_CAP(p) ((void)0)
#endif

/* BEEBS_STRLEN_CLAMP=<n> — DIAGNOSTIC. Give up after n bytes instead of scanning forever.
 *
 * Not a fix and never correct: a string longer than the clamp gets the wrong length. It
 * exists to turn one specific silicon failure into information. On the board strlen scanned
 * 31,342,951 bytes -- 120x the widest legitimate string capability in this domain (256 KB,
 * the heap) -- and the core wedged, presumably on reaching unmapped memory. Everything
 * after that point is invisible, so a single bad string hides the entire rest of the run.
 *
 * With the clamp the runaway returns a wrong answer instead of killing the core, and the
 * domain keeps going. That answers the question the wedge cannot: is this ONE bad string
 * with a working SQLite behind it, or the first of many failures? Pick n well above any
 * real SQLite string (64 KiB) so a clamped return is unambiguously the pathological case
 * and never a normal one -- under QEMU it must be unreachable, which the QEMU gate checks.
 */
#ifdef BEEBS_STRING_LINEAR_SAFE
bsize_t strlen(const char *s) {
  bsize_t i = 0;
  BEEBS_PRINT_CAP(s);
#ifdef BEEBS_STRLEN_CLAMP
  while (i < (bsize_t)(BEEBS_STRLEN_CLAMP) && s[i])
    i++;
#else
  while (s[i])
    i++;
#endif
  return i;
}
#else
bsize_t strlen(const char *s) {
  const char *p = s;
  while (*p)
    p++;
  return (bsize_t)(p - s);
}
#endif

/* strcmp/strcpy get the same treatment as strlen under BEEBS_STRING_LINEAR_SAFE, and for
   the same reason -- see the strlen header comment. Walking a pointer compiles to a loop
   that COPIES the cursor (`movc`, which capstone_flu_unit.anvil:6-27 makes destructive for
   a non-NONLIN source) and advances it in place; indexing lowers to CINCOFFSET reg-reg,
   which returns rs1 UNCHANGED (capstone_flu_unit.anvil:29-46).
   These two were missed when strlen was converted, and that omission is exactly why the
   board still wedged in sqlite3RegisterBuiltinFunctions after the string DATA was fixed:
   the patched amalgamation runs `strcmp(zName, "ltrim")` and nine more like it for EVERY
   builtin function, so strcmp is on that path ~10x per entry. memcpy/memmove/memset/memcmp
   need no change -- they already index (`for (; i < n; i++)`). */
#ifdef BEEBS_STRING_LINEAR_SAFE
int strcmp(const char *a, const char *b) {
  bsize_t i = 0;
  while (a[i] && a[i] == b[i])
    i++;
  return (int)(unsigned char)a[i] - (int)(unsigned char)b[i];
}

BEEBS_WRITER_ATTR char *strcpy(char *dst, const char *src) {
  bsize_t i = 0;
  while ((dst[i] = src[i]) != '\0')
    i++;
  return dst;
}
#else
int strcmp(const char *a, const char *b) {
  while (*a && (*a == *b)) {
    a++;
    b++;
  }
  return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

BEEBS_WRITER_ATTR char *strcpy(char *dst, const char *src) {
  char *d = dst;
  while ((*d++ = *src++))
    ;
  return dst;
}
#endif

#ifdef BEEBS_FREESTANDING_STRING_TEST
#include <stdio.h>
/* Our routines use the standard libc names, so we cannot include <string.h>
   here (it would redeclare/clash).  Instead exercise them and sanity-check the
   results directly. */
int main(void) {
  char buf[32], buf2[32];
  int fail = 0;

  for (int i = 0; i < 32; i++)
    buf[i] = (char)0xAA;
  memset(buf, 'x', 10);
  for (int i = 0; i < 10; i++)
    if (buf[i] != 'x')
      fail = 1;
  if ((unsigned char)buf[10] != 0xAA)
    fail = 1;

  const char *msg = "hello world";
  if (strlen(msg) != 11)
    fail = 1;

  memcpy(buf2, msg, 12);
  if (strcmp(buf2, "hello world") != 0)
    fail = 1;

  strcpy(buf, "abc");
  if (strcmp(buf, "abc") != 0 || strcmp(buf, "abd") >= 0)
    fail = 1;

  /* overlapping move: shift "0123456789" right by 2 */
  char ov[16] = "0123456789";
  memmove(ov + 2, ov, 8);
  if (ov[2] != '0' || ov[9] != '7')
    fail = 1;

  printf("freestanding string self-test: %s\n", fail ? "FAIL" : "ok");
  return fail;
}
#endif
