/* What every driver in this port needs to be a domain, and nothing else.
 *
 * Three drivers here enter a Capstone domain: the unprotected replay, the
 * Sublet replay, and the level below's own test. They differ in what they
 * measure and in nothing about being a domain, so this file is the part they
 * share: the entry trampoline, the way back to the monitor when something
 * faults, the scratch allocator the identity tables come out of, and the one
 * helper that prints a table row.
 *
 * It was three copies of the same eighty lines before, because each driver was
 * started from the one before it. The copies had already drifted.
 *
 * WHAT THE INCLUDER MUST DO
 *
 *   define PG_DOM_FAIL_MARKER before including, if it wants its own; the
 *     default is the replay's
 *   define pg_domain_entry(unsigned *res, unsigned func), which the
 *     trampoline jumps to
 *   set scratch_next and scratch_end before calling REPLAY_ALLOC
 *
 * WHAT IT GETS
 *
 *   give_up(code)         returns the core to the monitor, does not return
 *   die(msg), die_at(...) the same, with a message in the payload first
 *   scratch_alloc(n)      zeroed memory from whatever region it was pointed at
 *   row(name, was, got)   one line of a markdown table
 *   the region-size macros, and the hostcall struct the metadata region holds
 *
 * The symbol names are deliberately the same in all three images. They are
 * separate binaries, so nothing collides, and a single name means the
 * trampoline's assembly needs no preprocessor games to find its frame.
 */

/* ---- what the domain runtime and the freestanding half provide ----------- */
#define CAPSTONE_DPI_REGION_SHARE 1U

struct pg_hostcall_v0 {
    unsigned long long phase, opcode, offset, length;
    long long result, error;
};

void pg_domain_payload(char *base, unsigned long *length, unsigned long capacity);
void pg_domain_text(const char *s);
void pg_domain_uint(unsigned long v);

/* The region sizes the host was told to make. Both halves must agree, so they
   are build parameters and the host publishes what it used. */
#ifndef PG_REPLAY_PAYLOAD_SIZE
#define PG_REPLAY_PAYLOAD_SIZE 65536UL
#endif
#ifndef PG_REPLAY_ARENA_SIZE
#define PG_REPLAY_ARENA_SIZE (32UL * 1024UL * 1024UL)
#endif
#ifndef PG_REPLAY_TRACE_SIZE
#define PG_REPLAY_TRACE_SIZE (64UL * 1024UL * 1024UL)
#endif
#ifndef PG_DOM_FAIL_MARKER
#define PG_DOM_FAIL_MARKER "__CAPSTONE_PG_REPLAY_FAILED__\n"
#endif

static volatile struct pg_hostcall_v0 *meta;
static volatile char *payload;
static unsigned shares;

/* ---- the driver's own memory, not the manager's --------------------------
 * The identity tables come out of here. An identity table for a million
 * objects is twenty megabytes of capabilities, which is why it cannot be in
 * the image. Where the region comes from is the includer's business: the
 * unprotected replay carves it off the front of the arena, and the Sublet
 * replay is given one of its own, because its arena must stay linear and a
 * table walked with ordinary pointer arithmetic cannot live in a linear
 * region.
 */
static char *scratch_next, *scratch_end;

static void *
scratch_alloc(size_t n)
{
    n = (n + 15UL) & ~(size_t) 15UL;
    if (!scratch_next || scratch_next + n > scratch_end)
        return NULL;
    char *p = scratch_next;

    scratch_next += n;
    for (size_t i = 0; i < n; i++)
        p[i] = 0;
    return p;
}

/* ---- the way back to the monitor ----------------------------------------
 * A fault must return the core rather than park the domain: a domain that
 * spins is a host that never reads the payload, and a run that says nothing
 * is worse than one that says what went wrong. start.S's frame is recorded on
 * entry by the trampoline below and give_up restores it, so the host regains
 * the core with the message already in the payload.
 *
 * give_up does not return, which is also what lets a caller use it inside an
 * expression: the replay loop asks for a context by name, and a name it was
 * never given has no value to carry on with.
 */
unsigned char pg_domain_exit_frame[32] __attribute__((aligned(16), used));
static unsigned *domain_result;

static void
fail(const char *what)
{
    pg_domain_text("pg-domain: ");
    pg_domain_text(what);
    pg_domain_text("\n");
}

__attribute__((noreturn)) static void
give_up(unsigned code)
{
    pg_domain_text(PG_DOM_FAIL_MARKER);
    if (domain_result)
        *domain_result = code;
    __asm__ volatile(
        "1: auipc t0, %%pcrel_hi(pg_domain_exit_frame)\n"
        "  addi t0, t0, %%pcrel_lo(1b)\n"
        "  .insn r 0x5b, 0x1, 0xc, t0, gp, t0\n"
        "  .insn i 0x5b, 0x3, sp, 0(t0)\n"   /* ldc sp, 0(t0) */
        "  .insn i 0x5b, 0x3, ra, 16(t0)\n"  /* ldc ra, 16(t0) */
        "  ret\n" ::: "memory");
    for (;;)
        ;
}

__attribute__((noreturn)) static void
die(const char *what)
{
    fail(what);
    give_up(0xBADCAFE0u);
}

__attribute__((noreturn)) static void
die_at(unsigned long i, const char *what, unsigned long id)
{
    fail(what);
    pg_domain_text("  at record ");
    pg_domain_uint(i);
    pg_domain_text(", id ");
    pg_domain_uint(id);
    pg_domain_text("\n");
    give_up(0xBADCAFE1u);
}

/* ---- the entry ----------------------------------------------------------
 * The frame is recorded here, before anything else runs, so that a fault in
 * the first instruction of the body can still get back.
 */
void pg_domain_entry(unsigned *res, unsigned func);

__asm__(
    "  .text\n"
    "  .globl domain_main\n"
    "domain_main:\n"
    "1: auipc t0, %pcrel_hi(pg_domain_exit_frame)\n"
    "  addi t0, t0, %pcrel_lo(1b)\n"
    "  .insn r 0x5b, 0x1, 0xc, t0, gp, t0\n" /* cincoffset t0, gp, t0 */
    "  .insn s 0x5b, 0x4, sp, 0(t0)\n"       /* stc sp, 0(t0) */
    "  .insn s 0x5b, 0x4, ra, 16(t0)\n"      /* stc ra, 16(t0) */
    "  j pg_domain_entry\n");

static void
row(const char *name, unsigned long was, unsigned long got)
{
    pg_domain_text("| ");
    pg_domain_text(name);
    pg_domain_text(" | ");
    pg_domain_uint(was);
    pg_domain_text(" | ");
    pg_domain_uint(got);
    pg_domain_text(" |\n");
}

#define REPLAY_DIE(msg) die(msg)
#define REPLAY_DIE_AT(i, msg, id) die_at((i), (msg), (id))
#define REPLAY_ALLOC(n) scratch_alloc(n)
