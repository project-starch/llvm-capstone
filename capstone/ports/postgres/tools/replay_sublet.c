/* Replay a recorded PostgreSQL workload against PostgreSQL's memory manager
 * under Sublet, inside a Capstone domain.
 *
 * This is replay_domain.c with the level below changed, and that is the whole
 * of the difference: the loop is replay_core.inc, the same one the host driver
 * and the unprotected domain run, and the manager is the same source with
 * port/aset-sublet.patch applied. Everything the two arms report side by side
 * therefore comes from one loop over one recording.
 *
 * The regions, in the order the host shares them:
 *
 *   0  the hostcall metadata, whose `length` is how much of the payload is written
 *   1  the payload
 *   2  the arena, LINEAR, out of which the level below carves every sub-pool
 *   3  the trace, flattened, header and records
 *   4  the scratch, for the driver's identity tables
 *
 * Two of those need saying. The arena arrives linear, which the host does only
 * when it is told to: a region shared the default way is non-linear, and a
 * handle senior to a non-linear region has nothing to revoke. And it is taken
 * in the share handler rather than in the body, because a linear capability
 * assigned to a C pointer and read back on a later call comes back
 * non-linear. Both cost a run to learn.
 *
 * The scratch is a region of its own for the same reason from the other side.
 * The identity tables are walked with ordinary pointer arithmetic, which is
 * what a linear capability refuses, so they cannot live in the arena.
 *
 * What the measurement is about is what the discipline costs. The counters the
 * level below keeps are printed beside what the backend's manager asked, which
 * the trace itself carries, and beside the primitives the run actually
 * executed, which is what the board's revocation-node pool is spent on.
 */
#include "postgres.h"
#include <stddef.h>
#include <stdint.h>

#include "a11trace.h"

/* ---- what the domain runtime and the freestanding half provide ----------- */
#define CAPSTONE_DPI_REGION_SHARE 1U

struct pg_hostcall_v0 {
    unsigned long long phase, opcode, offset, length;
    long long result, error;
};

void pg_domain_payload(char *base, unsigned long *length, unsigned long capacity);
void pg_domain_text(const char *s);
void pg_domain_uint(unsigned long v);

#include "pg_subpool.h"

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
#ifndef PG_REPLAY_SCRATCH_SIZE
#define PG_REPLAY_SCRATCH_SIZE (32UL * 1024UL * 1024UL)
#endif

static volatile struct pg_hostcall_v0 *meta;
static volatile char *payload;
static char *trace;
static char *scratch;
static unsigned shares;
static unsigned long arena_type = 7;

/* ---- the tables are the driver's memory, not the manager's ---------------
 * They come out of the scratch region, which is not the arena: the arena is
 * linear so that the level below can carve and revoke it, and a table walked
 * with ordinary pointer arithmetic cannot live in a linear region. An identity
 * table for a million objects is twenty megabytes of capabilities, which is
 * why it cannot be in the image either.
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
 * entry and give_up restores it, so the host regains the core with the
 * message already in the payload.
 *
 * give_up does not return, which is also what lets the replay loop use it
 * inside an expression: the loop asks for a context by name, and a name it
 * was never given has no value to carry on with.
 */
unsigned char pg_replay_exit_frame[32] __attribute__((aligned(16), used));

/* What the refusing malloc in port/freestanding/pg_subpool_libc.c calls when a
 * context type the port does not cover asks for memory. It ends the domain at
 * a named place instead of handing out a heap nothing revokes. Defined below,
 * once give_up exists. */
__attribute__((noreturn)) void pg_subpool_refuse(const char *what);
static unsigned *domain_result;

static void
fail(const char *what)
{
    pg_domain_text("pg-replay: ");
    pg_domain_text(what);
    pg_domain_text("\n");
}

__attribute__((noreturn)) static void
give_up(unsigned code)
{
    pg_domain_text("__CAPSTONE_PG_REPLAY_FAILED__\n");
    if (domain_result)
        *domain_result = code;
    __asm__ volatile(
        "1: auipc t0, %%pcrel_hi(pg_replay_exit_frame)\n"
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

__attribute__((noreturn)) void
pg_subpool_refuse(const char *what)
{
    (void) what;                             /* the caller already said it */
    give_up(0xBAD8BAD8u);
}

#define REPLAY_DIE(msg) die(msg)
#define REPLAY_DIE_AT(i, msg, id) die_at((i), (msg), (id))
#define REPLAY_ALLOC(n) scratch_alloc(n)
#include "replay_core.inc"

__asm__(
    "  .text\n"
    "  .globl domain_main\n"
    "domain_main:\n"
    "1: auipc t0, %pcrel_hi(pg_replay_exit_frame)\n"
    "  addi t0, t0, %pcrel_lo(1b)\n"
    "  .insn r 0x5b, 0x1, 0xc, t0, gp, t0\n" /* cincoffset t0, gp, t0 */
    "  .insn s 0x5b, 0x4, sp, 0(t0)\n"       /* stc sp, 0(t0) */
    "  .insn s 0x5b, 0x4, ra, 16(t0)\n"      /* stc ra, 16(t0) */
    "  j pg_replay_domain_main\n");

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

void
pg_replay_domain_main(unsigned *res, unsigned func)
{
    if (func == CAPSTONE_DPI_REGION_SHARE) {
        switch (shares) {
            case 0: meta = (volatile struct pg_hostcall_v0 *) res; break;
            case 1: payload = (volatile char *) res; break;
            /* In the register the monitor put it in, and not a line later. */
            case 2: arena_type = pg_subpool_arena((void *) res,
                                                  PG_REPLAY_ARENA_SIZE); break;
            case 3: trace = (char *) res; break;
            case 4: scratch = (char *) res; break;
            default: break;
        }
        ++shares;
        return;
    }
    if (!meta || !payload || !trace || !scratch || shares < 5) {
        *res = 0xBAD0BAD0u;                  /* the shares never arrived */
        return;
    }
    domain_result = res;
    meta->length = 0;
    pg_domain_payload((char *) payload, (unsigned long *) &meta->length,
                      PG_REPLAY_PAYLOAD_SIZE);
    pg_domain_text("__CAPSTONE_PG_REPLAY_ENTRY__\n");

    struct a11_head *h = (struct a11_head *) trace;
    if (h->magic[0] != 'A' || h->magic[1] != '1' || h->magic[2] != '1'
        || h->recsize != sizeof(struct a11_rec) || h->endian != A11_ENDIAN
        || h->version != A11_VERSION || h->ppid != 0) {
        fail("the region does not hold a flattened trace this reader agrees with");
        give_up(0xBAD1BAD1u);
    }

    /* The identity tables come out of the scratch region. The footer says how
       big they have to be, and the host was told the same number. */
    struct a11_rec *r = (struct a11_rec *) (trace + sizeof *h);
    unsigned long n = 0;
    while ((char *) &r[n] + sizeof *r <= trace + PG_REPLAY_TRACE_SIZE
           && r[n].op != A11_END)
        n++;
    if ((char *) &r[n] + sizeof *r > trace + PG_REPLAY_TRACE_SIZE) {
        fail("no footer in the trace region: it is truncated");
        give_up(0xBAD2BAD2u);
    }
    n++;                                     /* the footer is a record too */

    unsigned long tables = (r[n - 1].s2 + 2) * 16UL + (r[n - 1].s3 + 2) * 16UL;

    tables = (tables + 4095UL) & ~4095UL;
    if (tables > PG_REPLAY_SCRATCH_SIZE) {
        fail("the scratch region cannot hold the identity tables");
        pg_domain_text("  wanted "); pg_domain_uint(tables);
        pg_domain_text(" bytes, the region is ");
        pg_domain_uint(PG_REPLAY_SCRATCH_SIZE);
        pg_domain_text("\n");
        give_up(0xBAD3BAD3u);
    }
    if (arena_type != 0) {
        fail("the arena is not a linear region");
        pg_domain_text("  its type is "); pg_domain_uint(arena_type);
        pg_domain_text(", where 0 is linear and 1 is non-linear.\n"
                       "  A handle senior to a region that is not linear has "
                       "nothing to revoke, so the host\n"
                       "  has to be told to share it linear: pass "
                       "--linear-arena.\n");
        give_up(0xBAD4BAD4u);
    }
    scratch_next = scratch;
    scratch_end = scratch + tables;

    struct replay_counts c;
    for (size_t i = 0; i < sizeof c; i++)
        ((char *) &c)[i] = 0;
    replay_run(r, n, &c);

    /*
     * What the manager asked, what the level below did, and what the hardware
     * was asked to do. The three are different questions and the table says
     * so: the first is the trace's own, the second is this port's bookkeeping,
     * and the third is the primitives, which is what the board's
     * revocation-node pool is spent on.
     */
    const struct sublet_stats *s = pg_subpool_primitives();
    struct pg_subpool_counts *k = &pg_subpool_counts;

    pg_domain_text("| what | the trace asked for |\n|---|---:|\n");
    row("create", 0, c.create);
    row("alloc", 0, c.alloc);
    row("free", 0, c.free);
    row("reset", 0, c.reset);
    row("delete", 0, c.delete);

    /*
     * Blocks are compared with the keepers taken out, and that is not a
     * convenience. Upstream keeps the keeper block across a reset; this port
     * revokes it with everything else and carves it again, because a
     * revocation cannot spare part of what it covers. Those carves are real
     * work the other arm does not do, so they are reported on their own line
     * and the line above them is the like-for-like one.
     */
    pg_domain_text("\n| the level below | in the backend | here |\n|---|---:|---:|\n");
    row("blocks taken, keepers apart", c.was_alloc, k->blocks - k->keepers);
    row("blocks held at once, most", c.was_peak, k->blocks_peak);
    row("keepers carved again after a reset", 0, k->keepers);
    row("blocks given back", 0, k->blocks_freed);
    row("blocks still held at the end", 0, k->blocks_live);

    pg_domain_text("\n| the discipline | count |\n|---|---:|\n");
    /*
     * Contexts created here is far below what the trace asked for, and that is
     * the manager's own doing: aset.c caches a deleted context on its own
     * freelist and the next create with matching parameters recycles it,
     * header and sub-pool and all. So this counts the contexts that ever
     * needed a sub-pool, which is what the arena has to hold.
     */
    row("contexts that needed a sub-pool", c.create, k->created);
    row("contexts destroyed outright", 0, k->destroyed);
    row("sub-pools at the peak", 0, k->pools_peak);
    row("second sub-pools handed out", 0, k->grown);
    /*
     * The trace's teardowns and the port's differ, and the difference is the
     * manager's own: mcxt.c does not call the reset method on a context that
     * is already reset, so a delete that follows a reset never reaches the
     * level below. The recording predicted that number before the port
     * existed, from the allocations each context had made.
     */
    row("teardowns the trace asked for", c.reset + c.delete, k->resets);
    row("of those, one revocation each", 0, k->revocations - k->extra_revocations);
    row("revocations of a second sub-pool", 0, k->extra_revocations);
    row("teardowns with nothing to revoke", 0, k->resets_empty);
    row("revocations in total", 0, k->revocations);
    row("chunks carved", 0, k->carves);
    row("chunks handed out", c.alloc, k->hands);
    row("chunks dropped", c.free, k->drops);
    row("chunk entries at the peak", 0, k->entries_peak);

    pg_domain_text("\n| primitive | times |\n|---|---:|\n");
    row("split", 0, s->split);
    row("mrev", 0, s->mrev);
    row("delin", 0, s->delin);
    row("revoke", 0, s->revoke);
    row("revocation nodes spent", 0, s->split + s->mrev);

    /*
     * The claim, checked here rather than by a host reading the numbers, and
     * as an identity rather than a bound:
     *
     *   revocations == teardowns - the ones with nothing to revoke
     *                  + the revocations of second sub-pools
     *
     * One per teardown of a context that holds something, and one more for
     * each further sub-pool that context was given, which it keeps for the
     * rest of its life. Nothing else can contribute, and if anything did the
     * two sides would differ.
     */
    unsigned long expect = k->resets - k->resets_empty + k->extra_revocations;

    pg_domain_text(k->revocations == expect
                   ? "__CAPSTONE_PG_SUBLET_ONE_EACH__\n"
                   : "__CAPSTONE_PG_SUBLET_NOT_ONE_EACH__\n");
    pg_domain_text(k->blocks == k->blocks_freed + k->blocks_live
                   ? "__CAPSTONE_PG_REPLAY_BALANCED__\n"
                   : "__CAPSTONE_PG_REPLAY_UNBALANCED__\n");
    pg_domain_text("__CAPSTONE_PG_REPLAY_DONE__\n");
    *res = 0x9C900000u | (unsigned) (c.delete & 0xFFFFu);
}
