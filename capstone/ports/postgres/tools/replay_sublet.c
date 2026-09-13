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

/* The level below for this arm: pg_subpool.c, a sub-pool per context under one
 * handle. The discipline is what this file reports on. */
#include "pg_subpool.h"

#ifndef PG_REPLAY_SCRATCH_SIZE
#define PG_REPLAY_SCRATCH_SIZE (48UL * 1024UL * 1024UL)
#endif

/* Everything about being a domain, shared with the other two drivers. */
#include "domain_frame.inc"
#include "replay_core.inc"

/* The regions this arm is handed, beyond the two every driver gets. The arena
 * is not among them because it never becomes a C pointer: it goes straight
 * into the level below's slot in the share handler, and arena_type is what
 * that returned. */
static char *trace;
static char *scratch;
static unsigned long arena_type = 7;

/* What the refusing malloc in port/freestanding/pg_subpool_libc.c calls when a
 * context type the port does not cover asks for memory. It ends the domain at
 * a named place instead of handing out a heap nothing revokes. */
__attribute__((noreturn)) void
pg_subpool_refuse(const char *what)
{
    (void) what;                             /* the caller already said it */
    give_up(0xBAD8BAD8u);
}

void
pg_domain_entry(unsigned *res, unsigned func)
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

    /* Two capability tables at sixteen bytes an entry, and with the data
       check on a third of lengths at four. The footer says how many identities
       were handed out, so this is exact rather than generous. */
    unsigned long tables = (r[n - 1].s2 + 2) * 16UL + (r[n - 1].s3 + 2) * 16UL;

#ifdef REPLAY_CHECK_DATA
    tables += (r[n - 1].s3 + 2) * 4UL;
#endif

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
#ifdef REPLAY_CHECK_DATA
    row("objects whose contents were read back", 0, c.checked);
#endif

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
