/* Replay a recorded PostgreSQL workload against PostgreSQL's memory manager,
 * inside a Capstone domain.
 *
 * The loop is replay_core.inc, the same one the host driver runs. This file is
 * the domain's half of it: it is handed regions instead of opening a file, and
 * it writes into the payload the host prints after the domain returns, because
 * a domain has no stdout.
 *
 * The regions, in the order the host shares them:
 *
 *   0  the hostcall metadata, whose `length` is how much of the payload is written
 *   1  the payload
 *   2  the arena: the driver's identity tables at the front, the level below the rest
 *   3  the trace, flattened, header and records
 *
 * What the measurement is about is what the manager asks of the level below,
 * and that level is pg_level0.c here rather than glibc. The counters it keeps
 * are printed beside what the backend's manager asked, which the trace itself
 * carries, so the domain says whether the two agree without a host reading it.
 */
#include "postgres.h"
#include <stddef.h>
#include <stdint.h>

#include "a11trace.h"

/* The level below for this arm: pg_level0.c, a first-fit allocator over one
 * region, in place of glibc. What the measurement is about is what the manager
 * asks of it, so its counters are printed beside what the backend's manager
 * asked, which the trace itself carries. */
void pg_level0_init(void *region, size_t bytes);
extern unsigned long pg_level0_taken, pg_level0_given, pg_level0_grown;
extern unsigned long pg_level0_live, pg_level0_peak;

/* Everything about being a domain, shared with the other two drivers. */
#include "domain_frame.inc"
#include "replay_core.inc"

/* The regions this arm is handed, beyond the two every driver gets. The
 * identity tables come off the front of the arena and the level below gets
 * the rest, so they never pass its counters. */
static char *arena;
static char *trace;

void
pg_domain_entry(unsigned *res, unsigned func)
{
    if (func == CAPSTONE_DPI_REGION_SHARE) {
        switch (shares) {
            case 0: meta = (volatile struct pg_hostcall_v0 *) res; break;
            case 1: payload = (volatile char *) res; break;
            case 2: arena = (char *) res; break;
            case 3: trace = (char *) res; break;
            default: break;
        }
        ++shares;
        return;
    }
    if (!meta || !payload || !arena || !trace) {
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

    /* The identity tables come off the front of the arena, the level below
       gets the rest. The footer says how big they have to be. */
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

#ifdef REPLAY_CHECK_DATA
    /* The data check's third table, of lengths, at four bytes an object. */
    tables += (r[n - 1].s3 + 2) * 4UL;
#endif
    tables = (tables + 4095UL) & ~4095UL;
    if (tables + (1UL << 20) > PG_REPLAY_ARENA_SIZE) {
        fail("the arena cannot hold the identity tables and a heap");
        pg_domain_text("  wanted "); pg_domain_uint(tables);
        pg_domain_text(" bytes for the tables\n");
        give_up(0xBAD3BAD3u);
    }
    scratch_next = arena;
    scratch_end = arena + tables;
    pg_level0_init(arena + tables, PG_REPLAY_ARENA_SIZE - tables);

    struct replay_counts c;
    for (size_t i = 0; i < sizeof c; i++)
        ((char *) &c)[i] = 0;
    replay_run(r, n, &c);

    unsigned long got_taken = pg_level0_taken, got_given = pg_level0_given,
                  got_grown = pg_level0_grown, got_peak = pg_level0_peak;

    pg_domain_text("| what | the trace asked for |\n|---|---:|\n");
    row("create", 0, c.create);
    row("alloc", 0, c.alloc);
    row("free", 0, c.free);
    row("reset", 0, c.reset);
    row("delete", 0, c.delete);
#ifdef REPLAY_CHECK_DATA
    row("objects whose contents were read back", 0, c.checked);
#endif
    pg_domain_text("\n| the level below | in the backend | here |\n|---|---:|---:|\n");
    row("blocks taken", c.was_alloc, got_taken);
    row("blocks given back", c.was_free, got_given);
    row("blocks grown or moved", c.was_realloc, got_grown);
    row("blocks held at once, most", c.was_peak, got_peak);

    /* The size classes differ from the host's, because a capability does not
       fit in an eight-byte chunk (port/aset-capstone.patch), so the block
       counts are not expected to match the backend's. What must match is the
       calls, and that the manager gave back what it took. */
    pg_domain_text(got_given + pg_level0_live == got_taken
                   ? "__CAPSTONE_PG_REPLAY_BALANCED__\n"
                   : "__CAPSTONE_PG_REPLAY_UNBALANCED__\n");
    pg_domain_text("__CAPSTONE_PG_REPLAY_DONE__\n");
    *res = 0x9C900000u | (unsigned) (c.delete & 0xFFFFu);
}
