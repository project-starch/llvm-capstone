/* Does the level below keep its promise? A domain that asks the hardware.
 *
 * The claim the paper makes about PostgreSQL is that a context's whole heap
 * dies in one revocation. port/freestanding/pg_subpool.c is the piece that has
 * to make that true, and it is the piece to test before the manager is patched
 * to call it, because a fault inside a patched aset.c would be hard to read
 * back to its cause.
 *
 * Seven scenarios, each one a claim the design makes:
 *
 *   1  a sub-pool hands out blocks, and chunks out of the blocks, and what is
 *      written through a handed-out alias reads back
 *   2  a reset is one revocation, and the sub-pool works again afterwards,
 *      which is the part only the hardware can answer: a revoke that killed a
 *      linear child hands the region back uninitialised, and init is refused
 *      until it has been written through
 *   3  a reset of a sub-pool nothing was carved from is no revocation at all
 *   4  a context that fills its sub-pool gets a second one, and then its reset
 *      is two revocations and no more
 *   5  a delete gives the sub-pool back, and the next create reuses it
 *   6  two thousand creates and deletes leave no sub-pool and no entry behind
 *   7  a block given back before a reset returns its chunk entries
 *
 * What is deliberately not tested here is that a revoked alias faults. It
 * does, and that is the point of the hardware, but a fault ends the domain, so
 * a test that provoked one could not also report the six results beside it.
 * capstone/tests/runtime-qemu/borrow-revoke-uaf-probe is where that is asked.
 */
#include <stddef.h>
#include <stdint.h>
#include "pg_subpool.h"

/* Its own marker, so a failure here is not read as a replay's. */
#define PG_DOM_FAIL_MARKER "__CAPSTONE_PG_SUBPOOL_FAILED__\n"

/* Everything about being a domain, shared with the two replay drivers. This
 * file drives no replay, so it includes no loop. */
#include "domain_frame.inc"

/* ---- how a scenario reports ---------------------------------------------- */
/* What the level below said about the arena it was handed, in the share
 * handler, because a linear capability assigned to a C pointer and read back
 * on a later call comes back non-linear. */
static unsigned long arena_type = 7;

static unsigned failures;

static void
claim(const char *what, unsigned long want, unsigned long got)
{
    pg_domain_text(want == got ? "| ok | " : "| FAILED | ");
    pg_domain_text(what);
    pg_domain_text(" | ");
    pg_domain_uint(want);
    pg_domain_text(" | ");
    pg_domain_uint(got);
    pg_domain_text(" |\n");
    if (want != got)
        failures++;
}

/* A pattern that tells a stale read from a fresh one: the epoch in the high
 * byte, the offset in the low ones, so a byte that survived a reset is not
 * mistaken for a byte the next carve wrote. */
static unsigned long
pattern(unsigned epoch, unsigned long off)
{
    return ((unsigned long) epoch << 56) | (off * 0x0101010101ul & 0xFFFFFFFFFFFFFFul);
}

/* Hand out `n` chunks from `b`, write the pattern through each alias, read it
 * back. Returns how many read back wrong. */
static unsigned
fill(pg_block *b, unsigned n, unsigned long bytes, unsigned epoch,
     unsigned int *idx)
{
    unsigned bad = 0;

    for (unsigned k = 0; k < n; k++) {
        unsigned int i = pg_subpool_carve(b, bytes);

        idx[k] = i;
        if (!i) {
            bad++;
            continue;
        }
        unsigned long *p = (unsigned long *) pg_subpool_hand(i);

        if (!p) {
            bad++;
            continue;
        }
        for (unsigned long w = 0; w < bytes / sizeof *p; w++)
            p[w] = pattern(epoch, k * 64 + w);
        for (unsigned long w = 0; w < bytes / sizeof *p; w++)
            if (p[w] != pattern(epoch, k * 64 + w))
                bad++;
    }
    return bad;
}

void pg_domain_entry(unsigned *res, unsigned func);

void
pg_domain_entry(unsigned *res, unsigned func)
{
    if (func == CAPSTONE_DPI_REGION_SHARE) {
        switch (shares) {
            case 0: meta = (volatile struct pg_hostcall_v0 *) res; break;
            case 1: payload = (volatile char *) res; break;
            /* The arena goes into its slot here and not later, and this is not
             * a matter of taste. A region arrives linear in a register, and a
             * linear capability that is assigned to a C pointer and read back
             * from it on a later call comes back non-linear: the first run of
             * this test said so, type 1 where it wanted 0, and a handle senior
             * to a non-linear region has nothing to revoke. So the share is
             * taken while it is still in the register the monitor put it in. */
            case 2: arena_type = pg_subpool_arena((void *) res,
                                                  PG_REPLAY_ARENA_SIZE); break;
            default: break;                  /* a fourth share is ignored */
        }
        ++shares;
        return;
    }
    if (!meta || !payload || shares < 3) {
        *res = 0xBAD0BAD0u;
        return;
    }
    domain_result = res;
    meta->length = 0;
    pg_domain_payload((char *) payload, (unsigned long *) &meta->length,
                      PG_REPLAY_PAYLOAD_SIZE);
    pg_domain_text("__CAPSTONE_PG_SUBPOOL_ENTRY__\n");

    if (arena_type != 0) {
        unsigned long ty = arena_type;

        pg_domain_text("the arena is not a linear region, its type is ");
        pg_domain_uint(ty);
        pg_domain_text("\n  0 linear, 1 non-linear, 2 revocable, "
                       "3 uninitialised, 7 an empty slot.\n"
                       "  A handle senior to a region that is not linear has "
                       "nothing to revoke, so the\n"
                       "  host has to hand this region over linear.\n");
        give_up(0xBAD4BAD4u);
    }

    pg_domain_text("| | claim | want | got |\n|---|---|---:|---:|\n");

    unsigned int idx[64];

    /* 1: blocks, chunks, and what is written reads back */
    pg_subpool *a = pg_subpool_create();

    if (!a)
        give_up(0xBAD1BAD1u);
    pg_block *b1 = pg_subpool_block(a, 8192);
    pg_block *b2 = pg_subpool_block(a, 8192);

    claim("two blocks out of one sub-pool", 2,
          (b1 ? 1u : 0u) + (b2 ? 1u : 0u));
    claim("the two blocks do not overlap", 1,
          b1 && b2 && (b1->endptr <= b2->base || b2->endptr <= b1->base));
    claim("sixteen chunks written and read back", 0, fill(b1, 16, 64, 1, idx));
    claim("sixteen carves counted", 16, pg_subpool_counts.carves);

    /* 2: a reset is one revocation, and the sub-pool works afterwards */
    unsigned long before = pg_subpool_counts.revocations;

    pg_subpool_reset(a);
    claim("a reset is one revocation", 1,
          pg_subpool_counts.revocations - before);

    pg_block *b3 = pg_subpool_block(a, 8192);

    claim("a block after the reset", 1, b3 != NULL);
    claim("sixteen chunks after the reset, written and read back", 0,
          b3 ? fill(b3, 16, 64, 2, idx) : 1u);

    /* 3: a reset of a sub-pool nothing was carved from */
    pg_subpool *empty = pg_subpool_create();

    if (!empty)
        give_up(0xBAD5BAD5u);
    before = pg_subpool_counts.revocations;
    pg_subpool_reset(empty);
    claim("a reset with nothing carved is no revocation", 0,
          pg_subpool_counts.revocations - before);
    claim("and it is counted as such", 1, pg_subpool_counts.resets_empty);

    /* 4: a context that fills its sub-pool gets a second one */
    pg_subpool *big = pg_subpool_create();
    unsigned blocks = 0;

    if (!big)
        give_up(0xBAD6BAD6u);

    while (pg_subpool_block(big, 8192))
        blocks++;
    claim("a 64 KiB sub-pool holds this many 8 KiB blocks", 8, blocks);
    claim("a second sub-pool for it", 1, pg_subpool_grow(big, PG_SUBPOOL_BYTES));
    claim("and a block comes out of the second", 1,
          pg_subpool_block(big, 8192) != NULL);
    before = pg_subpool_counts.revocations;
    pg_subpool_reset(big);
    claim("its reset is two revocations and no more", 2,
          pg_subpool_counts.revocations - before);

    /* 5: a delete gives the sub-pool back, and a create reuses it */
    unsigned long live = pg_subpool_counts.pools_live;

    pg_subpool_destroy(big);
    claim("a delete of a grown context gives back two sub-pools", 2,
          live - pg_subpool_counts.pools_live);
    pg_subpool_destroy(empty);
    pg_subpool_destroy(a);
    claim("nothing is left of the four", 0, pg_subpool_counts.pools_live);

    /* 6: two thousand creates and deletes leave nothing behind */
    unsigned long entries = pg_subpool_counts.entries_live;

    for (unsigned k = 0; k < 2000; k++) {
        pg_subpool *sp = pg_subpool_create();

        if (!sp) {
            claim("two thousand creates", 2000, k);
            break;
        }
        pg_block *b = pg_subpool_block(sp, 8192);

        if (b)
            fill(b, 4, 64, 3, idx);
        pg_subpool_destroy(sp);
    }
    claim("no sub-pool left after two thousand rounds", 0,
          pg_subpool_counts.pools_live);
    claim("no entry left either", entries, pg_subpool_counts.entries_live);

    /* 7: a block given back before a reset returns its chunk entries */
    pg_subpool *c = pg_subpool_create();
    pg_block *b4 = c ? pg_subpool_block(c, 8192) : NULL;

    claim("a sub-pool and a block for the last scenario", 1, b4 != NULL);
    if (!b4)
        give_up(0xBAD7BAD7u);
    fill(b4, 8, 64, 4, idx);
    entries = pg_subpool_counts.entries_live;
    pg_subpool_block_free(b4);
    claim("a freed block returns its eight entries", 8,
          entries - pg_subpool_counts.entries_live);
    pg_subpool_destroy(c);

    /* What the run spent in revocation nodes, which is the constraint the
     * board imposes and not the emulator: the node allocator is a bump head
     * with no reclamation on silicon, so what counts is every split and every
     * mrev ever done and not how many are live. sublet.h keeps the tally. */
    /* sublet.h keeps its tally per translation unit and the primitives ran in
     * the level below's, so the numbers come from there and not from this
     * file's own copy, which is all zeros. */
    const struct sublet_stats *s = pg_subpool_primitives();

    pg_domain_text("\n| primitive | times |\n|---|---:|\n");
    claim("split", s->split, s->split);
    claim("mrev", s->mrev, s->mrev);
    claim("delin", s->delin, s->delin);
    claim("a node each, so nodes spent", s->split + s->mrev,
          s->split + s->mrev);

    pg_domain_text("\n| what | count |\n|---|---:|\n");
    claim("revocations", pg_subpool_counts.revocations,
          pg_subpool_counts.revocations);
    claim("handles taken", pg_subpool_counts.handles,
          pg_subpool_counts.handles);
    claim("carves", pg_subpool_counts.carves, pg_subpool_counts.carves);
    claim("hands out", pg_subpool_counts.hands, pg_subpool_counts.hands);
    claim("sub-pools at the peak", pg_subpool_counts.pools_peak,
          pg_subpool_counts.pools_peak);
    claim("entries at the peak", pg_subpool_counts.entries_peak,
          pg_subpool_counts.entries_peak);

    pg_domain_text(failures ? "__CAPSTONE_PG_SUBPOOL_BAD__\n"
                            : "__CAPSTONE_PG_SUBPOOL_GOOD__\n");
    pg_domain_text("__CAPSTONE_PG_SUBPOOL_DONE__\n");
    *res = failures ? (0xBAD5000u | failures) : 0x9C9C0000u;
}
