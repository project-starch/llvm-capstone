/* Does an ancestor's withdrawal reach through a REAL manager, and stop where it
 * should?
 *
 * security-tests/capstone/subpool-lifetimes.c asks the level below the same family of questions and
 * answers them, and the security table says so. What it cannot answer is
 * whether any of it survives contact with PostgreSQL's own memory manager,
 * because the level below has no children: its sub-pools are regions, and a
 * region does not create regions. A context does. So the questions are asked
 * again here, over unmodified mcxt.c and an aset.c with one patch, with the
 * context TREE doing the delegating rather than a carve.
 *
 * That distinction is the whole reason this file exists. An ancestor that
 * revokes a region is one revocation. An ancestor that deletes a context is
 * mcxt.c walking firstchild and nextchild, deleting every descendant first, and
 * each of those is an allocator with its own sub-pool and its own handle.
 * Nothing in this driver tells it to do that: MemoryContextDelete is
 * upstream's, and the children never agree to anything.
 *
 * THE SHAPE OF A CONTEXT DECIDES WHAT ITS DELETE DOES, and a driver that used
 * only one shape would report half of this and call it the whole. Upstream
 * aset.c keeps a freelist of deleted contexts whose size parameters are one of
 * two standard sets, and AllocSetDelete returns such a context to it INSTEAD of
 * destroying it. The port keeps that policy, so the two shapes answer
 * differently and both answers are the port's:
 *
 *   a standard shape   delete revokes the context's sub-pool and PARKS it. The
 * memory is protected and is not back in the arena, and the next context of
 * that shape takes it a custom shape     freeListIndex is -1, so delete revokes
 * and the sub-pool goes back to the level below, where the arena can carve it
 * for anything
 *
 * The scenarios, in the order they run:
 *
 *   1  four contexts, a root with two children and a grandchild under one of
 * them, every object written and read back. Without this the scenarios below
 * are about nothing 2  the root deletes one child of the STANDARD shape. The
 * grandchild goes with it, the SIBLING and the root keep reading and keep
 * writing, and it costs one revocation per context. The sub-pools are parked
 * rather than returned, and the next create takes a parked one 3  the same tree
 * in the CUSTOM shape, and now the two sub-pools go back to the arena. A
 *      sibling that merely still reads could be reading memory nobody
 * reclaimed, so the accounting is half of the claim and the reads are the other
 * half 4  the root is RESET rather than deleted. Upstream's reset deletes
 * children first, so this is the ancestor withdrawal again by a different
 * route, and the root has to work afterwards. Only the hardware answers that: a
 * revoke that killed a linear child hands the region back uninitialised, and
 * init is refused until it has been written through 5  depth, which the
 * experiment design asks for by name. Chains of 2, 3 and 8 custom-shape
 *      contexts, an object at the bottom of each, the TOP deleted. What the
 * extra levels cost in revocation nodes is reported rather than assumed,
 * because a node budget is the thing this prototype runs out of first
 *
 * WHAT IS DELIBERATELY NOT HERE. That an alias into a deleted context faults.
 * It does, and it is the point, but a fault ends the domain and a domain that
 * faulted cannot report the scenarios beside it. That question has its own
 * image in the nginx port, and the level below's own answer is in
 * subpool_domain.c.
 */
#include "postgres.h"

#include <stddef.h>
#include <stdint.h>

#include "pg_subpool.h"
#include "utils/memutils.h"

/* Its own marker, so a failure here is not read as a replay's. */

#include "domain-runtime.h"

static unsigned long arena_type = 7;
static unsigned failures;

static void claim(const char *what, unsigned long want, unsigned long got) {
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

/* The epoch is in the high byte so a byte that survived a delete is not
 * mistaken for a byte the next allocation wrote. Both halves of scenario 2
 * depend on telling those apart. */
static unsigned long pattern(unsigned epoch, unsigned long off) {
  return ((unsigned long)epoch << 56) |
         (off * 0x0101010101UL & 0xFFFFFFFFFFFFFFUL);
}

#define WORDS 16 /* 128 bytes an object, small enough to share a block */

static void fill(unsigned long *p, unsigned epoch) {
  for (unsigned long w = 0; w < WORDS; w++)
    p[w] = pattern(epoch, w);
}

static unsigned holds(unsigned long *p, unsigned epoch) {
  for (unsigned long w = 0; w < WORDS; w++)
    if (p[w] != pattern(epoch, w))
      return 0;
  return 1;
}

/* Reads it, then writes it again and reads that. A survivor that can only be
 * read might be a survivor whose region was revoked and whose bytes merely have
 * not been reused yet. */
static unsigned still_works(unsigned long *p, unsigned epoch, unsigned next) {
  if (!holds(p, epoch))
    return 0;
  fill(p, next);
  return holds(p, next);
}

/* The standard shape, which aset.c parks on its own freelist when it is
 * deleted. */
static MemoryContext ctx_std(MemoryContext parent, const char *name) {
  return AllocSetContextCreateInternal(parent, name, ALLOCSET_SMALL_MINSIZE,
                                       ALLOCSET_SMALL_INITSIZE,
                                       ALLOCSET_SMALL_MAXSIZE);
}

/* The custom shape. The initial block size is neither of the two aset.c
 * recognises, so freeListIndex is -1 and a delete really destroys. Backends
 * create both kinds, and only this one lets a delete be watched all the way
 * back to the arena. */
#define CUSTOM_INITSIZE (2 * 1024)

static MemoryContext ctx_custom(MemoryContext parent, const char *name) {
  return AllocSetContextCreateInternal(parent, name, 0, CUSTOM_INITSIZE,
                                       ALLOCSET_SMALL_MAXSIZE);
}

/* What the refusing malloc in sublet/unsupported-allocators.c calls when a
 * context type the port does not cover asks for memory. It ends the domain at a
 * named place instead of handing out a heap nothing revokes. Only aset contexts
 * are created here, so this is a guard and not a path. */
__attribute__((noreturn)) void pg_subpool_refuse(const char *what) {
  (void)what; /* the caller already said it */
  give_up(0xBAD8BAD8u);
}

void pg_domain_entry(unsigned *res, unsigned func);

void pg_domain_entry(unsigned *res, unsigned func) {
  if (func == CAPSTONE_DPI_REGION_SHARE) {
    switch (shares) {
    case 0:
      meta = (volatile struct pg_hostcall_v0 *)res;
      break;
    case 1:
      payload = (volatile char *)res;
      break;
    /* In the register the monitor put it in, and not a line later: a linear
     * capability assigned to a C pointer and read back on a later call comes
     * back non-linear. */
    case 2:
      arena_type = pg_subpool_arena((void *)res, PG_REPLAY_ARENA_SIZE);
      break;
    default:
      break; /* the host shares more and this driver wants three */
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
  pg_domain_payload((char *)payload, (unsigned long *)&meta->length,
                    PG_REPLAY_PAYLOAD_SIZE);
  pg_domain_text("__CAPSTONE_PG_HIER_ENTRY__\n");

  if (arena_type != 0) {
    fail("the arena is not a linear region");
    pg_domain_text("  its type is ");
    pg_domain_uint(arena_type);
    pg_domain_text(
        ", where 0 is linear and 1 is non-linear.\n"
        "  A handle senior to a region that is not linear has nothing to "
        "revoke, so the host\n  has to be told to share it linear: pass "
        "--linear-arena.\n");
    give_up(0xBAD4BAD4u);
  }

  pg_domain_text("| | claim | want | got |\n|---|---|---:|---:|\n");

  /* ---- 1: the tree, and everything in it reads back --------------------- */
  MemoryContext a = ctx_std(NULL, "root");

  TopMemoryContext = CurrentMemoryContext = a;

  MemoryContext b = ctx_std(a, "child b");
  MemoryContext c = ctx_std(a, "child c");
  MemoryContext d = ctx_std(b, "grandchild d");

  unsigned long *pa = (unsigned long *)MemoryContextAlloc(a, WORDS * 8);
  unsigned long *pb = (unsigned long *)MemoryContextAlloc(b, WORDS * 8);
  unsigned long *pc = (unsigned long *)MemoryContextAlloc(c, WORDS * 8);
  unsigned long *pd = (unsigned long *)MemoryContextAlloc(d, WORDS * 8);

  claim("four contexts, four objects", 4,
        (pa != NULL) + (pb != NULL) + (pc != NULL) + (pd != NULL));
  if (!pa || !pb || !pc || !pd)
    give_up(0xBAD6BAD6u);

  fill(pa, 1);
  fill(pb, 2);
  fill(pc, 3);
  fill(pd, 4);
  claim("each reads back what it wrote", 4,
        holds(pa, 1) + holds(pb, 2) + holds(pc, 3) + holds(pd, 4));

  /* Each aset context takes a sub-pool of its own, which is the port's shape
   * and the reason a delete can be one revocation per context and not a walk of
   * chunks. */
  claim("a sub-pool per context", 4, pg_subpool_counts.pools_live);

  /* ---- 2: the root deletes one child, standard shape -------------------- */
  unsigned long pools_before = pg_subpool_counts.pools_live;
  unsigned long revokes_before = pg_subpool_counts.revocations;
  unsigned long made_before = pg_subpool_counts.created;

  /* Nothing here asks b or d to cooperate, and neither is told. mcxt.c walks
   * the children. */
  MemoryContextDelete(b);

  claim("one revocation per context, the grandchild included", 2,
        pg_subpool_counts.revocations - revokes_before);

  /* The sibling and the parent are the point. A revoke that followed the arena,
   * or one that took the parent's own block because the child was carved out of
   * it, would show here. */
  claim("the sibling still reads and still writes", 1, still_works(pc, 3, 5));
  claim("the root still reads and still writes", 1, still_works(pa, 1, 6));

  /* And the sub-pools were PARKED, not returned, because this is the shape
   * aset.c keeps a freelist of. Protected and not reclaimed is a different
   * thing from reclaimed, and a driver that only read the survivors would not
   * have noticed which of the two this is. */
  claim("their sub-pools are parked, not returned", pools_before,
        pg_subpool_counts.pools_live);

  /* The next context of that shape takes a parked one rather than carving the
   * arena. */
  MemoryContext e = ctx_std(a, "child e");
  unsigned long *pe = (unsigned long *)MemoryContextAlloc(e, WORDS * 8);

  claim("a new context after the delete gets memory", 1, pe != NULL);
  if (pe) {
    fill(pe, 7);
    claim("and it holds what is written through it", 1, holds(pe, 7));
  }
  claim("and it carved no new sub-pool", made_before,
        pg_subpool_counts.created);

  /* ---- 3: the same tree in the custom shape ----------------------------- */
  /* freeListIndex is -1 for these, so a delete destroys rather than parks and
   * the sub-pools are watched all the way back to the level below. */
  unsigned long live3 = pg_subpool_counts.pools_live;
  unsigned long gone3 = pg_subpool_counts.destroyed;

  MemoryContext f = ctx_custom(a, "custom f");
  MemoryContext g = ctx_custom(a, "custom g");
  MemoryContext h = ctx_custom(f, "custom grandchild h");

  unsigned long *pf = (unsigned long *)MemoryContextAlloc(f, WORDS * 8);
  unsigned long *pg = (unsigned long *)MemoryContextAlloc(g, WORDS * 8);
  unsigned long *ph = (unsigned long *)MemoryContextAlloc(h, WORDS * 8);

  claim("three custom contexts, three objects", 3,
        (pf != NULL) + (pg != NULL) + (ph != NULL));
  if (!pf || !pg || !ph)
    give_up(0xBAD7BAD7u);
  fill(pf, 11);
  fill(pg, 12);
  fill(ph, 13);
  claim("three more sub-pools", live3 + 3, pg_subpool_counts.pools_live);

  MemoryContextDelete(f); /* and h with it, neither asked */

  claim("the child and its grandchild go back to the arena", 2,
        pg_subpool_counts.destroyed - gone3);
  claim("two sub-pools fewer", live3 + 1, pg_subpool_counts.pools_live);
  claim("the custom sibling still reads and still writes", 1,
        still_works(pg, 12, 14));
  claim("the root still reads and still writes", 1, still_works(pa, 6, 15));

  /* ---- 4: the same withdrawal by the other route ------------------------ */
  /* Upstream's reset deletes the children first, so this is the ancestor
   * withdrawing from everything below it without any of them agreeing, and then
   * having to work itself. */
  unsigned long live4 = pg_subpool_counts.pools_live;
  unsigned long gone4 = pg_subpool_counts.destroyed;

  MemoryContextReset(a);

  /* Three children are alive at this point: c and e of the standard shape,
   * which park, and g of the custom one, which goes back. So the reset returns
   * exactly one sub-pool to the arena and destroys exactly one context, and
   * saying which is the point of having both shapes. */
  claim("the reset returns the custom child's sub-pool", live4 - 1,
        pg_subpool_counts.pools_live);
  claim("and destroys exactly that one", 1,
        pg_subpool_counts.destroyed - gone4);

  unsigned long *pa2 = (unsigned long *)MemoryContextAlloc(a, WORDS * 8);

  claim("and the root allocates again afterwards", 1, pa2 != NULL);
  if (pa2) {
    fill(pa2, 8);
    claim("through a region the revoke handed back uninitialised", 1,
          holds(pa2, 8));
  }

  /* ---- 5: depth, and what the extra levels cost ------------------------- */
  /* The design asks for total nesting depths 2, 3 and 8 with the terminal
   * object the same size, and for the additional nodes to be reported.
   * Custom-shape contexts, so no parked sub-pool makes one chain cheaper than
   * the next. Only split and mrev allocate a revocation node, so that sum is
   * the number, and it is read from the primitives rather than predicted. */
  static const unsigned depths[3] = {2, 3, 8};
  unsigned long nodes[3];

  /* EACH CHAIN IS BUILT AND TORN DOWN TWICE, AND THE SECOND ONE IS THE
   * MEASUREMENT. A sub-pool that has to be cut from the arena costs a split and
   * an mrev that a sub-pool taken off the level below's free list does not, so
   * a first chain prices the arena and a second prices the nesting. The first
   * run also leaves exactly as many sub-pools free as the second needs, which
   * is what makes the second one comparable across depths. Both runs check
   * every claim. */
  for (unsigned t = 0; t < 3; t++) {
    for (unsigned pass = 0; pass < 2; pass++) {
      const struct sublet_stats *s0 = pg_subpool_primitives();
      unsigned long n0 = s0->split + s0->mrev;
      unsigned long live0 = pg_subpool_counts.pools_live;

      MemoryContext chain = ctx_custom(a, "chain");
      MemoryContext top = chain;

      for (unsigned k = 1; k < depths[t]; k++)
        chain = ctx_custom(chain, "chain");

      unsigned long *deep =
          (unsigned long *)MemoryContextAlloc(chain, WORDS * 8);

      claim("an object at the bottom of the chain", 1, deep != NULL);
      if (deep) {
        fill(deep, 9);
        claim("which reads back", 1, holds(deep, 9));
      }
      claim("a sub-pool per level", depths[t],
            pg_subpool_counts.pools_live - live0);

      MemoryContextDelete(top); /* the top, and nothing below it is asked */

      claim("the whole chain goes with the top", live0,
            pg_subpool_counts.pools_live);

      const struct sublet_stats *s1 = pg_subpool_primitives();

      nodes[t] = s1->split + s1->mrev - n0;
    }
  }

  /* The point of the table below, stated as something that can fail. One more
   * level is one more context, which is one more sub-pool and one more keeper
   * block carved inside it. */
  claim("a level of nesting costs this many revocation nodes", 2,
        nodes[1] - nodes[0]);
  claim("and five more levels cost five times that", 10, nodes[2] - nodes[1]);

  claim("the root survived all three chains", 1, still_works(pa2, 8, 10));

  pg_domain_text("\n| depth | revocation nodes for the chain |\n|---|---:|\n");
  for (unsigned t = 0; t < 3; t++) {
    pg_domain_text("| ");
    pg_domain_uint(depths[t]);
    pg_domain_text(" | ");
    pg_domain_uint(nodes[t]);
    pg_domain_text(" |\n");
  }

  pg_domain_text("\n| what | count |\n|---|---:|\n");
  claim("revocations", pg_subpool_counts.revocations,
        pg_subpool_counts.revocations);
  claim("handles taken", pg_subpool_counts.handles, pg_subpool_counts.handles);
  claim("sub-pools at the peak", pg_subpool_counts.pools_peak,
        pg_subpool_counts.pools_peak);
  claim("sub-pools still live", pg_subpool_counts.pools_live,
        pg_subpool_counts.pools_live);

  pg_domain_text(failures ? "__CAPSTONE_PG_HIER_BAD__\n"
                          : "__CAPSTONE_PG_HIER_GOOD__\n");
  pg_domain_text("__CAPSTONE_PG_HIER_DONE__\n");
  *res = failures ? (0xBAD7000u | failures) : 0x9C7C0000u;
}
