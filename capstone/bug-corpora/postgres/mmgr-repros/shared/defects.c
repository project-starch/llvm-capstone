/* The eight PostgreSQL memory-context defects, as domain programs.
 *
 * One program, one defect per boot, chosen through the selection region -- the
 * port's own fixtures are built this way for a reason its source states: "a
 * fault ends the domain, so a test that provoked one could not also report the
 * results beside it".
 *
 * Each case is run twice against the SAME binary logic:
 *
 *   spatial  the manager over a plain first-fit backing allocator. Chunks are
 *            offsets inside one arena capability, so a freed chunk is still
 *            addressable and the stale access SUCCEEDS. Expected: complete.
 *   sublet   every chunk is its own capability, taken on hand-out and given
 *            back on free. The stale access is a revoked alias. Expected: a
 *            capability fault at the declared instruction.
 *
 * That pairing is the claim: bounds and provenance alone catch none of these
 * eight, and revocation catches all of them. Note what the spatial arm is and
 * is not -- it is this port's unprotected baseline, not a CHERI model; CHERI
 * would narrow bounds per allocation. It would still miss every case here,
 * because a reused chunk stays tagged and in bounds, but that argument is made
 * in prose and not by this arm.
 *
 * WHAT IS REDUCED
 *
 * The allocator is real: PostgreSQL 17.0's aset.c, mcxt.c and slab.c, compiled
 * unmodified but for the capability-ABI and Sublet patches the port applies.
 * The consumers are reduced to the allocator calls the upstream defect makes,
 * in the same order, because reaching them in place needs a backend, a planner,
 * a walsender or a concurrently dropped partition. Each case names its upstream
 * commit; the per-case PROVENANCE.md says line by line what was reduced.
 */
#include "domain-runtime.h"
#ifdef PG_DEFECTS_SUBLET
#include "pg_subpool.h"
#else
void pg_level0_init(void *, size_t);
#endif

#include "postgres.h"
#include "utils/memutils.h"
#include "utils/memutils_internal.h"

static unsigned long arena_type;
static const volatile unsigned *selection;
static unsigned char *volatile held;

_Noreturn void pg_subpool_refuse(const char *why) {
  fail(why);
  give_up(0xbad90001);
}

static void check(int condition) {
  if (!condition)
    give_up(0xbad90002);
}

/* The stale access, labelled so the host can require the fault to land HERE
 * rather than merely somewhere. */
__attribute__((noinline)) static unsigned
probe(const volatile unsigned char *p) {
  unsigned long value;
  __asm__ volatile(".globl pg_defect_probe\npg_defect_probe:\nlbu %0, 0(%1)"
                   : "=r"(value)
                   : "r"(p)
                   : "memory");
  return value;
}

__attribute__((noinline)) static void write_probe(volatile unsigned char *p) {
  __asm__ volatile(".globl pg_defect_write\npg_defect_write:\nsb %0, 0(%1)" ::"r"(
                       93UL),
                   "r"(p)
                   : "memory");
}

/* Publish the case and both probe addresses, so the host knows which
 * instruction a fault is allowed to be at. */
static void mark(unsigned which) {
  extern void pg_defect_probe(void), pg_defect_write(void);
  unsigned long code = 0xcf18000000000000UL | which;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x43, x0, %0, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %1, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %2, x0\n" ::"r"(code),
                   "r"(pg_defect_probe), "r"(pg_defect_write)
                   : "memory");
}

static MemoryContext aset_child(MemoryContext parent, const char *name) {
  return AllocSetContextCreateInternal(parent, name, ALLOCSET_SMALL_MINSIZE,
                                       ALLOCSET_SMALL_INITSIZE,
                                       ALLOCSET_SMALL_MAXSIZE);
}

/* sizeof(ReorderBufferChange) in 17.0, measured on the host rather than
 * assumed. Slab needs one fixed size per context. */
#define CHANGE_BYTES 80

static void defect(unsigned which) {
  MemoryContext root = AllocSetContextCreateInternal(NULL, "root", 0, 2048, 8192);
  TopMemoryContext = CurrentMemoryContext = root;

  if (which == 0) {
    /* Row 1 -- tuplestore, bug #19438, fix 1f5b6a5e5d.
     * dumptuples() frees tuples as it writes them but clears memtupdeleted
     * only after the loop, so a WRITETUP that throws leaves memtuples[]
     * holding freed chunks; tuplestore_end then frees them a second time.
     *
     * The stale access is the second pfree itself, so there is no probe: the
     * manager faults reading the revoked chunk's header before any bookkeeping
     * runs. Its oracle therefore accepts a fault anywhere, unlike the others. */
    unsigned char *memtuples[2];
    memtuples[0] = MemoryContextAlloc(root, 64);
    memtuples[1] = MemoryContextAlloc(root, 64);
    memtuples[0][0] = 11;
    pfree(memtuples[0]); /* WRITETUP freed it, then threw */
    mark(0);
    pfree(memtuples[0]); /* the end walk, from index 0 */
    held = memtuples[1];
  } else if (which == 1) {
    /* Row 2 -- vacuum dead_items, fix 3549ffb6af. The TidStore struct is
     * palloc'd in CurrentMemoryContext; dead_items_reset destroys and recreates
     * it but leaves vacrel->dead_items pointing at the old one, which the
     * context freelist hands straight back. */
    unsigned char *ts = MemoryContextAlloc(root, 64);
    ts[0] = 17;
    held = ts; /* vacrel->dead_items, never updated */
    pfree(ts); /* TidStoreDestroy */
    unsigned char *fresh = MemoryContextAlloc(root, 64); /* TidStoreCreate */
    fresh[0] = 19;
    mark(1);
    (void)probe(held);
  } else if (which == 2) {
    /* Row 3 -- the 2024 instance of the same defect, fix 83ce20d671, live only
     * because the port pins 17.0. Here the store's own context is deleted
     * first, which is what TidStoreDestroy does before freeing the struct. */
    MemoryContext rt = aset_child(root, "TID storage");
    unsigned char *ts = MemoryContextAlloc(root, 64);
    ts[0] = 23;
    held = ts;
    MemoryContextDelete(rt); /* MemoryContextDelete(ts->rt_context) */
    pfree(ts);               /* pfree(ts) */
    unsigned char *fresh = MemoryContextAlloc(root, 64);
    fresh[0] = 29;
    mark(2);
    (void)probe(held);
  } else if (which == 3) {
    /* Row 4 -- expand_partitioned_rtentry, fix ed394c4bdf. One Bitmapset, two
     * aliases; bms_del_member frees it through the field when the last member
     * goes, and the loop keeps reading the local. */
    unsigned char *set = MemoryContextAlloc(root, 64);
    set[0] = 31;
    unsigned char *field = set; /* relinfo->live_parts */
    held = set;                 /* the local live_parts */
    pfree(field);               /* bms_del_member emptied and freed it */
    unsigned char *fresh = MemoryContextAlloc(root, 64);
    fresh[0] = 37;
    mark(3);
    (void)probe(held);
  } else if (which == 4) {
    /* Row 5 -- free_child_join_sjinfo, bug #18806, fix 727bc6ac33f6. The child
     * SpecialJoinInfo shares its relid sets with the parent and frees them
     * unconditionally, once per partition pair. */
    unsigned char *relids = MemoryContextAlloc(root, 64);
    relids[0] = 41;
    unsigned char *child = relids; /* the child's copy of the pointer */
    held = relids;                 /* the parent still owns this */
    pfree(child);                  /* bms_free, in the child's cleanup */
    unsigned char *fresh = MemoryContextAlloc(root, 64);
    fresh[0] = 43;
    mark(4);
    (void)probe(held);
  } else if (which == 5) {
    /* Row 6 -- WindowAgg, fix 9d5ce4f1a00a. release_partition resets the
     * partition context in bulk; the top-level branch does not NULL the
     * by-ref results, so ecxt_aggvalues[] keeps pointing into it. The array
     * itself lives longer, which is what makes the stale read reachable. */
    MemoryContext partcontext = aset_child(root, "WindowAgg Partition");
    unsigned char *value = MemoryContextAlloc(partcontext, 64);
    value[0] = 47;
    held = value; /* econtext->ecxt_aggvalues[wfuncno] */
    MemoryContextReset(partcontext); /* release_partition */
    unsigned char *fresh = MemoryContextAlloc(partcontext, 64);
    fresh[0] = 53;
    mark(5);
    (void)probe(held);
  } else if (which == 6) {
    /* Row 7 -- pgoutput, fix a61592253e. entry_cxt is a grandchild of the
     * decoding context; an error tears that down, while RelationSyncCache
     * lives in CacheMemoryContext and keeps pointing into the dead arena. */
    MemoryContext decoding = aset_child(root, "logical decoding");
    MemoryContext entry_cxt = aset_child(decoding, "entry");
    unsigned char *filter = MemoryContextAlloc(entry_cxt, 64);
    filter[0] = 59;
    held = filter;                /* the cache entry's pointer */
    MemoryContextDelete(decoding); /* the error path, taking the grandchild */
    mark(6);
    (void)probe(held);
  } else if (which == 7) {
    /* Row 8 -- reorderbuffer, fix 9e0b4b1ab5. The change record comes from a
     * Slab context, so the free list is LIFO with one chunk size and the next
     * allocation returns the identical address every time. */
    MemoryContext change_context =
        SlabContextCreate(root, "Change", SLAB_DEFAULT_BLOCK_SIZE, CHANGE_BYTES);
    unsigned char *specinsert = MemoryContextAlloc(change_context, CHANGE_BYTES);
    specinsert[0] = 61;
    held = specinsert; /* change = specinsert, the loop cursor */
    pfree(specinsert); /* ReorderBufferReturnChange */
    unsigned char *successor =
        MemoryContextAlloc(change_context, CHANGE_BYTES);
    successor[0] = 67;
    check(successor == (unsigned char *)held); /* slab reuse is deterministic */
    mark(7);
    (void)probe(held);
  } else {
    give_up(0xbad90003);
  }
  (void)write_probe; /* the label must exist even where no case writes */
}

void pg_domain_entry(unsigned *res, unsigned func) {
  if (func == CAPSTONE_DPI_REGION_SHARE) {
    switch (shares++) {
    case 0:
      meta = (void *)res;
      break;
    case 1:
      payload = (void *)res;
      break;
    case 2:
#ifdef PG_DEFECTS_SUBLET
      arena_type = pg_subpool_arena(res, PG_REPLAY_ARENA_SIZE);
#else
      pg_level0_init(res, PG_REPLAY_ARENA_SIZE);
#endif
      break;
    case 3:
      selection = (void *)res;
      break;
    }
    return;
  }
  domain_result = res;
  check(shares >= 4 && meta && payload && selection && arena_type == 0);
  meta->length = 0;
  pg_domain_payload((char *)payload, (unsigned long *)&meta->length,
                    PG_REPLAY_PAYLOAD_SIZE);
  unsigned which = selection[0];
  check(which < 8);
  defect(which);
  /* Only the spatial arm is expected to arrive here. */
  pg_domain_text("__CAPSTONE_PG_DEFECT_COMPLETED__\n");
}
