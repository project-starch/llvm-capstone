#ifndef CAPSTONE_BENCHMARKS_SQLITE_REVOKE_ON_FREE_ALLOC_H
#define CAPSTONE_BENCHMARKS_SQLITE_REVOKE_ON_FREE_ALLOC_H

/* A revoke-on-free, non-coalescing linear allocator (row3 fork B2).
 *
 * Every allocation is an INDEPENDENTLY REVOCABLE capability: xMalloc carves a
 * fresh SPLIT sub-capability off a monitor-granted linear arena (its own
 * revocation-tree node), MREVs it while it is still LIN, and hands back the
 * delin'd NONLIN alias; xFree REVOKEs that allocation's node. So a pointer
 * SQLite returned to its caller and then freed at sqlite3_finalize FAULTS on any
 * later dereference -- SQLite's own pointer, revoked by SQLite's own free path,
 * with no wrapper and no copy.
 *
 * WHY THIS EXISTS (see history/09-07-2026_23-05-00_...-steps-1-3.md, Step 3):
 * memsys5 cannot do this. Its allocations are `&zPool[i]`, a cincoffset that
 * inherits the pool's rev_node_id/type/bounds, so MREV of one would sweep the
 * whole heap; and the pool must be delin'd before SQLite copies it, after which
 * csmrev asserts. SPLIT is the only fresh-node derivation, and it is one-way
 * (the emulator has no merge/unsplit op). This allocator routes around all of
 * that by giving every allocation its own SPLIT node and NEVER coalescing:
 *
 *   - the arena only ever shrinks from the top (one-way SPLIT);
 *   - xFree revokes and marks the slot dead but cannot return space to the
 *     arena, so a long-running or realloc-heavy workload FRAGMENTS. That is the
 *     deliberate cost, and the generality limit measured in Phase 2.
 *
 * No SQLite dependency here: this is the raw allocator, used both by the Phase-0
 * feasibility probe and by the sqlite3_mem_methods wrapper.
 */

/* cssplit rd, rs1, rs2 -- split rs1 at address rs2. rs1 keeps [base, mid); rd
 * gets [mid, end) with a FRESH revocation-tree node, so the two are
 * independently revocable. No builtin/mnemonic exists, hence the raw R-type
 * encoding (Func7=0b0000110, Func3=001, opcode=0b1011011). The emulator makes
 * the op a no-op unless rd != rs1, hence the early-clobber on the result.
 * SHRINK/SHRINKTO are NOT a substitute: they copy rev_node_id unchanged. */
static inline void *rof_split(void **lo, unsigned long mid) {
  void *hi;
  void *l = *lo;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x06, %0, %1, %2"
                   : "=&r"(hi), "+r"(l)
                   : "r"(mid));
  *lo = l;
  return hi;
}

/* One live allocation. `rev` is the revocation handle (a capability, so this
 * struct must be 16-aligned and rev at offset 0). `base == 0` marks a free slot:
 * base is the allocation's start address (the SPLIT mid), always non-zero for a
 * real arena, and strictly decreasing across allocations, so it is a unique key
 * that never collides even after a slot is reused. */
typedef struct {
  void *rev;
  unsigned long base;
  unsigned long size;
} rof_slot;

#ifndef ROF_MAX_SLOTS
#define ROF_MAX_SLOTS 16384u
#endif

static rof_slot rof_slots[ROF_MAX_SLOTS] __attribute__((aligned(16)));
static unsigned rof_nslots;
static void *rof_arena; /* LIN remainder, parked in .bss (stc/ldc keep it LIN) */

/* When set, rof_free recycles the slot but does NOT fire the revoke. This exists
 * only to build the no-revoke control: the identical program, allocator and free
 * path, minus the one REVOKE, so a post-free read succeeds and the domain
 * returns -- disambiguating the -O0 cause-24 fault from a plain spill reload. */
static int rof_no_revoke;

/* Bytes carved from the arena and bytes handed back to callers -- Phase-2
 * generality instrumentation. rof_arena_bytes only grows (no coalesce). */
static unsigned long rof_carved_total;
static unsigned long rof_live_bytes;
static unsigned rof_live_count;

static inline void rof_init(void *grant) {
  rof_arena = grant;
  rof_nslots = 0;
  rof_carved_total = 0;
  rof_live_bytes = 0;
  rof_live_count = 0;
}

/* Capstone caps want 16-byte alignment for storing caps; keep every allocation
 * 16-aligned so an allocation that later holds a capability is storable. */
static inline unsigned long rof_roundup(unsigned long n) {
  return (n + 15UL) & ~15UL;
}

static inline void *rof_malloc(unsigned long n) {
  if (n == 0)
    return (void *)0;
  n = rof_roundup(n);

  void *cur = rof_arena; /* LIN */
  unsigned long base = __builtin_capstone_cap_get_base(cur);
  unsigned long end = __builtin_capstone_cap_get_end(cur);
  /* cssplit asserts base < mid < end, so the arena must keep a non-empty head. */
  if (n == 0 || end <= base || end - base <= n)
    return (void *)0; /* depleted */

  void *hi = rof_split(&cur, end - n); /* hi = [end-n, end), fresh node, LIN */
  rof_arena = cur;                     /* remainder [base, end-n) */

  void *rev = __builtin_capstone_cap_mrev(hi);   /* while LIN, senior to hi */
  void *alias = __builtin_capstone_cap_delin(hi); /* NONLIN, copyable */

  unsigned i;
  for (i = 0; i < rof_nslots; ++i)
    if (rof_slots[i].base == 0)
      break;
  if (i == rof_nslots) {
    if (rof_nslots >= ROF_MAX_SLOTS)
      return (void *)0;
    i = rof_nslots++;
  }
  rof_slots[i].rev = rev;
  rof_slots[i].base = end - n;
  rof_slots[i].size = n;

  rof_carved_total += n;
  rof_live_bytes += n;
  ++rof_live_count;
  return alias;
}

static inline rof_slot *rof_find(void *p) {
  unsigned long b = __builtin_capstone_cap_get_base(p);
  for (unsigned i = 0; i < rof_nslots; ++i)
    if (rof_slots[i].base == b)
      return &rof_slots[i];
  return (rof_slot *)0;
}

/* The lifecycle point. Every alias derived from this allocation -- including a
 * copy the caller cached across finalize -- stops dereferencing after this. The
 * arena is NOT reclaimed (one-way SPLIT); the slot is marked reusable only. */
static inline void rof_free(void *p) {
  if (!p)
    return;
  rof_slot *s = rof_find(p);
  if (!s)
    return;
  if (!rof_no_revoke)
    __builtin_capstone_cap_revoke(s->rev);
  rof_live_bytes -= s->size;
  --rof_live_count;
  s->base = 0;
}

static inline unsigned long rof_size(void *p) {
  if (!p)
    return 0;
  rof_slot *s = rof_find(p);
  return s ? s->size : 0;
}

/* Tag-PRESERVING copy of `n` bytes. Both allocations are 16-byte aligned and
 * 16-multiple sized (rof_roundup), so the whole range copies as capability-wide
 * words: `*(void**)` uses ldc/stc, which move a capability WITH its tag (and a
 * scalar as a scalar -- ldc dispatches on the memory tag). A byte loop would
 * strip the tag off any capability the block holds, so a pointer SQLite stored
 * in a realloc'd structure would come back untagged and fault on the next deref.
 * This mirrors the capability-preserving memcpy in the domain's libc shim. */
static inline void rof_copy_caps(void *dst, const void *src, unsigned long n) {
  void **d = (void **)dst;
  void *const *s = (void *const *)src;
  unsigned long words = n / sizeof(void *);
  for (unsigned long i = 0; i < words; ++i)
    d[i] = s[i];
}

#endif
