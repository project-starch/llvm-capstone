/* Hosted implementation of the existing PostgreSQL manager hook ABI.
 * Metadata stays outside revocable storage. Manager pointers retain poison
 * authority; published chunk pointers lose it. This is a trusted, serial
 * adapter. Batched builds transfer the published SQLite thresholds; quarantined
 * storage cannot be reused before a successful sweep. */
#include "pg_subpool.h"
#include "poisoncap.h"
#include <cheri/cheric.h>
#include <cheri/revoke.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

#ifdef PG_REUSE_GAP_OBSERVER
#include "../../../../../experiments/study/reuse-gap-observer.h"
#define PG_REUSE_SLOTS (1u << 17)
static struct reuse_gap_slot pg_reuse_slots[PG_REUSE_SLOTS];
static struct reuse_gap_observer pg_reuse;
#endif

#ifndef CHERI_PERM_POISON
#error "PG_POISONCAP requires the PoisonCap SDK"
#endif

struct pg_subpool { pg_block *blocks; unsigned live; };
struct pg_subpool_counts pg_subpool_counts;
static pg_subpool pools[PG_SUBPOOL_MAX];
static pg_block blocks[PG_BLOCK_MAX];
/* PostgreSQL's cursors account logical bytes. CHERI also needs padding for
 * representable chunk bounds; track the physical cursor independently so the
 * four managers retain their own block and size-class decisions. */
static size_t used[PG_BLOCK_MAX];
static pg_chunk chunks[PG_CHUNK_MAX];
static unsigned char active[PG_CHUNK_MAX];
static size_t chunk_entries_live, chunk_entries_peak, live_chunk_bytes;
static unsigned free_ids[PG_CHUNK_MAX], free_count, next_id = 1;
/* A block header reservation or an unissued carve has no client aliases. */
static unsigned char issued_in_block[PG_BLOCK_MAX];
#ifdef PG_POISONCAP_BATCHED
/* Published MEMSYS5 thresholds, with revoke-before-full-queue-drain repair.
 * No allocation/reuse-triggered drain. Links remain in the external table. */
#define PG_QUARANTINE_CAPACITY 4096u
#define PG_QUARANTINE_MIN_HELD (16u << 20)
static size_t capacity_sweeps, threshold_sweeps;
static unsigned char pending[PG_CHUNK_MAX];
static unsigned pending_ids[PG_CHUNK_MAX], pending_count, pending_peak;
static size_t withheld_bytes, withheld_peak;
/* Whole blocks retain their chunk entries until a sweep, so their poisoned
 * contents can be reclaimed together with the chunk queue. */
static unsigned char pending_blocks[PG_BLOCK_MAX];
static unsigned pending_block_ids[PG_BLOCK_MAX], pending_block_count;
static unsigned pending_block_peak;
static size_t withheld_block_bytes, withheld_block_peak;
#endif
static unsigned char headers[PG_SUBPOOL_MAX][PG_HEADER_BYTES]
    __attribute__((aligned(16)));
static unsigned char header_live[PG_SUBPOOL_MAX];
static unsigned mode, initialized;
static size_t tolerated;
static size_t sweeps, poisoned, cleared, zeroed, mapped, padding;
static size_t managed_reset_sweeps;

static _Noreturn void refuse(const char *why) {
  fprintf(stderr, "PG_POISONCAP refused: %s\n", why);
  _Exit(1); /* PostgreSQL exit callbacks may allocate from the exhausted table. */
}

#ifdef PG_POISONCAP_BATCHED
static void poison_only(void *ptr, size_t n) {
  if (!mode || !n) return;
  if ((cheri_getaddress(ptr) & 15) || (n & 15) ||
      (cheri_getperm(ptr) & (CHERI_PERM_POISON | CHERI_PERM_SW_VMEM)) !=
          (CHERI_PERM_POISON | CHERI_PERM_SW_VMEM))
    refuse("poison geometry or authority");
  for (size_t i = 0; i < n; i += 16) {
    void *p = (unsigned char *)ptr + i;
    __asm__ volatile("cpoison %0, 0(%0)" : : "C"(p) : "memory");
  }
  poisoned += n;
}

static void release_entries(pg_block *b);

static void flush_pending(void) {
  if (!pending_count && !pending_block_count) return;
  struct cheri_revoke_syscall_info info = {0};
  if (cheri_revoke(CHERI_REVOKE_LAST_PASS | CHERI_REVOKE_IGNORE_START |
                   CHERI_REVOKE_TAKE_STATS, 0, &info))
    refuse("quarantine sweep failed");
  ++sweeps;
  for (unsigned q = 0; q < pending_count; ++q) {
    unsigned i = pending_ids[q];
    /* A block-level clear below already covers any queued chunk it owns. */
    if (!pending_blocks[chunks[i].block]) {
      void *ptr = chunks[i].slot;
      size_t n = cheri_getlen(ptr);
      for (size_t off = 0; off < n; off += 16) {
        void *p = (unsigned char *)ptr + off;
        __asm__ volatile("cclearpoison %0, 0(%0)" : : "C"(p) : "memory");
      }
      memset(ptr, 0, n);
      cleared += n;
      zeroed += n;
    }
    pending[i] = 0;
  }
  for (unsigned q = 0; q < pending_block_count; ++q) {
    unsigned i = pending_block_ids[q];
    pg_block *b = &blocks[i];
    size_t n = used[i];
    for (size_t off = 0; off < n; off += 16) {
      void *p = (unsigned char *)b->region + off;
      __asm__ volatile("cclearpoison %0, 0(%0)" : : "C"(p) : "memory");
    }
    memset(b->region, 0, n);
    cleared += n;
    zeroed += n;
    release_entries(b);
    pending_blocks[i] = 0;
  }
  pending_count = 0;
  withheld_bytes = 0;
  pending_block_count = 0;
  withheld_block_bytes = 0;
}

static void before_enqueue(void) {
  if (pending_count + pending_block_count == PG_QUARANTINE_CAPACITY) {
    ++capacity_sweeps;
    flush_pending();
  }
}

static void after_enqueue(void) {
  size_t quarantined = withheld_bytes + withheld_block_bytes;
  size_t held = live_chunk_bytes + quarantined;
  if (held >= PG_QUARANTINE_MIN_HELD && quarantined >= held / 4) {
    ++threshold_sweeps;
    flush_pending();
  }
}

int pg_poisoncap_defer_reuse(unsigned i) {
  if (!i || i >= PG_CHUNK_MAX) refuse("invalid reuse index");
  return mode && pending[i];
}
#endif

void pg_poisoncap_init(unsigned selected) {
  if (initialized || selected > 1 || !feature_present("cheri_caprevoke_poison"))
    refuse("initialization or platform");
  initialized = 1;
  mode = selected;
#ifdef PG_REUSE_GAP_OBSERVER
  reuse_gap_init(&pg_reuse, pg_reuse_slots, PG_REUSE_SLOTS);
#endif
}

static void invalidate(void *ptr, size_t n) {
  if (!mode || !n)
    return;
  if ((cheri_getaddress(ptr) & 15) || (n & 15) ||
      (cheri_getperm(ptr) & (CHERI_PERM_POISON | CHERI_PERM_SW_VMEM)) !=
          (CHERI_PERM_POISON | CHERI_PERM_SW_VMEM))
    refuse("poison geometry or authority");
  for (size_t i = 0; i < n; i += 16) {
    void *p = (unsigned char *)ptr + i;
    __asm__ volatile("cpoison %0, 0(%0)" : : "C"(p) : "memory");
  }
  struct cheri_revoke_syscall_info info = {0};
  if (cheri_revoke(CHERI_REVOKE_LAST_PASS | CHERI_REVOKE_IGNORE_START |
                   CHERI_REVOKE_TAKE_STATS, 0, &info))
    refuse("sweep failed; storage cannot be reused");
  ++sweeps;
  poisoned += n;
  for (size_t i = 0; i < n; i += 16) {
    void *p = (unsigned char *)ptr + i;
    __asm__ volatile("cclearpoison %0, 0(%0)" : : "C"(p) : "memory");
  }
  cleared += n;
  /* Clearing access state does not erase the poison capability in memory.
   * Erase it before a new lease can be published or a later sweep can see it. */
  memset(ptr, 0, n);
  zeroed += n;
}

pg_subpool *pg_subpool_create(void) {
  if (!initialized)
    pg_poisoncap_init(1); /* Direct-link clients default to protected mode. */
  for (unsigned i = 0; i < PG_SUBPOOL_MAX; ++i)
    if (!pools[i].live) {
      pools[i].live = 1;
      ++pg_subpool_counts.created;
      return &pools[i];
    }
  return NULL;
}

/* Backing blocks are retained until process exit, including in mode 0.
 * Therefore libc free/revocation cannot explain a protected/unprotected pair.
 * A released block is reused first-fit; pool reset does not return it to libc. */
int pg_subpool_grow(pg_subpool *sp, unsigned long bytes) {
  (void)sp; (void)bytes;
  return 0; /* pg_subpool_block already tries the whole backing table. */
}

pg_block *pg_subpool_block(pg_subpool *sp, unsigned long bytes) {
  if (!sp || !sp->live || !bytes || bytes > (64UL << 20))
    return NULL;
  bytes = (bytes + 15) & ~15UL;
  /* Small chunks are exact at 16-byte alignment. For larger chunks, rounding
   * plus alignment consumes less than twice their logical size on this SDK.
   * The carve checks this bound instead of widening into adjacent storage. */
  size_t reserve = (2 * bytes + 8191) & ~4095UL;
  pg_block *b = NULL;
  for (unsigned i = 0; i < PG_BLOCK_MAX; ++i)
    if (!blocks[i].pool && blocks[i].region &&
#ifdef PG_POISONCAP_BATCHED
        !pending_blocks[i] &&
#endif
        cheri_getlen(blocks[i].region) >= reserve) {
      b = &blocks[i];
      break;
    }
  if (!b)
    for (unsigned i = 0; i < PG_BLOCK_MAX; ++i)
      if (!blocks[i].region) {
        b = &blocks[i];
        void *region = mmap(NULL, reserve, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANON, -1, 0);
        if (region == MAP_FAILED)
          return NULL;
        /* libc allocation removes SW_VMEM. Without it the sweep revokes the
         * manager too, and cclearpoison faults on its own dead capability. */
        if ((cheri_getperm(region) & (CHERI_PERM_POISON | CHERI_PERM_SW_VMEM)) !=
            (CHERI_PERM_POISON | CHERI_PERM_SW_VMEM))
          refuse("backing lacks manager permissions");
        b->region = region;
        mapped += reserve;
        break;
      }
  if (!b) refuse("block metadata capacity exhausted");
  b->base = cheri_getaddress(b->region);
  b->freeptr = b->base;
  b->endptr = b->base + bytes;
  used[b - blocks] = 0;
  issued_in_block[b - blocks] = 0;
  b->pool = sp;
  b->pool_next = sp->blocks;
  b->pool_prev = NULL;
  if (sp->blocks) sp->blocks->pool_prev = b;
  sp->blocks = b;
  ++pg_subpool_counts.blocks;
  return b;
}

static void retire_chunk(unsigned i) {
  if (!active[i]) return;
#ifdef PG_REUSE_GAP_OBSERVER
  reuse_gap_release(&pg_reuse, (uint64_t)cheri_getaddress(chunks[i].slot));
#endif
  size_t n = cheri_getlen(chunks[i].slot);
  if (live_chunk_bytes < n) refuse("live chunk accounting underflow");
  live_chunk_bytes -= n;
  active[i] = 0;
}

static void release_entries(pg_block *b) {
  unsigned i = b->chunk_head;
  while (i) {
    unsigned next = chunks[i].block_next;
    retire_chunk(i);
    if (!chunk_entries_live) refuse("chunk metadata accounting underflow");
    --chunk_entries_live;
    memset(&chunks[i], 0, sizeof chunks[i]);
    active[i] = 0;
    if (free_count >= PG_CHUNK_MAX) refuse("chunk index free list overflow");
    free_ids[free_count++] = i;
    i = next;
  }
  b->chunk_head = b->chunk_tail = b->chunks = 0;
  issued_in_block[b - blocks] = 0;
}

void pg_subpool_block_free(pg_block *b) {
  if (!b || !b->pool) refuse("double block release");
#ifdef PG_POISONCAP_BATCHED
  if (mode && issued_in_block[b - blocks]) {
    unsigned i = b - blocks;
    before_enqueue();
    if (pending_block_count >= PG_BLOCK_MAX)
      refuse("block quarantine capacity exhausted");
    /* Replace individual quarantined chunks by their containing block span,
     * so held/quarantined bytes never count overlapping storage twice. */
    unsigned keep = 0;
    for (unsigned q = 0; q < pending_count; ++q) {
      unsigned c = pending_ids[q];
      if (chunks[c].block == i) {
        withheld_bytes -= cheri_getlen(chunks[c].slot);
        pending[c] = 0;
      } else pending_ids[keep++] = c;
    }
    pending_count = keep;
    for (unsigned c = b->chunk_head; c; c = chunks[c].block_next)
      retire_chunk(c);
    poison_only(b->region, used[i]);
    pending_blocks[i] = 1;
    pending_block_ids[pending_block_count++] = i;
    if (pending_block_count > pending_block_peak)
      pending_block_peak = pending_block_count;
    withheld_block_bytes += used[i];
    if (withheld_block_bytes > withheld_block_peak)
      withheld_block_peak = withheld_block_bytes;
  } else {
    release_entries(b);
  }
#else
  invalidate(b->region, used[b - blocks]);
  release_entries(b);
#endif
  if (b->pool_prev) b->pool_prev->pool_next = b->pool_next;
  else b->pool->blocks = b->pool_next;
  if (b->pool_next) b->pool_next->pool_prev = b->pool_prev;
  b->pool = NULL;
  b->pool_next = b->pool_prev = b->prev = b->next = NULL;
  b->aset = NULL;
  ++pg_subpool_counts.blocks_freed;
#ifdef PG_POISONCAP_BATCHED
  if (mode) after_enqueue();
#endif
}

void pg_subpool_reset(pg_subpool *sp) {
  ++pg_subpool_counts.resets;
  while (sp->blocks) pg_subpool_block_free(sp->blocks);
}
void pg_subpool_destroy(pg_subpool *sp) {
  pg_subpool_reset(sp);
  sp->live = 0;
  ++pg_subpool_counts.destroyed;
}
pg_block *pg_subpool_managed_block(pg_subpool *sp, unsigned long bytes,
                                 unsigned long prefix) {
  if (prefix > bytes) return NULL;
  pg_block *b = pg_subpool_block(sp, bytes);
  if (b) {
    b->freeptr += prefix;
    used[b - blocks] = prefix;
  }
  return b;
}
void pg_subpool_managed_reset(pg_block *b, unsigned long prefix) {
  if (prefix > b->endptr - b->base) refuse("managed prefix");
  if (mode && issued_in_block[b - blocks]) {
#ifdef PG_POISONCAP_BATCHED
    flush_pending();
#endif
    ++managed_reset_sweeps;
    invalidate(b->region, used[b - blocks]);
  }
  release_entries(b);
  b->freeptr = b->base + prefix;
  used[b - blocks] = prefix;
}
void pg_subpool_managed_free(pg_block *b) { pg_subpool_block_free(b); }

unsigned int pg_subpool_carve(pg_block *b, unsigned long bytes) {
  bytes = (bytes + 15) & ~15UL;
  if (!bytes || bytes > b->endptr - b->freeptr) return 0;
  unsigned i;
  if (free_count) i = free_ids[--free_count];
  else if (next_id < PG_CHUNK_MAX) i = next_id++;
  else refuse("chunk metadata capacity exhausted");
  size_t n = CHERI_REPRESENTABLE_LENGTH(bytes);
  size_t mask = CHERI_REPRESENTABLE_ALIGNMENT_MASK(n);
  size_t start = (b->base + used[b - blocks] + ~mask) & mask;
  size_t offset = start - b->base;
  if (offset > cheri_getlen(b->region) || n > cheri_getlen(b->region) - offset)
    refuse("representability padding exhausted backing");
  void *p = (unsigned char *)b->region + offset;
  void *bounded = cheri_setboundsexact(p, n);
  if (!cheri_gettag(bounded)) refuse("unrepresentable chunk bounds");
  padding += offset + n - used[b - blocks] - bytes;
  used[b - blocks] = offset + n;
  chunks[i].slot = bounded;
  chunks[i].bytes = bytes;
  chunks[i].block = b - blocks;
  if (b->chunk_tail) chunks[b->chunk_tail].block_next = i;
  else b->chunk_head = i;
  b->chunk_tail = i;
  ++b->chunks;
  if (++chunk_entries_live > chunk_entries_peak)
    chunk_entries_peak = chunk_entries_live;
  b->freeptr += bytes;
  ++pg_subpool_counts.carves;
  return i;
}
void *pg_subpool_hand(unsigned int i) {
  if (!i || i >= PG_CHUNK_MAX || !chunks[i].slot || active[i])
    refuse("invalid chunk handout");
#ifdef PG_POISONCAP_BATCHED
  if (pending[i]) refuse("attempt to reissue quarantined chunk");
#endif
  active[i] = 1;
  live_chunk_bytes += cheri_getlen(chunks[i].slot);
  issued_in_block[chunks[i].block] = 1;
  ++pg_subpool_counts.hands;
  void *client = cheri_clearperm(chunks[i].slot,
                                CHERI_PERM_POISON | CHERI_PERM_SW_VMEM);
#ifdef PG_REUSE_GAP_OBSERVER
  reuse_gap_attempt(&pg_reuse);
  reuse_gap_issue(&pg_reuse, (uint64_t)cheri_getaddress(client),
                  (uint64_t)chunks[i].bytes);
#endif
  return client;
}
void pg_subpool_drop(unsigned int i) {
  if (!i || i >= PG_CHUNK_MAX) refuse("invalid chunk release");
  if (!active[i]) {
    /* An unprotected arm has to be unprotected. This active[] table is the
     * ADAPTER's bookkeeping, not the manager's, so in mode 0 it would stop a
     * double free before the allocator ever sees it -- and a case whose defect
     * IS a double free would then have its control arm refuse, leaving nothing
     * to pair the protected arm against. In mode 1 the second release is still
     * a refusal, because there the storage is genuinely gone.
     *
     * Counted rather than silent: a mode-0 run that tolerated one says so in
     * its report, so this can never be mistaken for a clean run. */
    if (mode)
      refuse("double chunk release");
    ++tolerated;
    return;
  }
#ifdef PG_POISONCAP_BATCHED
  if (mode) {
    before_enqueue();
    size_t bytes = cheri_getlen(chunks[i].slot);
    if (pending_count >= PG_CHUNK_MAX) refuse("quarantine index capacity");
    poison_only(chunks[i].slot, bytes);
    pending[i] = 1;
    pending_ids[pending_count++] = i;
    if (pending_count > pending_peak) pending_peak = pending_count;
    withheld_bytes += bytes;
    if (withheld_bytes > withheld_peak) withheld_peak = withheld_bytes;
  }
#else
  invalidate(chunks[i].slot, cheri_getlen(chunks[i].slot));
#endif
  retire_chunk(i);
  ++pg_subpool_counts.drops;
#ifdef PG_POISONCAP_BATCHED
  if (mode) after_enqueue();
#endif
}
pg_chunk *pg_subpool_entry(unsigned int i) { return &chunks[i]; }
pg_block *pg_subpool_block_at(unsigned int i) { return &blocks[i]; }
unsigned int pg_subpool_block_index(pg_block *b) { return b - blocks; }
unsigned int pg_subpool_first_chunk(pg_block *b) { return b->chunk_head; }
unsigned int pg_subpool_next_chunk(unsigned int i) { return chunks[i].block_next; }
void pg_subpool_count_keeper(void) { ++pg_subpool_counts.keepers; }
void *pg_subpool_header(unsigned long bytes) {
  if (bytes > PG_HEADER_BYTES) return NULL;
  for (unsigned i = 0; i < PG_SUBPOOL_MAX; ++i)
    if (!header_live[i]) {
      header_live[i] = 1;
      memset(headers[i], 0, PG_HEADER_BYTES);
      return headers[i];
    }
  return NULL;
}
void pg_subpool_header_free(void *p) {
  for (unsigned i = 0; i < PG_SUBPOOL_MAX; ++i)
    if (p == headers[i] && header_live[i]) {
      header_live[i] = 0;
      return;
    }
  refuse("invalid context header");
}
void pg_poisoncap_report(void) {
  size_t table_bytes = sizeof chunks + sizeof active + sizeof free_ids;
#ifdef PG_POISONCAP_BATCHED
  table_bytes += sizeof pending + sizeof pending_ids;
  printf("PG_POISONCAP_POLICY queue_capacity=%u min_held=%u fraction_denominator=4 "
         "capacity_sweeps=%zu threshold_sweeps=%zu live_chunk_bytes=%zu\n",
         PG_QUARANTINE_CAPACITY, PG_QUARANTINE_MIN_HELD,
         capacity_sweeps, threshold_sweeps, live_chunk_bytes);
#endif
  printf("PG_POISONCAP_METADATA chunk_capacity=%u chunk_live=%zu chunk_peak=%zu "
         "chunk_table_bytes=%zu block_capacity=%u\n", PG_CHUNK_MAX,
         chunk_entries_live, chunk_entries_peak, table_bytes, PG_BLOCK_MAX);
#ifdef PG_REUSE_GAP_OBSERVER
  printf("PG_REUSE_GAP attempts=%llu issues=%llu releases=%llu reuses=%llu "
         "distinct=%llu capacity=%u error=%u bins=",
         (unsigned long long)pg_reuse.attempts,
         (unsigned long long)pg_reuse.issues,
         (unsigned long long)pg_reuse.releases,
         (unsigned long long)pg_reuse.reuses,
         (unsigned long long)pg_reuse.distinct_starts,
         PG_REUSE_SLOTS, pg_reuse.error);
  for (unsigned i = 0; i < 32; ++i)
    printf("%s%llu", i ? "," : "", (unsigned long long)pg_reuse.bins[i]);
  printf("\n");
#endif

#ifdef PG_POISONCAP_BATCHED
  /* Do not drain the queue for reporting: its retained bytes are a result. */
  printf("PG_POISONCAP_QUEUE pending=%u peak_pending=%u withheld=%zu peak_withheld=%zu "
         "blocks=%u peak_blocks=%u block_bytes=%zu peak_block_bytes=%zu\n",
         pending_count, pending_peak, withheld_bytes, withheld_peak,
         pending_block_count, pending_block_peak, withheld_block_bytes,
         withheld_block_peak);
#endif
  printf("PG_POISONCAP mode=%u sweeps=%zu poison_bytes=%zu clear_bytes=%zu "
         "zeroed_bytes=%zu hands=%lu drops=%lu tolerated_double_drops=%zu "
         "mapped_bytes=%zu padding_bytes=%zu pointer_bytes=%zu "
         "managed_reset_sweeps=%zu\n",
         mode, sweeps, poisoned, cleared, zeroed,
         pg_subpool_counts.hands, pg_subpool_counts.drops, tolerated, mapped,
         padding, sizeof(void *), managed_reset_sweeps);
}
