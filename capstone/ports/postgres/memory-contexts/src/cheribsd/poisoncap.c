/* Hosted implementation of the existing PostgreSQL manager hook ABI.
 * Metadata stays outside revocable storage. Manager pointers retain poison
 * authority; published chunk pointers lose it. This is a trusted, serial
 * adapter, with synchronous sweeps before storage can be reused. */
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
static unsigned char headers[PG_SUBPOOL_MAX][PG_HEADER_BYTES]
    __attribute__((aligned(16)));
static unsigned char header_live[PG_SUBPOOL_MAX];
static unsigned mode, initialized;
static size_t tolerated;
static size_t sweeps, poisoned, cleared, zeroed, mapped, padding;

static _Noreturn void refuse(const char *why) {
  fprintf(stderr, "PG_POISONCAP refused: %s\n", why);
  exit(1);
}

void pg_poisoncap_init(unsigned selected) {
  if (initialized || selected > 1 || !feature_present("cheri_caprevoke_poison"))
    refuse("initialization or platform");
  initialized = 1;
  mode = selected;
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
  if (!b) return NULL;
  b->base = cheri_getaddress(b->region);
  b->freeptr = b->base;
  b->endptr = b->base + bytes;
  used[b - blocks] = 0;
  b->pool = sp;
  b->pool_next = sp->blocks;
  b->pool_prev = NULL;
  if (sp->blocks) sp->blocks->pool_prev = b;
  sp->blocks = b;
  ++pg_subpool_counts.blocks;
  return b;
}

static void release_entries(pg_block *b) {
  unsigned i = b->chunk_head;
  while (i) {
    unsigned next = chunks[i].block_next;
    memset(&chunks[i], 0, sizeof chunks[i]);
    active[i] = 0;
    i = next;
  }
  b->chunk_head = b->chunk_tail = b->chunks = 0;
}

void pg_subpool_block_free(pg_block *b) {
  if (!b || !b->pool) refuse("double block release");
  invalidate(b->region, used[b - blocks]);
  release_entries(b);
  if (b->pool_prev) b->pool_prev->pool_next = b->pool_next;
  else b->pool->blocks = b->pool_next;
  if (b->pool_next) b->pool_next->pool_prev = b->pool_prev;
  b->pool = NULL;
  b->pool_next = b->pool_prev = b->prev = b->next = NULL;
  b->aset = NULL;
  ++pg_subpool_counts.blocks_freed;
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
  invalidate(b->region, used[b - blocks]);
  release_entries(b);
  b->freeptr = b->base + prefix;
  used[b - blocks] = prefix;
}
void pg_subpool_managed_free(pg_block *b) { pg_subpool_block_free(b); }

unsigned int pg_subpool_carve(pg_block *b, unsigned long bytes) {
  bytes = (bytes + 15) & ~15UL;
  if (!bytes || bytes > b->endptr - b->freeptr) return 0;
  unsigned i;
  for (i = 1; i < PG_CHUNK_MAX && chunks[i].slot; ++i) {}
  if (i == PG_CHUNK_MAX) return 0;
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
  b->freeptr += bytes;
  ++pg_subpool_counts.carves;
  return i;
}
void *pg_subpool_hand(unsigned int i) {
  if (!i || i >= PG_CHUNK_MAX || !chunks[i].slot || active[i])
    refuse("invalid chunk handout");
  active[i] = 1;
  ++pg_subpool_counts.hands;
  return cheri_clearperm(chunks[i].slot,
                         CHERI_PERM_POISON | CHERI_PERM_SW_VMEM);
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
  invalidate(chunks[i].slot, cheri_getlen(chunks[i].slot));
  active[i] = 0;
  ++pg_subpool_counts.drops;
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
  printf("PG_POISONCAP mode=%u sweeps=%zu poison_bytes=%zu clear_bytes=%zu "
         "zeroed_bytes=%zu hands=%lu drops=%lu tolerated_double_drops=%zu "
         "mapped_bytes=%zu padding_bytes=%zu pointer_bytes=%zu\n",
         mode, sweeps, poisoned, cleared, zeroed,
         pg_subpool_counts.hands, pg_subpool_counts.drops, tolerated, mapped,
         padding, sizeof(void *));
}
