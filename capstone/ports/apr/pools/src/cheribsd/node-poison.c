/* CheriBSD with PoisonCap: every node is one mapped region that keeps SW_VMEM
 * and POISON authority for the manager; what APR is handed is an exactly
 * bounded alias without either, so a sweep revokes it and nothing else.
 *
 * Mode 0 publishes and invalidates nothing: the allocator as upstream ships
 * it, under exact bounds. Mode 1 invalidates a node at release -- poison
 * every granule, one synchronous sweep, clear, zero -- and at discard, so the
 * alias a pool handed out is dead by the time APR reissues the node. That is
 * the same transition the Sublet adapter revokes at; the bucket adapter's
 * pieces live inside the node and die with it. The memnode header fields
 * upstream reads across a transition, index and endp, are put back from the
 * record, as the Sublet adapter does. Trusted, serial, no policy of its own. */
#include "port.h"
#include "poison.h"
#include "apr_shim.h"
#include "apr_allocator.h"
#include <capstone/capability-slot.h>
#include <cheri/cheric.h>
#include <cheri/revoke.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef CHERI_PERM_POISON
#error "APRP_POISONCAP requires the PoisonCap SDK"
#endif

#define PAGE 4096UL
#define RECORDS (APRP_PAYLOAD_BYTES / (2 * PAGE))

struct record {
  void *region; /* the manager's pointer: full bounds, poison authority */
  void *alias;  /* what APR holds */
  size_t size;
  uint32_t index;
  unsigned live, released_once, discarded, lent;
};
static struct record *records;
static unsigned count, mode;
static uint64_t reuses, releases, discards;
static size_t backing, sweeps, poisoned, node_invalidations, piece_invalidations;

static _Noreturn void refuse(const char *why) {
  fprintf(stderr, "APRP_POISONCAP refused: %s\n", why);
  exit(1);
}
void aprp_payload_init(void *region) { (void)region; /* regions are mapped here */ }
void aprp_set_mode(unsigned selected) {
  if (selected > 1 || !feature_present("cheri_caprevoke_poison"))
    refuse("mode or platform");
  mode = selected;
  records = aprp_meta_calloc(RECORDS, sizeof *records);
  if (!records)
    aprp_fail(503);
}
unsigned aprp_mode(void) { return mode; }

static struct record *record_for(const void *node, unsigned exact) {
  ptraddr_t address = cheri_getaddress(node);
  for (unsigned i = 0; i < count; ++i) {
    struct record *r = &records[i];
    ptraddr_t base = cheri_getaddress(r->region);
    if (exact ? address == base : address >= base && address < base + r->size)
      return r;
  }
  aprp_fail(516);
}
void *aprp_poison_authority(const void *p) { return record_for(p, 0)->region; }
void *aprp_poison_publish(void *manager, size_t n) {
  void *bounded = cheri_setboundsexact(manager, n);
  if (!cheri_gettag(bounded))
    refuse("unrepresentable bounds"); /* nodes and pieces are 16-byte multiples */
  return cheri_clearperm(bounded, CHERI_PERM_POISON | CHERI_PERM_SW_VMEM);
}
void aprp_poison_invalidate(void *ptr, size_t n) {
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
  if (cheri_revoke(CHERI_REVOKE_LAST_PASS | CHERI_REVOKE_IGNORE_START | CHERI_REVOKE_TAKE_STATS, 0, &info))
    refuse("sweep failed; storage cannot be reused");
  ++sweeps;
  poisoned += n;
  for (size_t i = 0; i < n; i += 16) {
    void *p = (unsigned char *)ptr + i;
    __asm__ volatile("cclearpoison %0, 0(%0)" : : "C"(p) : "memory");
  }
  memset(ptr, 0, n); /* clearing access state does not erase the poison capability in memory */
}
/* Invalidation zeroed the header; give upstream back the two fields it reads
 * across a transition, through the fresh alias so the store is checked. */
static void *republish(struct record *r) {
  r->alias = aprp_poison_publish(r->region, r->size);
  apr_memnode_t *node = r->alias;
  node->index = r->index;
  node->endp = (char *)r->alias + r->size;
  return r->alias;
}
void *aprp_node_backing(size_t size) {
  if (size < 2 * PAGE || (size & (PAGE - 1)) || count == RECORDS)
    return NULL;
  void *region = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (region == MAP_FAILED)
    return NULL;
  if ((cheri_getperm(region) & (CHERI_PERM_POISON | CHERI_PERM_SW_VMEM)) !=
      (CHERI_PERM_POISON | CHERI_PERM_SW_VMEM))
    refuse("mapping lacks poison authority");
  struct record *r = &records[count++];
  r->region = region;
  r->size = size;
  r->index = (uint32_t)((size / PAGE) - 1);
  backing += size;
  r->alias = aprp_poison_publish(region, size);
  return r->alias;
}
void *aprp_node_issue(void *node) {
  struct record *r = record_for(node, 1);
  if (r->live || r->discarded)
    aprp_fail(518);
  if (r->released_once)
    ++reuses;
  r->live = 1;
  return r->alias;
}
void *aprp_node_release(void *node) {
  struct record *r = record_for(node, 1);
  if (!r->live || r->discarded)
    aprp_fail(519);
  r->live = 0;
  r->lent = 0;
  r->released_once = 1;
  ++releases;
  if (!mode)
    return r->alias;
  aprp_poison_invalidate(r->region, r->size);
  ++node_invalidations;
  return republish(r);
}
void aprp_node_discard(void *node) {
  struct record *r = record_for(node, 1);
  if (r->live || r->discarded)
    aprp_fail(520);
  r->discarded = 1;
  ++discards;
  if (mode) {
    aprp_poison_invalidate(r->region, r->size);
    ++node_invalidations;
  }
  /* Regions stay mapped until process exit, so libc can never explain a pair. */
}
void *aprp_node_lend(void *node, struct capstone_cap_slot *rest) {
  struct record *r = record_for(node, 1);
  if (!r->live || r->discarded || r->lent)
    aprp_fail(521);
  r->lent = 1;
  rest->c = NULL;
  return r->alias;
}
void aprp_poison_count_piece(void) { ++piece_invalidations; }
void aprp_stats(struct aprp_header *out) {
  out->nodes = count;
  out->node_reuses = reuses;
  out->node_releases = releases;
  out->node_discards = discards;
  out->backing_used = backing;
  out->metadata = aprp_meta_used();
}
void aprp_poison_report(void) {
  printf("APRP_POISONCAP mode=%u sweeps=%zu poison_bytes=%zu node_invalidations=%zu "
         "piece_invalidations=%zu nodes=%u pointer_bytes=%zu\n",
         mode, sweeps, poisoned, node_invalidations, piece_invalidations, count, sizeof(void *));
  fflush(stdout);
}
