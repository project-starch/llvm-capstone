/* Capstone: one Sublet region per node, and authority that follows the
 * free-list transitions apr_pools.c already has.
 *
 * A node is APR's unit of storage -- MIN_ALLOC or more, a multiple of
 * BOUNDARY_SIZE, carved once from the payload region and never returned to
 * it, because under APR_ALLOCATOR_MAX_FREE_UNLIMITED upstream never returns
 * one either. Upstream still decides everything about nodes: which bucket a
 * released node joins, which node the next request pops. This file decides
 * only who may still address it.
 *
 * mode 0 (spatial): each node keeps the alias it was carved with. A handle
 * saved across apr_pool_destroy still names the storage, which by then is the
 * next pool's -- the defect as upstream ships it.
 * mode 1 (sublet): release and issue each revoke the node's subtree and mint a
 * fresh alias, so that handle is dead. Revocation clears the region, and the
 * node header upstream needs across a transition -- index and endp -- is put
 * back from the adapter's record; next and first_avail upstream rewrites
 * itself after the call, as it always did. */
#include "port.h"
#include "apr_shim.h"
#include "apr_allocator.h"
#include <string.h>
#include <sublet/sublet.h>

#define PAGE 4096UL
#define MIN_NODE (2 * PAGE)
#define RECORDS (APRP_PAYLOAD_BYTES / MIN_NODE)
#define PAGES (APRP_PAYLOAD_BYTES / PAGE)

struct record {
  capstone_cap_slot region;
  void *alias;
  size_t size;
  uint32_t index;
  unsigned live, released_once, discarded;
};
static capstone_cap_slot remaining;
static uintptr_t base, cursor, end;
static struct record *records;
static uint16_t *page_map; /* record number + 1 for every page of a node */
static unsigned count, protected_mode;
static uint64_t reuses, releases, discards;

void aprp_payload_init(void *region) {
  capstone_cap_store(&remaining, region);
  base = cursor = capstone_cap_base(&remaining);
  end = base + APRP_PAYLOAD_BYTES;
  if (capstone_cap_type(&remaining) != CAPSTONE_CAP_LINEAR ||
      capstone_cap_end(&remaining) != end || (base & (PAGE - 1)))
    aprp_fail(501);
}
void aprp_set_mode(unsigned mode) {
  if (mode > 1)
    aprp_fail(502);
  protected_mode = mode;
  records = aprp_meta_calloc(RECORDS, sizeof *records);
  page_map = aprp_meta_calloc(PAGES, sizeof *page_map);
  if (!records || !page_map)
    aprp_fail(503);
}

static struct record *record_for(const void *node) {
  uintptr_t address = (uintptr_t)node;
  if (address < base || address >= end || (address & (PAGE - 1)))
    aprp_fail(515);
  unsigned slot = page_map[(address - base) / PAGE];
  if (!slot)
    aprp_fail(516);
  struct record *r = &records[slot - 1];
  /* Upstream only ever hands the adapter a node's own address. */
  if ((uintptr_t)r->alias != address)
    aprp_fail(517);
  return r;
}
/* Revocation cleared the header; give upstream back the two fields it reads
 * across a transition. Written through the alias, so the store is checked. */
static void *restore_header(struct record *r) {
  apr_memnode_t *node = r->alias;
  node->index = r->index;
  node->endp = (char *)r->alias + r->size;
  return r->alias;
}

void *aprp_node_backing(size_t size) {
  if (size < MIN_NODE || (size & (PAGE - 1)))
    aprp_fail(504);
  if (count == RECORDS || cursor > end - size)
    return NULL; /* upstream turns this into APR_ENOMEM */
  struct record *r = &records[count];
  r->size = size;
  r->index = (uint32_t)((size / PAGE) - 1);
  sublet_carve(&remaining, cursor + size, &r->region);
  r->alias = sublet_take(&r->region);
  for (uintptr_t page = cursor; page < cursor + size; page += PAGE)
    page_map[(page - base) / PAGE] = (uint16_t)(count + 1);
  cursor += size;
  ++count;
  return r->alias;
}
void *aprp_node_issue(void *node) {
  struct record *r = record_for(node);
  if (r->live || r->discarded)
    aprp_fail(518);
  if (r->released_once)
    ++reuses;
  r->live = 1;
  if (!protected_mode)
    return r->alias;
  sublet_give(&r->region);
  r->alias = sublet_take(&r->region);
  return restore_header(r);
}
void *aprp_node_release(void *node) {
  struct record *r = record_for(node);
  if (!r->live || r->discarded)
    aprp_fail(519);
  r->live = 0;
  r->released_once = 1;
  ++releases;
  if (!protected_mode)
    return r->alias;
  sublet_give(&r->region);
  r->alias = sublet_take(&r->region);
  return restore_header(r);
}
void aprp_node_discard(void *node) {
  struct record *r = record_for(node);
  if (r->live || r->discarded)
    aprp_fail(520);
  r->discarded = 1;
  ++discards;
  /* Only apr_allocator_destroy comes here, at the very end. The storage is
   * never handed out again, so reclaiming the authority is all there is. */
  if (protected_mode)
    sublet_give(&r->region);
}
void aprp_stats(struct aprp_header *out) {
  out->nodes = count;
  out->node_reuses = reuses;
  out->node_releases = releases;
  out->node_discards = discards;
  out->backing_used = cursor - base;
  out->metadata = aprp_meta_used();
}
