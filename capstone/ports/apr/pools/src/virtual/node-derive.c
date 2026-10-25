/* Virtual profile, nested: authority over a node follows the free-list
 * transitions apr_pools.c already has, with the virtual ISA's two Sublet
 * instructions (docs/design/virtual-capstone/isa.md) and no region of our own.
 *
 * A node's memory comes from the platform's malloc, as in the stock adapter
 * (src/cheribsd/node-malloc.c), 4 KiB aligned so every node size APR asks for
 * is exactly representable. Its OWNER is a CDERIVE child of that memory
 * covering the node; it carries MANAGE, and it is the alias APR's free list
 * holds. A pool never sees the owner:
 *   issue    CDERIVE a fresh child of the owner, covering the node; the pool
 *            gets it, and so does everything the pool hands out of the node.
 *   release  CREVOKE(owner, that child): every handle into the node, and every
 *            bucket piece derived below it (src/virtual/bucket-derive.c), dies.
 *            Upstream gets the owner back to write the free-list link through.
 *   discard  free() of the memory, from apr_allocator_destroy only.
 * CREVOKE does not touch payload bytes, so the header upstream reads across a
 * transition (index, endp) is still there; nothing is restored.
 *
 * Mode 1 only: the cases' driver selects it (APRP_DRIVER_MODE=1), and a build
 * that pairs this adapter with a mode-0 driver stops rather than run as if it
 * were the stock arm. */
#include "port.h"
#include <capstone/capability.h>
#include <capstone/capability-slot.h>
#include <stdlib.h>

#define RECORDS (APRP_PAYLOAD_BYTES / (2 * 4096UL))

struct record {
  void *memory; /* what malloc returned: freed at discard */
  void *owner;  /* CDERIVE of memory over the node, MANAGE; the free list's alias */
  void *issued; /* the live child while a pool holds the node, else NULL */
  size_t size;
  unsigned live, released_once, discarded, lent;
};
static struct record *records;
static unsigned count;
static uint64_t reuses, releases, discards;
static size_t backing;

void aprp_payload_init(void *region) { (void)region; /* malloc is the region */ }
void aprp_set_mode(unsigned mode) {
  if (mode != 1)
    aprp_fail(506);
  records = aprp_meta_calloc(RECORDS, sizeof *records);
  if (!records)
    aprp_fail(503);
}
/* A child of `parent` over [p, p + size). */
static void *derive(void *parent, const void *p, size_t size) {
  return capstone_cap_derive(parent, (uintptr_t)p - __builtin_capstone_cap_get_base(parent), size);
}
/* Address equality, not tag: a revoked child still names its node. */
static struct record *record_for(const void *node) {
  for (unsigned i = 0; i < count; ++i)
    if ((uintptr_t)records[i].owner == (uintptr_t)node)
      return &records[i];
  aprp_fail(516);
}
void *aprp_node_backing(size_t size) {
  if (count == RECORDS || !size || (size & 4095))
    return NULL; /* upstream turns this into APR_ENOMEM */
  void *memory = aligned_alloc(4096, size);
  if (!memory)
    return NULL;
  struct record *r = &records[count++];
  r->memory = memory;
  r->owner = derive(memory, memory, size);
  r->size = size;
  backing += size;
  return r->owner;
}
void *aprp_node_issue(void *node) {
  struct record *r = record_for(node);
  if (r->live || r->discarded || r->issued)
    aprp_fail(518);
  if (r->released_once)
    ++reuses;
  r->live = 1;
  r->issued = derive(r->owner, r->owner, r->size);
  return r->issued;
}
void *aprp_node_release(void *node) {
  struct record *r = record_for(node);
  if (!r->live || r->discarded || !r->issued)
    aprp_fail(519);
  capstone_cap_revoke_child(r->owner, r->issued);
  r->issued = NULL;
  r->live = 0;
  r->released_once = 1;
  r->lent = 0;
  ++releases;
  return r->owner;
}
void aprp_node_discard(void *node) {
  struct record *r = record_for(node);
  if (r->live || r->discarded)
    aprp_fail(520);
  r->discarded = 1;
  ++discards;
  free(r->memory);
}
unsigned aprp_mode(void) { return 1; }
/* The bucket allocator carves the issued child itself: its pieces are CDERIVE
 * children of it, so the node's release takes them all. `rest` is unused. */
void *aprp_node_lend(void *node, struct capstone_cap_slot *rest) {
  struct record *r = record_for(node);
  if (!r->live || r->discarded || r->lent)
    aprp_fail(521);
  r->lent = 1;
  rest->c = NULL;
  return r->issued;
}
void aprp_stats(struct aprp_header *out) {
  out->nodes = count;
  out->node_reuses = reuses;
  out->node_releases = releases;
  out->node_discards = discards;
  out->backing_used = backing;
  out->metadata = aprp_meta_used();
}
