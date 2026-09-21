/* CheriBSD, stock: nodes come from the platform's own malloc and go back
 * through its own free, exactly as upstream APR does.
 *
 * This is the arm that asks the shipping temporal mechanism the question at
 * the level where it lives. libc revocation acts on free(); APR files a
 * destroyed pool's node on its own free list and pops it straight back for the
 * next pool, and under the default APR_ALLOCATOR_MAX_FREE_UNLIMITED never
 * calls free() on that path. The only free() here is aprp_node_discard, from
 * apr_allocator_destroy at the very end. Mode 0 only: there is no authority to
 * revoke here that libc does not already own, and a mode number does not
 * conjure one. */
#include "port.h"
#include <stdlib.h>

#define RECORDS (APRP_PAYLOAD_BYTES / (2 * 4096UL))

struct record {
  void *node;
  size_t size;
  unsigned live, released_once, discarded;
};
static struct record *records;
static unsigned count;
static uint64_t reuses, releases, discards;
static size_t backing;

void aprp_payload_init(void *region) { (void)region; /* malloc is the region */ }
void aprp_set_mode(unsigned mode) {
  if (mode != 0)
    aprp_fail(505);
  records = aprp_meta_calloc(RECORDS, sizeof *records);
  if (!records)
    aprp_fail(503);
}
/* malloc places nodes where it likes, so no page map: the corpus holds a handful
 * of nodes and a scan is exact. Address equality, not tag: a discarded node's
 * pointer may be untagged and must still be found. */
static struct record *record_for(const void *node) {
  for (unsigned i = 0; i < count; ++i)
    if ((__UINTPTR_TYPE__)records[i].node == (__UINTPTR_TYPE__)node)
      return &records[i];
  aprp_fail(516);
}
void *aprp_node_backing(size_t size) {
  if (count == RECORDS)
    return NULL;
  void *node = malloc(size);
  if (!node)
    return NULL;
  struct record *r = &records[count++];
  r->node = node;
  r->size = size;
  backing += size;
  return node;
}
void *aprp_node_issue(void *node) {
  struct record *r = record_for(node);
  if (r->live || r->discarded)
    aprp_fail(518);
  if (r->released_once)
    ++reuses;
  r->live = 1;
  return r->node;
}
void *aprp_node_release(void *node) {
  struct record *r = record_for(node);
  if (!r->live || r->discarded)
    aprp_fail(519);
  r->live = 0;
  r->released_once = 1;
  ++releases;
  return r->node;
}
void aprp_node_discard(void *node) {
  struct record *r = record_for(node);
  if (r->live || r->discarded)
    aprp_fail(520);
  r->discarded = 1;
  ++discards;
  free(r->node); /* the one and only free(): apr_allocator_destroy */
}
void aprp_stats(struct aprp_header *out) {
  out->nodes = count;
  out->node_reuses = reuses;
  out->node_releases = releases;
  out->node_discards = discards;
  out->backing_used = backing;
  out->metadata = aprp_meta_used();
}
