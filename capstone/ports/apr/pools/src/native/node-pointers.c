/* Hosted: the same seam with plain pointers and no authority. A node keeps the
 * address it was carved with for the whole run, which is exactly what upstream
 * gives every consumer -- so the native arms measure the allocator as
 * shipped. Selecting mode 1 here is refused rather than silently accepted: a
 * native execution does not acquire revocation from a mode number. */
#include "port.h"

#define PAGE 4096UL
#define MIN_NODE (2 * PAGE)
#define RECORDS (APRP_PAYLOAD_BYTES / MIN_NODE)
#define PAGES (APRP_PAYLOAD_BYTES / PAGE)

struct record {
  unsigned char *node;
  size_t size;
  unsigned live, released_once, discarded;
};
static unsigned char *base;
static size_t used;
static struct record *records;
static uint16_t *page_map;
static unsigned count;
static uint64_t reuses, releases, discards;

void aprp_payload_init(void *region) {
  if (((uintptr_t)region & (PAGE - 1)))
    aprp_fail(501);
  base = region;
  used = 0;
}
void aprp_set_mode(unsigned mode) {
  if (mode != 0)
    aprp_fail(505);
  records = aprp_meta_calloc(RECORDS, sizeof *records);
  page_map = aprp_meta_calloc(PAGES, sizeof *page_map);
  if (!records || !page_map)
    aprp_fail(503);
}
static struct record *record_for(const void *node) {
  uintptr_t off = (unsigned char *)node - base;
  if ((unsigned char *)node < base || off >= APRP_PAYLOAD_BYTES || (off & (PAGE - 1)))
    aprp_fail(515);
  unsigned slot = page_map[off / PAGE];
  if (!slot)
    aprp_fail(516);
  struct record *r = &records[slot - 1];
  if (r->node != node)
    aprp_fail(517);
  return r;
}
void *aprp_node_backing(size_t size) {
  if (size < MIN_NODE || (size & (PAGE - 1)))
    aprp_fail(504);
  if (count == RECORDS || used > APRP_PAYLOAD_BYTES - size)
    return NULL;
  struct record *r = &records[count];
  r->node = base + used;
  r->size = size;
  for (size_t page = used; page < used + size; page += PAGE)
    page_map[page / PAGE] = (uint16_t)(count + 1);
  used += size;
  ++count;
  return r->node;
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
}
void aprp_stats(struct aprp_header *out) {
  out->nodes = count;
  out->node_reuses = reuses;
  out->node_releases = releases;
  out->node_discards = discards;
  out->backing_used = used;
  out->metadata = aprp_meta_used();
}
