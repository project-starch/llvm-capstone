/* CheriBSD, stock: pages and objects come from the platform's own malloc and
 * go back through its own free, exactly as upstream memcached does.
 *
 * This is the arm that asks the shipping temporal mechanism the question at
 * the level where it lives. libc revocation acts on free(); slabs pushes a
 * freed chunk on its class's slots list and pops it straight back, and
 * cache.c pushes an object on its STAILQ and pops it uncleared -- neither
 * calls free() on that path. The only free()s here are a page from the global
 * pool under memory_release and an object cache.c discards over its limit or
 * at cache_destroy. Chunks have no storage of their own: they are offsets into
 * a malloc'd page, addressed through the page's capability, as on any CheriBSD
 * build of memcached. Mode 0 only: there is no authority to revoke here that
 * libc does not already own, and a mode number does not conjure one. */
#include "mc_slabs_shim.h"
#include "port.h"
#include <stdlib.h>

#define MAX_PAGES 768
#define MAX_OBJECTS 8192

enum state { CARVED, FREE, LIVE, GONE };

struct page {
  unsigned char *base;
  size_t size;
  unsigned char *states; /* one per chunk once carved */
  uint32_t chunk_size, perslab;
  unsigned clsid, discarded;
};
struct object {
  unsigned char *base;
  size_t size;
  unsigned state;
};
static struct page *pages;
static struct object *objects;
static unsigned char **issued_once; /* per page, one flag per chunk */
static unsigned page_count, object_count;
static uint64_t chunk_reuses, chunk_releases, object_reuses, object_releases;
static size_t backing;

void mcp_payload_init(void *region) { (void)region; /* malloc is the region */ }
void mcp_set_mode(unsigned mode) {
  if (mode != 0)
    mcp_fail(505);
  pages = mcp_meta_calloc(MAX_PAGES, sizeof *pages);
  issued_once = mcp_meta_calloc(MAX_PAGES, sizeof *issued_once);
  objects = mcp_meta_calloc(MAX_OBJECTS, sizeof *objects);
  if (!pages || !issued_once || !objects)
    mcp_fail(503);
}

/* malloc places units where it likes, so no page map: the corpus holds a
 * handful of pages and objects and a scan is exact. Address equality, not
 * tag: a discarded unit's pointer may be untagged and must still be found. */
static struct page *page_for(const void *address, unsigned *index) {
  __UINTPTR_TYPE__ at = (__UINTPTR_TYPE__)address;
  for (unsigned i = 0; i < page_count; ++i) {
    __UINTPTR_TYPE__ base = (__UINTPTR_TYPE__)pages[i].base;
    if (at >= base && at - base < pages[i].size) {
      *index = i;
      return &pages[i];
    }
  }
  mcp_fail(516);
}
static unsigned chunk_for(const void *address, struct page **holder, unsigned *page_index) {
  struct page *pg = page_for(address, page_index);
  __UINTPTR_TYPE__ offset = (__UINTPTR_TYPE__)address - (__UINTPTR_TYPE__)pg->base;
  if (!pg->states || offset % pg->chunk_size || offset / pg->chunk_size >= pg->perslab)
    mcp_fail(517);
  *holder = pg;
  return (unsigned)(offset / pg->chunk_size);
}

void *mcp_page_backing(size_t size) {
  if (page_count == MAX_PAGES)
    return NULL;
  void *page = malloc(size);
  if (!page)
    return NULL;
  struct page *pg = &pages[page_count++];
  pg->base = page;
  pg->size = size;
  backing += size;
  return page;
}
void *mcp_page_carve(void *page, unsigned id, uint32_t chunk_size, uint32_t perslab) {
  unsigned index;
  struct page *pg = page_for(page, &index);
  if ((__UINTPTR_TYPE__)page != (__UINTPTR_TYPE__)pg->base || pg->states || pg->discarded)
    mcp_fail(506);
  if (!chunk_size || !perslab || (size_t)chunk_size * perslab > pg->size)
    mcp_fail(507);
  pg->states = mcp_meta_calloc(perslab, 1);
  issued_once[index] = mcp_meta_calloc(perslab, 1);
  if (!pg->states || !issued_once[index])
    mcp_fail(503);
  pg->chunk_size = chunk_size;
  pg->perslab = perslab;
  pg->clsid = id;
  /* The page's own capability stays what upstream would have had: malloc's
   * bounds over the whole page, every chunk an offset into it. */
  return pg->base;
}
void *mcp_chunk_at(void *page, unsigned index) {
  unsigned page_index;
  struct page *pg = page_for(page, &page_index);
  if (!pg->states || index >= pg->perslab)
    mcp_fail(508);
  return pg->base + (size_t)index * pg->chunk_size;
}
void *mcp_chunk_release(void *chunk, unsigned id) {
  struct page *pg;
  unsigned page_index;
  unsigned i = chunk_for(chunk, &pg, &page_index);
  if (id != pg->clsid)
    mcp_fail(519);
  if (pg->states[i] == CARVED) {
    pg->states[i] = FREE;
    return pg->base + (size_t)i * pg->chunk_size;
  }
  if (pg->states[i] != LIVE)
    mcp_fail(519);
  pg->states[i] = FREE;
  ++chunk_releases;
  return pg->base + (size_t)i * pg->chunk_size;
}
void *mcp_chunk_issue(void *chunk) {
  struct page *pg;
  unsigned page_index;
  unsigned i = chunk_for(chunk, &pg, &page_index);
  if (pg->states[i] != FREE)
    mcp_fail(518);
  if (issued_once[page_index][i])
    ++chunk_reuses;
  issued_once[page_index][i] = 1;
  pg->states[i] = LIVE;
  return pg->base + (size_t)i * pg->chunk_size;
}
void mcp_page_discard(void *page) {
  unsigned index;
  struct page *pg = page_for(page, &index);
  if ((__UINTPTR_TYPE__)page != (__UINTPTR_TYPE__)pg->base || pg->states || pg->discarded)
    mcp_fail(520);
  pg->discarded = 1;
  free(pg->base); /* memory_release: a page from the global pool, never split */
}

static struct object *object_for(const void *address) {
  for (unsigned i = 0; i < object_count; ++i)
    if ((__UINTPTR_TYPE__)objects[i].base == (__UINTPTR_TYPE__)address)
      return &objects[i];
  mcp_fail(525);
}
void *mcp_object_backing(size_t size) {
  if (!size)
    mcp_fail(524);
  if (object_count == MAX_OBJECTS)
    return NULL;
  void *object = malloc(size);
  if (!object)
    return NULL;
  struct object *o = &objects[object_count++];
  o->base = object;
  o->size = size;
  o->state = LIVE;
  backing += size;
  return object;
}
void *mcp_object_issue(void *object) {
  struct object *o = object_for(object);
  if (o->state != FREE)
    mcp_fail(528);
  o->state = LIVE;
  ++object_reuses;
  return o->base;
}
void *mcp_object_release(void *object) {
  struct object *o = object_for(object);
  if (o->state != LIVE)
    mcp_fail(529);
  o->state = FREE;
  ++object_releases;
  return o->base;
}
void mcp_object_discard(void *object) {
  struct object *o = object_for(object);
  if (o->state == GONE)
    mcp_fail(530);
  o->state = GONE;
  free(o->base); /* cache.c over its limit, or cache_destroy */
}
void mcp_stats(struct mcp_header *out) {
  out->pages = page_count;
  out->chunk_reuses = chunk_reuses;
  out->chunk_releases = chunk_releases;
  out->object_reuses = object_reuses;
  out->object_releases = object_releases;
  out->backing_used = backing;
  out->metadata = mcp_meta_used();
}
