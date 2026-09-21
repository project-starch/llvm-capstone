/* The lifetime ledger behind both allocators: which storage is a slab page, a
 * chunk of one, or a cache object; who holds it; and when its authority is
 * renewed. Upstream still decides everything about placement -- slabs.c which
 * class a chunk belongs to and which chunk the next slabs_alloc pops, cache.c
 * which object comes off its STAILQ. This file decides only who may still
 * address the storage, through the authority layer (Sublet primitives in a
 * domain, plain pointers hosted), and is the same file on both, so the two
 * targets measure one bookkeeping.
 *
 * Slabs. A page is carved from the payload when memory_allocate asks and never
 * goes back, as upstream's pages never do. When do_slabs_newslab has zeroed it
 * the page's alias is exchanged for one region per chunk. A chunk is then
 * carved, on the free list, or live:
 *   mode 0 (spatial): a chunk keeps the alias it was carved with. A pointer
 *   held across slabs_free still names the storage, which by then is the next
 *   item's -- the defect as upstream ships it.
 *   mode 1 (sublet): issue and release each revoke the chunk and mint a fresh
 *   alias, so that pointer is dead. Revocation clears the chunk; slabs_clsid,
 *   which upstream keeps on free memory, is put back from the page's record.
 * Objects. cache.c's are the same shape one level down -- malloc'd once,
 * pushed and popped, freed only over a limit or at destroy -- and get the same
 * treatment, one region per object. */
#include "mc_slabs_shim.h"
#include "port.h"

#define MIN_PAGE (64UL * 1024)
#define MAX_PAGES (MCP_PAGE_HALF / MIN_PAGE)
#define GRAINS (MCP_PAGE_HALF / MCP_GRAIN)
#define MAX_OBJECTS 8192

enum state { CARVED, FREE, LIVE, GONE };

struct chunk {
  capstone_cap_slot region;
  void *alias;
  unsigned state, issued_once;
};
struct page {
  capstone_cap_slot region;
  void *alias;
  uintptr_t base;
  size_t size;
  struct chunk *chunks; /* perslab of them once the page is carved */
  uint32_t chunk_size, perslab;
  unsigned clsid, discarded;
};
struct object {
  capstone_cap_slot region;
  void *alias;
  uintptr_t base;
  size_t size;
  unsigned state;
};
static struct page *pages;
static uint16_t *page_map; /* page number + 1 for every grain of a page */
static struct object *objects;
static unsigned page_count, object_count, protected_mode;
static uint64_t chunk_reuses, chunk_releases, object_reuses, object_releases;

void mcp_payload_init(void *payload) { mcp_authority_init(payload); }
void mcp_set_mode(unsigned mode) {
  if (mode > 1)
    mcp_fail(502);
  /* A hosted execution does not acquire revocation from a mode number. */
  if (mode == 1 && !mcp_authority_can_revoke())
    mcp_fail(505);
  protected_mode = mode;
  mcp_authority_set_mode(mode);
  pages = mcp_meta_calloc(MAX_PAGES, sizeof *pages);
  page_map = mcp_meta_calloc(GRAINS, sizeof *page_map);
  objects = mcp_meta_calloc(MAX_OBJECTS, sizeof *objects);
  if (!pages || !page_map || !objects)
    mcp_fail(503);
}

/* ---- slabs ---------------------------------------------------------------- */
static struct page *page_for(uintptr_t address) {
  uintptr_t base = mcp_authority_base(MCP_PAGES);
  if (address < base || address - base >= MCP_PAGE_HALF)
    mcp_fail(515);
  unsigned number = page_map[(address - base) / MCP_GRAIN];
  if (!number)
    mcp_fail(516);
  return &pages[number - 1];
}
static struct chunk *chunk_for(uintptr_t address, struct page **holder) {
  struct page *pg = page_for(address);
  uintptr_t offset = address - pg->base;
  if (!pg->chunks || offset % pg->chunk_size || offset / pg->chunk_size >= pg->perslab)
    mcp_fail(517);
  struct chunk *c = &pg->chunks[offset / pg->chunk_size];
  /* Upstream only ever hands the adapter a chunk's own address. */
  if ((uintptr_t)c->alias != address)
    mcp_fail(517);
  *holder = pg;
  return c;
}
/* Revocation cleared the chunk; put back the one field upstream keeps on free
 * memory and reads across a transition. Written through the alias, so the
 * store is checked. */
static void *renew_chunk(struct page *pg, struct chunk *c) {
  c->alias = mcp_authority_renew(&c->region);
  ((item *)c->alias)->slabs_clsid = (uint8_t)pg->clsid;
  return c->alias;
}

void *mcp_page_backing(size_t size) {
  if (size < MIN_PAGE || size % MCP_GRAIN)
    mcp_fail(504);
  if (page_count == MAX_PAGES)
    return NULL; /* upstream's memory_allocate returning NULL: no new slab */
  struct page *pg = &pages[page_count];
  if (!mcp_authority_carve(MCP_PAGES, size, &pg->region, &pg->base))
    return NULL;
  pg->size = size;
  pg->alias = mcp_authority_take(&pg->region);
  uintptr_t base = mcp_authority_base(MCP_PAGES);
  for (uintptr_t grain = pg->base; grain < pg->base + size; grain += MCP_GRAIN)
    page_map[(grain - base) / MCP_GRAIN] = (uint16_t)(page_count + 1);
  ++page_count;
  return pg->alias;
}
void *mcp_page_carve(void *page, unsigned id, uint32_t chunk_size, uint32_t perslab) {
  struct page *pg = page_for((uintptr_t)page);
  if ((uintptr_t)page != pg->base || pg->chunks || pg->discarded)
    mcp_fail(506);
  /* A chunk is a region, and a region is whole capabilities (CHUNK_ALIGN_BYTES=16). */
  if (!chunk_size || chunk_size % 16 || !perslab || (size_t)chunk_size * perslab > pg->size)
    mcp_fail(507);
  pg->chunks = mcp_meta_calloc(perslab, sizeof *pg->chunks);
  if (!pg->chunks)
    mcp_fail(503);
  pg->chunk_size = chunk_size;
  pg->perslab = perslab;
  pg->clsid = id;
  /* The whole-page alias upstream zeroed through dies here; the region comes
   * back linear and is cut, in order, into the chunks the split will file. */
  mcp_authority_reclaim(&pg->region);
  pg->alias = NULL;
  for (uint32_t i = 0; i < perslab; ++i) {
    struct chunk *c = &pg->chunks[i];
    mcp_authority_split(&pg->region, pg->base + (uintptr_t)(i + 1) * chunk_size, &c->region);
    c->alias = mcp_authority_take(&c->region);
    c->state = CARVED;
  }
  return pg->chunks[0].alias;
}
void *mcp_chunk_at(void *page, unsigned index) {
  struct page *pg = page_for((uintptr_t)page);
  if (!pg->chunks || index >= pg->perslab)
    mcp_fail(508);
  return pg->chunks[index].alias;
}
void *mcp_chunk_release(void *chunk, unsigned id) {
  struct page *pg;
  struct chunk *c = chunk_for((uintptr_t)chunk, &pg);
  if (id != pg->clsid)
    mcp_fail(519);
  if (c->state == CARVED) {
    /* The split filing a fresh chunk: nobody has held it, no alias to kill. */
    c->state = FREE;
    return c->alias;
  }
  if (c->state != LIVE)
    mcp_fail(519);
  c->state = FREE;
  ++chunk_releases;
  return protected_mode ? renew_chunk(pg, c) : c->alias;
}
void *mcp_chunk_issue(void *chunk) {
  struct page *pg;
  struct chunk *c = chunk_for((uintptr_t)chunk, &pg);
  if (c->state != FREE)
    mcp_fail(518);
  if (c->issued_once)
    ++chunk_reuses;
  c->issued_once = 1;
  c->state = LIVE;
  return protected_mode ? renew_chunk(pg, c) : c->alias;
}
void mcp_page_discard(void *page) {
  struct page *pg = page_for((uintptr_t)page);
  if ((uintptr_t)page != pg->base || pg->chunks || pg->discarded)
    mcp_fail(520);
  pg->discarded = 1;
  /* Only memory_release comes here, with a page from the global pool that
   * was never split. Nothing is handed out again; reclaiming the authority
   * is all there is. */
  if (protected_mode)
    mcp_authority_reclaim(&pg->region);
}

/* ---- cache.c objects ------------------------------------------------------ */
static struct object *object_for(uintptr_t address) {
  /* Objects are carved in address order, so the table is sorted. */
  unsigned lo = 0, hi = object_count;
  while (lo < hi) {
    unsigned mid = (lo + hi) / 2;
    if (objects[mid].base < address)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo == object_count || objects[lo].base != address ||
      (uintptr_t)objects[lo].alias != address)
    mcp_fail(525);
  return &objects[lo];
}
void *mcp_object_backing(size_t size) {
  size_t rounded = (size + 15) & ~(size_t)15;
  if (!rounded)
    mcp_fail(524);
  if (object_count == MAX_OBJECTS)
    return NULL; /* upstream's malloc returning NULL: cache_alloc fails */
  struct object *o = &objects[object_count];
  if (!mcp_authority_carve(MCP_OBJECTS, rounded, &o->region, &o->base))
    return NULL;
  o->size = rounded;
  o->alias = mcp_authority_take(&o->region);
  o->state = LIVE;
  ++object_count;
  return o->alias;
}
void *mcp_object_issue(void *object) {
  struct object *o = object_for((uintptr_t)object);
  if (o->state != FREE)
    mcp_fail(528);
  o->state = LIVE;
  ++object_reuses; /* an object is only ever on the list after being live */
  if (protected_mode)
    o->alias = mcp_authority_renew(&o->region);
  return o->alias;
}
void *mcp_object_release(void *object) {
  struct object *o = object_for((uintptr_t)object);
  if (o->state != LIVE)
    mcp_fail(529);
  o->state = FREE;
  ++object_releases;
  if (protected_mode)
    o->alias = mcp_authority_renew(&o->region);
  return o->alias;
}
void mcp_object_discard(void *object) {
  struct object *o = object_for((uintptr_t)object);
  if (o->state == GONE)
    mcp_fail(530);
  o->state = GONE;
  if (protected_mode)
    mcp_authority_reclaim(&o->region);
}

void mcp_stats(struct mcp_header *out) {
  out->pages = page_count;
  out->chunk_reuses = chunk_reuses;
  out->chunk_releases = chunk_releases;
  out->object_reuses = object_reuses;
  out->object_releases = object_releases;
  out->backing_used = mcp_authority_used(MCP_PAGES) + mcp_authority_used(MCP_OBJECTS);
  out->metadata = mcp_meta_used();
}
