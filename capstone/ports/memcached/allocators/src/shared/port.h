/* The allocators port's two interfaces in one header.
 *
 * The first is the fixed-width protocol between a launcher and the program:
 * a 12-word header, a trace of events after it, and the same header written
 * back as the report. It is the pymalloc and APR ports' shape with memcached's
 * names, so the corpus runners and the guest loader read the same offsets.
 *
 * The second is what the patched slabs.c and cache.c ask of their adapter.
 * Upstream still decides everything about placement -- the class table, which
 * chunk the next slabs_alloc pops, which object the next cache_alloc pops. The
 * adapter supplies the storage under a page, a chunk or an object and, in the
 * protected mode, the authority over it. */
#ifndef MEMCACHED_ALLOCATORS_PORT_H
#define MEMCACHED_ALLOCATORS_PORT_H
#include <stddef.h>
#include <stdint.h>
#include <capstone/capability-slot.h>

#define MCP_PAYLOAD_BYTES (64UL * 1024 * 1024)
/* Slab pages are carved upward from the bottom of the payload, cache objects
 * from the top half down; neither ever returns, as neither does upstream. */
#define MCP_PAGE_HALF (48UL * 1024 * 1024)
#define MCP_OBJECT_HALF (MCP_PAYLOAD_BYTES - MCP_PAGE_HALF)
#define MCP_META_BYTES (16UL * 1024 * 1024)
#define MCP_FILE_BYTES (8UL * 1024 * 1024)
#define MCP_GRAIN 4096UL
#define MCP_MAGIC UINT64_C(0x315342414c53434d) /* "MCSLABS1", little-endian */
enum { MCP_PAGES = 0, MCP_OBJECTS = 1 };

struct mcp_event {
  uint64_t op, id, size, value;
};
/* Field order is load-bearing: the domain runner reads `completed` at word 4. */
struct mcp_header {
  uint64_t magic, count, mode, status, completed, pages, chunk_reuses,
      chunk_releases, object_reuses, object_releases, backing_used, metadata;
};

/* Ends the run with a code. In a domain it returns to the launcher with the
 * code in the report and the result word; hosted, it exits non-zero. Codes are
 * three digits, grouped: 4xx entry, 5xx adapter, 6xx services, 7xx case. */
_Noreturn void mcp_fail(unsigned code);

/* The metadata heap: adapter records, the slab lists and cache.c's control
 * blocks live here, never inside a page, so revoking a chunk never revokes its
 * own bookkeeping. grow_pointers copies element-wise because the slab list
 * holds page aliases, which are capabilities in a domain. */
void mcp_meta_init(void *metadata);
void *mcp_meta_alloc(size_t);
void *mcp_meta_calloc(size_t, size_t);
void mcp_meta_free(void *);
char *mcp_meta_strdup(const char *);
void **mcp_meta_grow_pointers(void **old, size_t count, size_t new_count);
size_t mcp_meta_used(void);

/* The payload region and the mode. mode 0 is spatial: a chunk or object keeps
 * the alias it was carved with across the free list, so a stale pointer still
 * names live storage. mode 1 is sublet: release and issue each revoke the
 * unit's authority and mint a fresh alias, so a stale pointer is a dead one. */
void mcp_payload_init(void *payload);
void mcp_set_mode(unsigned mode);

/* The slabs seam. backing/discard replace malloc/free of a page; carve turns a
 * zeroed page into its class's chunks and returns the first chunk's alias,
 * whose address is the page's; chunk_at names chunk `index` of that page for
 * the split. issue/release sit on the slots-list transitions where upstream
 * reuses a chunk WITHOUT ever calling free. Both return the alias upstream
 * must use from then on -- after a revoke the old one is dead -- with the one
 * header field upstream keeps on free memory, slabs_clsid, restored. */
void *mcp_page_backing(size_t size);
void mcp_page_discard(void *page);
void *mcp_page_carve(void *page, unsigned id, uint32_t chunk_size, uint32_t perslab);
void *mcp_chunk_at(void *page, unsigned index);
void *mcp_chunk_issue(void *chunk);
void *mcp_chunk_release(void *chunk, unsigned id);

/* The cache.c seam: the same four transitions for one object. */
void *mcp_object_backing(size_t size);
void *mcp_object_issue(void *object);
void *mcp_object_release(void *object);
void mcp_object_discard(void *object);

void mcp_stats(struct mcp_header *out);

/* Supplied by the program built through the seam: the corpus case, or a test. */
void mcp_replay(const struct mcp_header *input, struct mcp_header *out);

/* The authority layer, one file per platform, underneath the ledger in
 * src/shared/leases.c. Regions are capstone_cap_slot on both; hosted, a slot
 * holds a plain pointer and there is nothing to revoke. */
void mcp_authority_init(void *payload);
int mcp_authority_can_revoke(void);
uintptr_t mcp_authority_base(unsigned half);
size_t mcp_authority_used(unsigned half);
/* The next `size` bytes of a half; 0 when it is exhausted. */
int mcp_authority_carve(unsigned half, size_t size, capstone_cap_slot *out, uintptr_t *base);
/* The prefix of a linear region up to `end`; the rest stays in `from`. */
void mcp_authority_split(capstone_cap_slot *from, uintptr_t end, capstone_cap_slot *out);
/* An alias to a region; the slot keeps the handle. */
void *mcp_authority_take(capstone_cap_slot *region);
/* Revoke every alias and mint a fresh one. Hosted: refused, 505. */
void *mcp_authority_renew(capstone_cap_slot *region);
/* Revoke every alias; the region rests, linear, in its slot. */
void mcp_authority_reclaim(capstone_cap_slot *region);
#endif
