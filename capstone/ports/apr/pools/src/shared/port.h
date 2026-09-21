/* The pools port's two interfaces in one header.
 *
 * The first is the fixed-width protocol between a launcher and the program:
 * a 12-word header, a trace of events after it, and the same header written
 * back as the report. It is the pymalloc port's shape with APR's names, so the
 * corpus runners and the guest loader read the same offsets on both.
 *
 * The second is what the patched apr_pools.c asks of its adapter. Upstream
 * still decides everything about pools and nodes -- sizes, buckets, free-list
 * order, which node a pool_create pops back. The adapter supplies the storage
 * under a node and, in the protected mode, the authority over it. */
#ifndef APR_POOLS_PORT_H
#define APR_POOLS_PORT_H
#include <stddef.h>
#include <stdint.h>

#define APRP_PAYLOAD_BYTES (64UL * 1024 * 1024)
#define APRP_META_BYTES (16UL * 1024 * 1024)
#define APRP_FILE_BYTES (8UL * 1024 * 1024)
#define APRP_MAGIC UINT64_C(0x314c4f4f50525041) /* "APRPOOL1", little-endian */

struct aprp_event {
  uint64_t op, id, size, value;
};
/* Field order is load-bearing: the domain runner reads `completed` at word 4. */
struct aprp_header {
  uint64_t magic, count, mode, status, completed, nodes, node_reuses,
      node_releases, node_discards, backing_used, metadata, checksum;
};

/* Ends the run with a code. In a domain it returns to the launcher with the
 * code in the report and the result word; hosted, it exits non-zero. Codes are
 * three digits, grouped: 4xx entry, 5xx adapter, 6xx services, 7xx case. */
_Noreturn void aprp_fail(unsigned code);

/* The metadata heap: adapter records and the allocator struct live here,
 * never inside a node, so revoking a node never revokes its own bookkeeping. */
void aprp_meta_init(void *metadata);
void *aprp_meta_alloc(size_t);
void *aprp_meta_calloc(size_t, size_t);
void aprp_meta_free(void *);
size_t aprp_meta_used(void);

/* The payload region and the mode. mode 0 is spatial: a node's alias persists
 * across the free list, so a stale handle still names live storage. mode 1 is
 * sublet: release and issue each revoke the node's authority and mint a fresh
 * alias, so a stale handle is a dead one. */
void aprp_payload_init(void *payload);
void aprp_set_mode(unsigned mode);

/* The seam apr_pools.c is patched to. backing/discard replace malloc/free of a
 * node; issue/release sit on the free-list transitions where upstream reuses
 * a node WITHOUT ever calling free. Both return the alias upstream must use
 * from then on -- after a revoke the old one is dead -- with the node header
 * fields upstream needs (index, endp) restored, because revocation clears
 * the region. */
void *aprp_node_backing(size_t size);
void aprp_node_discard(void *node);
void *aprp_node_issue(void *node);
void *aprp_node_release(void *node);

void aprp_stats(struct aprp_header *out);

/* Supplied by the program built through the seam: the corpus case, or a test. */
void aprp_replay(const struct aprp_header *input, struct aprp_header *out);
#endif
