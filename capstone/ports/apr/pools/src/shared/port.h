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
/* The mode the run was started in: 0 spatial, 1 sublet. */
unsigned aprp_mode(void);

/* Lend a live node to a client that carves it -- apr-util's bucket allocator.
 * Domain, mode 1: the node's alias is revoked, a handle senior to the whole
 * node stays in the record, the part behind the memnode header moves out
 * LINEAR into `rest` for the client to carve, and the header is retaken as the
 * alias upstream keeps. Mode 0, native and CheriBSD: `rest` is cleared and the
 * node's own pointer is returned; the client carves by address. The node's
 * release revokes everything carved from it, as before. */
struct capstone_cap_slot; /* the runtime's slot type; declared, not included, so every target reads this header */
void *aprp_node_lend(void *node, struct capstone_cap_slot *rest);

/* The bucket allocator's seam (APRP_BUCKETS). Upstream still decides which
 * block, which node and in which order; the adapter supplies the authority
 * over a piece and keeps the freelist the freed node can no longer hold. */
void *aprb_block_lend(void *block);
void *aprb_carve(void *block, size_t size);
void aprb_file(void *list, void *node);
struct apr_memnode_t;
void *aprb_reissue(void *list, struct apr_memnode_t **memnode);
void aprb_blocks_returning(void *blocks);
void *aprb_probe(void *mem);
void aprb_stats(unsigned long *pieces, unsigned long *reissues, unsigned long *files);

/* Supplied by the program built through the seam: the corpus case, or a test. */
void aprp_replay(const struct aprp_header *input, struct aprp_header *out);
#endif
