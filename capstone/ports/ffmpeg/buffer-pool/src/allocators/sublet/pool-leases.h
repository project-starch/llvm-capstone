#ifndef FFPOOL_POOL_LEASES_H
#define FFPOOL_POOL_LEASES_H

struct payload_block;
struct ff2_header;

/* Mode 2 issues fresh authority for each lease and revokes it on pool return.
 */
void *ff2_sublet_issue(struct payload_block *block);
void ff2_sublet_return(struct payload_block *block);

/* The underlying Sublet primitives count operations per translation unit.
 * Add this module's lease operations to the backend's backing-lifetime counts.
 */
void ff2_sublet_add_stats(struct ff2_header *report);

#endif
