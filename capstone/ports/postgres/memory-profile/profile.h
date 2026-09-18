#ifndef PG_MEMORY_PROFILE_H
#define PG_MEMORY_PROFILE_H
#include "a11trace.h"
#include <stddef.h>

/* Backing bytes contain payload; these columns must not be summed together.
 * Record-storage bytes exclude the profile itself and global QEMU metadata. */
struct pg_memory_backing {
  unsigned long backing_bytes, assigned_bytes, block_bytes, stranded_bytes;
  unsigned long metadata_live, metadata_reserved;
  unsigned long pools, blocks, entries;
  unsigned long nodes_created, revokes, init_bytes;
  unsigned long arena_capacity;
};
void pg_memory_backing(struct pg_memory_backing *out);
void pg_memory_begin(const struct a11_rec *footer);
void pg_memory_event(unsigned long index, const struct a11_rec *event,
                     const unsigned int *lengths);
void pg_memory_end(unsigned long events);
#endif
