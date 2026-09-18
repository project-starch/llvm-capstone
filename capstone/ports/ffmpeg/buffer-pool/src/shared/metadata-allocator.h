#ifndef FFPOOL_METADATA_ALLOCATOR_H
#define FFPOOL_METADATA_ALLOCATOR_H

#include <stddef.h>

/* Supplies the av_malloc/av_free family from a caller-owned bounded arena. */
void ff2_memory_init(void *base, size_t bytes);
size_t ff2_metadata_used(void);

#endif
