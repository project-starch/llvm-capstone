/* What patch 0006 (MC_CAPSTONE_SLAB_SUBLET) asks of the glue in mcapp-slab-sublet.c: the component
 * port's seam (ports/memcached/allocators/src/shared/port.h), one locked wrapper per hook, so the
 * patched slabs.c, cache.c and memcached.c never call the adapter directly. */
#ifndef MCAPP_SLAB_SUBLET_H
#define MCAPP_SLAB_SUBLET_H
#include <stddef.h>
#include <stdint.h>

void mcs_init(void);

void *mcs_page_backing(size_t size);
void mcs_page_discard(void *page);
void *mcs_page_carve(void *page, unsigned id, uint32_t chunk_size, uint32_t perslab);
void *mcs_chunk_at(void *page, unsigned index);
void *mcs_chunk_issue(void *chunk);
void *mcs_chunk_release(void *chunk, unsigned id);

void *mcs_object_backing(size_t size);
void *mcs_object_issue(void *object);
void *mcs_object_release(void *object);
void mcs_object_discard(void *object);

void *mcs_meta_calloc(size_t n, size_t size);
void mcs_meta_free(void *p);
char *mcs_meta_strdup(const char *s);
void **mcs_meta_grow_pointers(void **old, size_t count, size_t new_count);
#endif
