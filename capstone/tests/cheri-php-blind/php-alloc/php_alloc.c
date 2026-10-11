/* See php_alloc.h for provenance: this is PHP 5.0.0's non-ZEND_MM allocator
 * path transcribed, with ZEND_DEBUG=0 and MEMORY_LIMIT=0. */
#include "php_alloc.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* alloc_globals, the AG() macro's target. Single-threaded, ZTS undefined. */
static struct {
    zend_mem_header *head;
    zend_mem_header *cache[MAX_CACHED_MEMORY][MAX_CACHED_ENTRIES];
    unsigned int     cache_count[MAX_CACHED_MEMORY];
} ag;

#define AG(x) ag.x

/* zend_alloc.c:113 */
#define ADD_POINTER_TO_LIST(p)      \
    p->pNext = AG(head);            \
    if (AG(head)) {                 \
        AG(head)->pLast = p;        \
    }                               \
    AG(head) = p;                   \
    p->pLast = (zend_mem_header *) NULL;

/* zend_alloc.c:100 */
#define REMOVE_POINTER_FROM_LIST(p) \
    if (p == AG(head)) {            \
        AG(head) = p->pNext;        \
    } else {                        \
        p->pLast->pNext = p->pNext; \
    }                               \
    if (p->pNext) {                 \
        p->pNext->pLast = p->pLast; \
    }

#define DATA_OF(p) ((void *)((char *)(p) + sizeof(zend_mem_header) + MEM_HEADER_PADDING))
#define HDR_OF(ptr) ((zend_mem_header *)((char *)(ptr) - sizeof(zend_mem_header) - MEM_HEADER_PADDING))

void zend_alloc_startup(void)
{
    memset(&ag, 0, sizeof ag);
}

/* zend_alloc.c:139 _emalloc */
void *_emalloc(size_t size)
{
    zend_mem_header *p;
    unsigned int real_size  = (unsigned int) REAL_SIZE(size);
    unsigned int cache_index = real_size >> 3;

#if !ZEND_DISABLE_MEMORY_CACHE
    if ((cache_index < MAX_CACHED_MEMORY) && (AG(cache_count)[cache_index] > 0)) {
        /* CACHE HIT: the block is re-served WITHOUT touching malloc, so it is
         * never free()d and never leaves the live list. This is the path that
         * hides 23 of the corpus's use-after-frees from ASAN. */
        p = AG(cache)[cache_index][--AG(cache_count)[cache_index]];
        p->cached = 0;
        p->size = (unsigned int) size;
        return DATA_OF(p);
    } else {
#endif
        p = (zend_mem_header *) malloc(sizeof(zend_mem_header) + MEM_HEADER_PADDING + real_size);
#if !ZEND_DISABLE_MEMORY_CACHE
    }
#endif
    if (!p) {
        fprintf(stderr, "FATAL:  emalloc():  Unable to allocate %ld bytes\n", (long) size);
        exit(1);
    }
    p->cached = 0;
    ADD_POINTER_TO_LIST(p);
    p->size = (unsigned int) size;
    return DATA_OF(p);
}

/* zend_alloc.c:247 _efree */
void _efree(void *ptr)
{
    zend_mem_header *p = HDR_OF(ptr);
    unsigned int real_size  = (unsigned int) REAL_SIZE(p->size);
    unsigned int cache_index = real_size >> 3;

#if !ZEND_DISABLE_MEMORY_CACHE
    if ((cache_index < MAX_CACHED_MEMORY) && (AG(cache_count)[cache_index] < MAX_CACHED_ENTRIES)) {
        /* THE BLINDING PATH: pushed onto the size-class cache and returned.
         * No free(). The block stays malloc-live, stays on the live list, and
         * every pointer into it stays a valid in-bounds capability. */
        AG(cache)[cache_index][AG(cache_count)[cache_index]++] = p;
        p->cached = 1;
        return;
    }
#endif
    REMOVE_POINTER_FROM_LIST(p);
    free(p);
}

/* zend_alloc.c:345 _erealloc, non-ZEND_DEBUG path */
void *_erealloc(void *ptr, size_t size)
{
    zend_mem_header *p = HDR_OF(ptr);
    unsigned int real_size = (unsigned int) REAL_SIZE(size);

    REMOVE_POINTER_FROM_LIST(p);
    p = (zend_mem_header *) realloc(p, sizeof(zend_mem_header) + MEM_HEADER_PADDING + real_size);
    if (!p) {
        fprintf(stderr, "FATAL:  erealloc():  Unable to allocate %ld bytes\n", (long) size);
        exit(1);
    }
    ADD_POINTER_TO_LIST(p);
    p->size = (unsigned int) size;
    return DATA_OF(p);
}

char *_estrndup(const char *s, unsigned int length)
{
    char *p = (char *) _emalloc(length + 1);
    memcpy(p, s, length);
    p[length] = 0;
    return p;
}

/* zend_alloc.c:468 shutdown_memory_manager.
 * NOTE, and it matters for the measurement: the cache IS drained here and each
 * cached block IS free()d. So a block efree'd mid-request is malloc-live for
 * the whole request and is only released at shutdown -- i.e. strictly AFTER
 * every use site the corpus triggers exercise. */
void shutdown_memory_manager(void)
{
#if !ZEND_DISABLE_MEMORY_CACHE
    {
        unsigned int i, j;
        zend_mem_header *ptr;

        for (i = 1; i < MAX_CACHED_MEMORY; i++) {
            for (j = 0; j < AG(cache_count)[i]; j++) {
                ptr = AG(cache)[i][j];
                REMOVE_POINTER_FROM_LIST(ptr);
                free(ptr);
            }
            AG(cache_count)[i] = 0;
        }
    }
#endif
    {
        zend_mem_header *p = AG(head), *t;
        while (p) {
            t = p;
            p = p->pNext;
            free(t);
        }
        AG(head) = NULL;
    }
}

size_t php_alloc_real_size(size_t n)   { return (size_t) REAL_SIZE(n); }
size_t php_alloc_header_bytes(void)    { return sizeof(zend_mem_header) + MEM_HEADER_PADDING; }
int    php_alloc_cache_is_on(void)     { return !ZEND_DISABLE_MEMORY_CACHE; }
