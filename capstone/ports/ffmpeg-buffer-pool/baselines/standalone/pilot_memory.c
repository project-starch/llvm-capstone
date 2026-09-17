/* Minimal dependency adapter for the isolated buffer.c pilot.
 * This bounded bump heap deliberately does not recycle backing allocations.
 * AVBufferPool's own free list still runs unchanged. This adapter establishes
 * functional bring-up only, not heap cost, Sublet protection, or reclamation.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "libavutil/mem.h"
#include "libavutil/log.h"

#define HEAP_BYTES (128 * 1024)
#define ALIGNMENT 64
static _Alignas(ALIGNMENT) uint8_t heap[HEAP_BYTES];
static size_t used;

void *av_malloc(size_t size)
{
    uint8_t *p;
    size_t rounded;
    if (size > HEAP_BYTES - ALIGNMENT)
        return NULL;
    rounded = (size + ALIGNMENT - 1) & ~(size_t)(ALIGNMENT - 1);
    if (used > HEAP_BYTES - ALIGNMENT - rounded)
        return NULL;
    p = heap + used;
    *(size_t *)p = size;
    used += ALIGNMENT + rounded;
    return p + ALIGNMENT;
}

void *av_mallocz(size_t size)
{
    void *p = av_malloc(size);
    if (p)
        memset(p, 0, size);
    return p;
}

void av_free(void *p)
{
    (void)p;
}

void av_freep(void *slot)
{
    void **p = slot;
    av_free(*p);
    *p = NULL;
}

void *av_realloc(void *p, size_t size)
{
    void *next;
    size_t previous;
    if (!size) {
        av_free(p);
        return NULL;
    }
    next = av_malloc(size);
    if (!next || !p)
        return next;
    previous = *(size_t *)((uint8_t *)p - ALIGNMENT);
    memcpy(next, p, previous < size ? previous : size);
    av_free(p);
    return next;
}

/* buffer.c uses logging only immediately before an unconditional abort. */
void av_log(void *avcl, int level, const char *fmt, ...)
{
    (void)avcl;
    (void)level;
    (void)fmt;
}

#ifdef FFPOOL_DOMAIN
_Noreturn void abort(void)
{
    __builtin_trap();
}
#endif
