/* What ngx_palloc.c wants from the level below, over the domain's region.
 *
 * The allocator itself is upstream byte for byte; this is the three calls it makes and the one
 * variable it reads. The allocator under it is the PostgreSQL port's pg_level0.c, included rather
 * than copied, so both ports stand on the same level 0 and a difference between their numbers is
 * never the level below. That file's own header asks for exactly this.
 *
 * ngx_memalign wants sixteen-byte alignment for a pool (NGX_POOL_ALIGNMENT), and pg_level0 aligns
 * every payload to sixteen already, so plain malloc satisfies it. If a caller ever asks for more,
 * the wrapper refuses rather than returning something that merely looks aligned.
 */
#include "ngx_shim.h"

void *malloc(size_t);
void free(void *);

/* What nginx has taken from the level below and not given back. The balance scenario asks this
   rather than the arena's free bytes, because the question is whether the PORT returns what
   upstream returns, not whether the level below fragments. */
unsigned long ngx_level0_live;

/* nginx reads this to decide what counts as a large allocation. The domain has no page table to
 * ask, and the number only has to be the same in both arms. */
uintptr_t ngx_pagesize = 4096;

void *ngx_alloc(size_t size, ngx_log_t *log) {
    (void) log;
    void *p = malloc(size);
    if (p != NULL) {
        ++ngx_level0_live;
    }
    return p;
}

void ngx_free_wrap(void *p) {
    if (p != NULL) {
        --ngx_level0_live;
    }
    free(p);
}

void *ngx_memalign(size_t alignment, size_t size, ngx_log_t *log) {
    (void) log;
    if (alignment > 16) {
        return NULL;            /* refused, not faked */
    }
    void *p = malloc(size);
    if (p != NULL) {
        ++ngx_level0_live;
    }
    return p;
}
