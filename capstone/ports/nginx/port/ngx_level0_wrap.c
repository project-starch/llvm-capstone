/* What ngx_palloc.c wants from the level below, over the domain's region.
 *
 * The allocator itself is upstream byte for byte; this is the three calls it makes and the one
 * variable it reads. The allocator under it is the PostgreSQL port's pg_level0.c, included rather
 * than copied, so both ports stand on the same level 0 and a difference between their numbers is
 * never the level below. That file's own header asks for exactly this.
 *
 * ngx_memalign is asked for more than sixteen by real nginx, ngx_pagesize by the radix tree and
 * the direct IO alignment by the output chain, and pg_level0 aligns only to sixteen. So every
 * allocation here reserves one capability in front of what it returns and parks the pointer
 * malloc gave in it, which lets the returned address be moved up to any boundary and still be
 * freed exactly. Uniform on purpose: a free that had to work out which kind of allocation it was
 * looking at would be a place to get it wrong.
 *
 * The protected arm needs the same answer and gets it differently, by carving the padding off the
 * block and letting it go until the revocation takes it back. Both arms answer, so a difference
 * between their numbers is still never the level below.
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

/* One capability of header, holding what malloc returned, so any alignment can be answered and
   still freed exactly. */
#define NGX_L0_HDR  sizeof(void *)

static void *ngx_l0_alloc(size_t alignment, size_t size) {
    unsigned char *raw = malloc(size + alignment + NGX_L0_HDR);
    if (raw == NULL) {
        return NULL;
    }

    unsigned char *p = raw + NGX_L0_HDR;
    unsigned long off = (unsigned long) (void *) p & (unsigned long) (alignment - 1);
    if (off != 0) {
        p += alignment - off;       /* pointer arithmetic, never a cast back from an integer */
    }

    ((void **) p)[-1] = raw;
    ++ngx_level0_live;
    return p;
}

void *ngx_alloc(size_t size, ngx_log_t *log) {
    (void) log;
    return ngx_l0_alloc(NGX_L0_HDR, size);
}

void ngx_free_wrap(void *p) {
    if (p == NULL) {
        return;
    }
    --ngx_level0_live;
    free(((void **) p)[-1]);
}

void *ngx_memalign(size_t alignment, size_t size, ngx_log_t *log) {
    (void) log;
    if (alignment < NGX_L0_HDR) {
        alignment = NGX_L0_HDR;
    }
    return ngx_l0_alloc(alignment, size);
}
