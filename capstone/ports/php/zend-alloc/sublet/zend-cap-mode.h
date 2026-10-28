/* The PHP allocator seam: the arm, the counters, and the three calls the patched zend_alloc.c
 * makes. There is no state per block: the block's authority lives in the libc's slot record,
 * and __capstone_malloc_derive / __capstone_malloc_reclaim are the libc using it
 * (runtime/virtual/heap.c). The previous seam rebuilt that record for itself -- a hash index,
 * a record heap and six failure codes, 886 lines -- which the measurement of 2026-10-27
 * (520 cells, three runs, identical) was taken on; this is the same shape with the index gone.
 *
 * WHAT THE SEAM IS. PHP 5.0.0 is not a chunk allocator -- Zend/zend_mm.h unconditionally
 * undefines ZEND_MM -- so ZEND_DO_MALLOC is plain malloc and the structure is flat: one malloc
 * per emalloc, a 48-byte zend_mem_header in front of the payload. The caller's pointer becomes a
 * CDERIVE child of that malloc'd object, bounded to the payload, so no capability spans the
 * header and the payload:
 *
 *     B                 B+48                                   B+48+bound(n)
 *     |<-- 48 header -->|<------------- payload ---------------->|
 *        PHP's own          the caller's child, ended at every efree
 *
 * At _efree the libc ends that child and hands back the object at B, so everything below the
 * first line of _efree -- p->size, p->cached, the AG(cache) park, REMOVE_POINTER_FROM_LIST --
 * runs on live memory exactly as upstream. A block handed back from the cache is re-issued as a
 * FRESH child, which is what leaves the previous tenant's alias dead over live bytes. */
#ifndef ZEND_CAP_MODE_H
#define ZEND_CAP_MODE_H

#include <stddef.h>

/* ---- the two mode bits, which must not be one, and the third found by measurement ----------
 *
 * pymalloc's pym_set_mode(0|1) conflates the bound with revocation; this suite must not, because
 * the two axes need different controls:
 *
 *   spatial pair   control REVOKE          fault REVOKE|EXACT    differs in the BOUND alone
 *   temporal pair  control 0               fault REVOKE          differs in REVOCATION alone
 *
 * The third bit exists because this platform's own malloc revokes: the heap's free rotates the
 * block's authority, so every alias to a block that reaches ZEND_DO_FREE dies whatever this layer
 * does. "No inner revocation" is therefore NOT "stock PHP lifetimes" here, and the temporal axis
 * has three columns:
 *
 *   mode 0  (stock)     nothing revoked anywhere: no child, and the block is never returned to the
 *                       heap. A freed block stays readable exactly as on stock PHP -- what ASAN sees.
 *   mode 4  (libc)      the block is returned, so mallocng revokes per BLOCK, but no child per
 *                       object. Catches a use-after-free on a block PHP actually freed; MISSES one
 *                       on a block AG(cache) absorbed.
 *   mode 6  (seam)      both: a child per issue, ended at every efree, cache-absorbed or not.
 *   mode 7  (exact)     as 6, bounded to the request instead of REAL_SIZE(n).
 *
 * The leak in mode 0 is bounded by one trigger's allocation volume and is the price of a faithful
 * control: a control that revokes is not a control. */
#define ZEND_CAP_MODE_EXACT   1u   /* 0 = bound to REAL_SIZE(n), stock PHP's rounding; 1 = to n */
#define ZEND_CAP_MODE_REVOKE  2u   /* 0 = no child per object; 1 = derive per issue, reclaim per efree */
#define ZEND_CAP_MODE_RETURN  4u   /* 0 = a freed block is LEAKED; 1 = returned to the heap */

/* THE ARM, NAMED OUT LOUD AND NEVER GUESSED: an unset PHP_CAP_MODE is a refusal, not a default.
 * zend_cap_mode() reads it once and prints the ZEND_CAP_MODE line the runner keys on. */
unsigned int zend_cap_mode(void);
void zend_cap_set_mode(unsigned int mode);

/* The seam calls, from the patched zend_alloc.c's macros. */
void *zend_cap_block(size_t size);                       /* ZEND_DO_MALLOC: malloc, counted */
void *zend_cap_issue(void *p, size_t hdr, size_t size);  /* ZEND_CAP_PAYLOAD: p + hdr, or a child of p */
void *zend_cap_reclaim(void *ptr, size_t hdr);           /* ZEND_CAP_HEADER: ptr - hdr, or the block, after ending ptr */
void  zend_cap_release(void *p);                         /* ZEND_DO_FREE: free(p), or nothing in the stock arm */

/* How much the seam did, for the gate that requires a revoking arm to revoke strictly more than
 * its control. Printed at exit as the PHP_VIRTUAL SEAM line. */
void zend_cap_stats(unsigned long out[8]);
#define ZEND_CAP_STAT_BLOCKS   0   /* objects issued fresh (a derive on a block not from the cache) */
#define ZEND_CAP_STAT_FREES    1   /* ZEND_DO_FREE calls, returned or leaked */
#define ZEND_CAP_STAT_ISSUES   2   /* children derived, fresh or from the cache */
#define ZEND_CAP_STAT_RETIRES  3   /* children reclaimed */
#define ZEND_CAP_STAT_REVOKES  4   /* == RETIRES: one CREVOKE per reclaim */
#define ZEND_CAP_STAT_SPLITS   5   /* always 0: there is no split any more */
#define ZEND_CAP_STAT_REALLOCS 6   /* always 0: realloc is reclaim then issue, counted there */
#define ZEND_CAP_STAT_LIVE     7   /* issues - retires */

#endif
