/* The PHP allocator seam: the arm, the counters, and the calls the patched zend_alloc.c makes.
 * There is no state per block: the block's authority lives in the libc's slot record, and
 * __capstone_malloc_{derive,object,reclaim} are the libc using it (runtime/virtual/heap.c). The
 * previous seam rebuilt that record for itself -- a hash index, a record heap and six failure
 * codes, 886 lines -- which the measurement of 2026-10-27 (520 cells, three runs, identical) was
 * taken on; this is the same shape with the index gone.
 *
 * WHAT THE SEAM IS. PHP 5.0.0 is not a chunk allocator -- Zend/zend_mm.h unconditionally
 * undefines ZEND_MM -- so ZEND_DO_MALLOC is plain malloc and the structure is flat: one malloc
 * per emalloc, a 48-byte zend_mem_header in front of the payload. The block is rounded up to a
 * 16-byte grain, at least 64 bytes, as the previous seam did, so that a payload bounded to
 * REAL_SIZE(n) -- or to n -- always lies inside it, emalloc(0) included. The caller's pointer is
 * bounded to the payload in EVERY arm, so no capability the caller holds spans the header:
 *
 *     B                 B+48                                   B+48+bound(n)
 *     |<-- 48 header -->|<------------- payload ---------------->|
 *        PHP's own          the caller's pointer: a child with a lifetime of its own in the
 *                           revoking arms, a narrowed copy of the block otherwise
 *
 * At _efree the caller's child ends through the libc's reclaim, which hands back the block at
 * B, so everything below the first line of _efree -- p->size, p->cached, the AG(cache) park,
 * REMOVE_POINTER_FROM_LIST -- runs on live memory exactly as upstream. _erealloc recovers the
 * block WITHOUT ending anything (zend_cap_header), allocates a fresh block, copies, and only
 * then releases the old one: a failed realloc leaves the caller's pointer alive, as upstream's
 * ADD_POINTER_TO_LIST(orig) requires. A block handed back from the cache is re-issued as a FRESH
 * child, which is what leaves the previous tenant's alias dead over live bytes. */
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
 * The temporal pair holds the bound at REAL_SIZE in both arms (zend_cap_issue narrows the
 * control's pointer exactly as it bounds the fault arm's child), so the only variable is whether
 * efree ends a lifetime. The third bit exists because this platform's own malloc revokes: the
 * heap's free rotates the block's authority, so every alias to a block that reaches ZEND_DO_FREE
 * dies whatever this layer does. "No inner revocation" is therefore NOT "stock PHP lifetimes"
 * here, and the temporal axis has three columns:
 *
 *   mode 0  (stock)     nothing revoked anywhere: no child, and a block is never returned to the
 *                       heap, realloc included (a fresh block, a copy, the old one kept). A freed
 *                       block stays readable exactly as on stock PHP -- what ASAN sees.
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

/* The seam calls, from the patched zend_alloc.c's macros. `hdr` is sizeof(zend_mem_header) plus
 * its padding, passed by the macros so this file never includes PHP's headers. */
void *zend_cap_block(size_t bytes);                            /* ZEND_DO_MALLOC: a 16-grained block, counted */
void *zend_cap_issue(void *p, size_t hdr, size_t size);        /* ZEND_CAP_PAYLOAD: the caller's pointer over the payload */
void *zend_cap_header(void *ptr, size_t hdr);                  /* ZEND_CAP_HEADER: the block holding ptr; nothing ends */
void *zend_cap_reclaim(void *ptr, size_t hdr);                 /* ZEND_CAP_RECLAIM: the block, after ending ptr's lifetime */
void *zend_cap_reblock(void *p, size_t bytes, size_t oldbytes); /* ZEND_DO_REALLOC: a fresh block, the copy, the old released */
void  zend_cap_release(void *p);                               /* ZEND_DO_FREE: free(p), or nothing without RETURN */

/* How much the seam did, for the gate that requires a revoking arm to revoke strictly more than
 * its control. Printed at exit as the PHP_VIRTUAL SEAM line. */
void zend_cap_stats(unsigned long out[8]);
#define ZEND_CAP_STAT_BLOCKS   0   /* blocks taken from the heap */
#define ZEND_CAP_STAT_FREES    1   /* ZEND_DO_FREE calls and realloc's old blocks, returned or leaked */
#define ZEND_CAP_STAT_ISSUES   2   /* payloads issued, fresh or from the cache, every arm */
#define ZEND_CAP_STAT_RETIRES  3   /* lifetimes ended by this layer: reclaims, and realloc's old blocks in the revoking arms */
#define ZEND_CAP_STAT_REVOKES  4   /* == RETIRES: one CREVOKE per reclaim, one free per old block */
#define ZEND_CAP_STAT_SPLITS   5   /* always 0: there is no split any more */
#define ZEND_CAP_STAT_REALLOCS 6   /* reblocks */
#define ZEND_CAP_STAT_LIVE     7   /* issues - retires */

#endif
