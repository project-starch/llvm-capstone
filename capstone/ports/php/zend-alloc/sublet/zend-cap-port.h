/* The PHP allocator seam: geometry, the two mode bits, the per-block record, and the hooks
 * Zend/zend_alloc.c is patched to call.
 *
 * WHAT THE SEAM IS. PHP 5.0.0 is not a chunk allocator -- Zend/zend_mm.h unconditionally
 * undefines ZEND_MM, so ZEND_DO_MALLOC is plain malloc and the structure is flat: one malloc
 * per emalloc, a 48-byte zend_mem_header in front of the payload. The seam splits that one
 * block in two, so that NO capability spans the boundary:
 *
 *     B                 B+48                                   B+48+round16(REAL_SIZE(n))
 *     |<-- 48 header -->|<------------- payload ---------------->|
 *        never revoked     one region, revoked on every efree
 *
 * WHY THE SPLIT IS THE WHOLE DESIGN. zend_alloc.c computes the client pointer as p + 48 and
 * recovers the header as ptr - 48. Both directions need one capability spanning header and
 * payload -- and if that is what the caller holds, then _efree must revoke it, which kills the
 * header too, because they share one revocation node. PHP then cannot read p->size, write
 * p->cached, or park p in AG(cache), all of which happen AFTER the logical free; worse, every
 * pNext/pLast alias of that block held in a NEIGHBOUR's header dies with it, so the next
 * unlink walks a dead capability. The physical port paid that bill: it had to hoist
 * REMOVE_POINTER_FROM_LIST above the revoke and make it unconditional, a deviation from PHP's
 * own _efree that took four false CAUGHT verdicts to find and that still costs a NO-READING on
 * CRASH-057.
 *
 * With the split, the header outlives every payload issued from the block, so:
 *   - REMOVE_POINTER_FROM_LIST (active here: ZEND_MM is undefined, so the intrusive list is
 *     real) dereferences p->pLast and p->pNext, which are other blocks' HEADERS, all live;
 *   - PHP's own ordering already unlinks before ZEND_DO_FREE, so the list is repaired while
 *     everything it touches is still alive;
 *   - AG(cache) parks the header pointer and hands it back later, unmodified.
 * zend_alloc.c therefore keeps its control flow byte for byte, and the deviation disappears by
 * construction rather than by careful ordering.
 *
 * THE PATCH IS SIX DERIVATION SITES AND THREE MACROS, every hunk behind #ifdef ZEND_CAP_SUBLET.
 * With the define absent the file compiles to the same object as the pristine one, which the
 * build asserts with cmp. Nothing else moves: not REAL_SIZE, not cache_index, not the list
 * macros, not the cache policy.
 */
#ifndef ZEND_CAP_PORT_H
#define ZEND_CAP_PORT_H

#include <stddef.h>
#include <sublet/sublet.h>

/* ---- geometry ------------------------------------------------------------------------- */

/* sizeof(zend_mem_header) given this tree's configuration: ZEND_DEBUG 0 and MEMORY_LIMIT 0
 * (main/php_config.h:2373,2382) leave {pNext, pLast, size:31, cached:1}, which at
 * __alignof__(void *) == 16 is 48 bytes with MEM_HEADER_PADDING 0. zend-cap-blocks.c asserts
 * it against PHP's own sizeof at compile time; a tree configured otherwise must not build. */
#define ZEND_CAP_HDR    48UL

/* A capability. Sublet regions are "aligned and sized to whole capabilities": sublet_give_to
 * fills its region in 16-byte steps before init, and a region that is not a whole number of
 * capabilities long cannot be written through -- init then traps. So the payload is padded up
 * to a multiple of this, and the loan is 48 + that. */
#define ZEND_CAP_GRAIN  16UL
#define ZEND_CAP_ROUND(n) (((unsigned long)(n) + ZEND_CAP_GRAIN - 1UL) & ~(ZEND_CAP_GRAIN - 1UL))

/* ---- the two mode bits, which must not be one ----------------------------------------- */

/* pymalloc's pym_set_mode(0|1) conflates the bound with revocation; this suite must not, because
 * the two axes need different controls:
 *
 *   spatial pair   control REVOKE          fault REVOKE|EXACT    differs in the BOUND alone
 *   temporal pair  control 0               fault REVOKE          differs in REVOCATION alone
 *
 * The spatial control keeps revocation ON so that it also exercises the machinery the temporal
 * sanity arm exists to validate; the temporal pair holds the bound at REAL_SIZE so the only
 * variable is whether efree revokes. */
#define ZEND_CAP_MODE_EXACT   1u   /* 0 = bound to REAL_SIZE(n), stock PHP's rounding; 1 = to n */
#define ZEND_CAP_MODE_REVOKE  2u   /* 0 = one take per block, never revoked; 1 = give+take per retire */
#define ZEND_CAP_MODE_RETURN  4u   /* 0 = a freed block's loan is LEAKED; 1 = returned to the heap */

/* WHY THERE IS A THIRD BIT, found by measurement and not by design.
 *
 * The first temporal pair was mode 0 against mode 2, and BOTH arms caught CRASH-003 at the same
 * instruction -- Zend/zend_execute.c:669, a write through a zval pointer read out of freed
 * memory. The reason is that this platform's own malloc revokes: __capstone_sublet_free_linear
 * rotates the heap's authority, which is senior to the loan, so every alias to a block that
 * reaches ZEND_DO_FREE dies whatever this layer does. "No inner revocation" is therefore NOT
 * "stock PHP lifetimes" here, and a catch in such a pair cannot be attributed to the seam
 * rather than to mallocng underneath it.
 *
 * So the loan's return is its own variable, and the temporal axis has three columns:
 *
 *   mode 0  (stock)     nothing revoked anywhere: no retire, the loan is leaked, the record
 *                       stays indexed. A freed block stays readable exactly as it does on
 *                       stock PHP, so this is the honest-gap column -- what ASAN sees.
 *   mode 4  (platform)  the loan is returned, so mallocng revokes per BLOCK, but there is no
 *                       per-object retire. Catches a use-after-free on a block PHP actually
 *                       freed; MISSES one on a block AG(cache) absorbed, which is the 23 of
 *                       the 31 the size-class cache hides.
 *   mode 6  (seam)      both: per-object retire at every efree, cache-absorbed or not.
 *
 * mode 4 is what makes a mode-6 catch attributable to the SEAM: the two differ in the
 * per-object retire alone. Without it the instrument cannot separate its own contribution from
 * the platform's, which is the question to ask before the claim rather than after it.
 *
 * The leak in mode 0 is bounded by one trigger's allocation volume and is the price of a
 * faithful control: a control that revokes is not a control. */

/* ---- failure, which must never read as the workload failing --------------------------- */

/* PHP's own answer to a failed allocation is "FATAL: emalloc(): Unable to allocate" and
 * exit(1) (zend_alloc.c:189-195), which in a results table reads as the trigger failing rather
 * than as the instrument running out. Every way this layer can fail instead reports a distinct
 * marker and aborts, so a runner can classify it as infrastructure and refuse a verdict. */
#define ZEND_CAP_FAIL_LOAN      1  /* the heap would not lend a block */
#define ZEND_CAP_FAIL_RECORDS   2  /* the metadata heap is exhausted */
#define ZEND_CAP_FAIL_UNKNOWN   3  /* a pointer PHP freed or re-used that this layer never issued */
#define ZEND_CAP_FAIL_ALIGN     4  /* a loan whose base is not capability-aligned */
#define ZEND_CAP_FAIL_NODES     5  /* the revocation node table could not be reserved */
#define ZEND_CAP_FAIL_GEOMETRY  6  /* a split or a region that is not a whole number of grains */

void zend_cap_fail(int code, const char *what, unsigned long a, unsigned long b);

/* ---- the per-block record --------------------------------------------------------------
 *
 * A sublet_cap is a PLACE, not a value: a linear capability moves when it is copied, so an
 * allocator under Sublet keeps every capability it owns in a slot in memory. A record must
 * therefore never be memcpy'd, realloc'd or moved, which is why the metadata heap is a chain
 * of fixed slabs rather than a growable array.
 *
 * The three slots come first so that each sits at a 16-byte offset from a 16-aligned record. */
struct zend_cap_block {
    sublet_cap senior;      /* taken BEFORE any split: one revoke ends every descendant */
    sublet_cap header;      /* [B, B+48): the handle, once the header alias is out */
    sublet_cap payload;     /* [B+48, end): the region, or the handle while a payload is issued */
    void *alias;            /* the FULL payload alias -- what the ALLOCATOR copies through */
    void *header_alias;     /* what PHP holds as `p` */
    struct zend_cap_block *next;    /* hash chain */
    unsigned long key;      /* B + 48, the payload base: the one index key */
    unsigned long base;     /* B, the loan's base, for __capstone_sublet_free_linear */
    unsigned long bytes;    /* the loan's length, a whole number of grains */
    unsigned long size;     /* the caller's last requested size, for realloc's copy length */
    unsigned int issued;    /* a payload alias is currently out */
    unsigned int live;      /* this record is in use. NOT base == 0: the physical port used a
                             * zero-base sentinel in its slot table and it produced a false
                             * positive (zend_capstone_alloc.h:340-353). */
};

/* ---- the metadata heap and the index (zend-cap-backing.c) ----------------------------- */

struct zend_cap_block *zend_cap_record_new(void);
void zend_cap_record_free(struct zend_cap_block *r);
void zend_cap_index(struct zend_cap_block *r);
void zend_cap_unindex(struct zend_cap_block *r);
struct zend_cap_block *zend_cap_find(unsigned long key);
struct zend_cap_block *zend_cap_find_containing(unsigned long addr);
void zend_cap_backing_stats(unsigned long out[4]);   /* records live, peak, slabs, lookups */

/* ---- the seam (zend-cap-blocks.c), called from the patched zend_alloc.c ---------------- */

unsigned int zend_cap_mode(void);
void zend_cap_set_mode(unsigned int mode);

void *zend_cap_block_alloc(size_t bytes);              /* ZEND_DO_MALLOC  -> the HEADER alias */
void  zend_cap_block_free(void *p);                    /* ZEND_DO_FREE    (p = the header) */
void *zend_cap_block_realloc(void *p, size_t bytes);   /* ZEND_DO_REALLOC (p = the header) */

void *zend_cap_payload(void *p, size_t size);          /* ZEND_CAP_PAYLOAD(p, size) */
void *zend_cap_header(void *ptr);                      /* ZEND_CAP_HEADER(ptr) */
void  zend_cap_retire(void *p);                        /* ZEND_CAP_RETIRE(p) */

/* How much the seam did, for the gate that requires the fault arm to revoke strictly more than
 * the control. sublet_stats is static per translation unit (sublet.h:46-48), so heap.c's
 * counters do not see this layer's -- it needs its own. */
void zend_cap_stats(unsigned long out[8]);
/* Arm the atexit printer for the two PHP_VIRTUAL SEAM lines. Idempotent, and called
 * from the first block allocation: on real PHP nothing else is in a position to print
 * the revoke count, and the measurement's arm-ordering gate is built on it. */
void zend_cap_report_at_exit(void);
#define ZEND_CAP_STAT_BLOCKS   0
#define ZEND_CAP_STAT_FREES    1
#define ZEND_CAP_STAT_ISSUES   2
#define ZEND_CAP_STAT_RETIRES  3
#define ZEND_CAP_STAT_REVOKES  4
#define ZEND_CAP_STAT_SPLITS   5
#define ZEND_CAP_STAT_REALLOCS 6
#define ZEND_CAP_STAT_LIVE     7

#endif
