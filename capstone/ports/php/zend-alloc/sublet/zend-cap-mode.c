/* The arm, the counters and the seam calls. Built only in seam mode (-DZEND_CAP_SUBLET); see
 * zend-cap-mode.h for what the seam is and why the three arms differ. */
#include "zend-cap-mode.h"
#include <malloc_capstone.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned int cap_mode, cap_mode_set;
static unsigned long st_blocks, st_frees, st_issues, st_retires, st_reallocs;

/* A capability grain. The block is rounded up to a multiple of this, at least a header plus one
 * grain, so that both the header and a payload bounded to REAL_SIZE(n) lie inside it -- including
 * emalloc(0), whose block would otherwise be the header alone and whose one-byte payload would
 * then lie outside it (CDERIVE refuses that with cause 29). */
#define ZEND_CAP_GRAIN  16UL
#define ZEND_CAP_ROUND(n) (((unsigned long)(n) + ZEND_CAP_GRAIN - 1UL) & ~(ZEND_CAP_GRAIN - 1UL))

/* Every way this layer can fail reports a distinct marker and aborts, so a runner classifies it
 * as infrastructure and refuses a verdict: PHP's own answer to a failed allocation is "FATAL:
 * emalloc(): Unable to allocate" and exit(1), which in a results table reads as the trigger
 * failing rather than as the instrument running out. */
#define ZEND_CAP_FAIL_MODE      1   /* no arm named */
#define ZEND_CAP_FAIL_NODES     2   /* the heap could not grow its node table for a child */
#define ZEND_CAP_FAIL_UNKNOWN   3   /* a pointer PHP frees or reallocs that is in no block of this heap */
#define ZEND_CAP_FAIL_INTERIOR  4   /* a pointer whose cursor is not at its payload's start */
#define ZEND_CAP_FAIL_GEOMETRY  5   /* a request or bound that does not fit the block (an overflowing size) */
static void cap_fail(int code, const char *what, unsigned long a, unsigned long b)
{
    fflush(stdout);
    fprintf(stderr, "ZEND_CAP_FAIL code=%d %s a=0x%lx b=0x%lx (blocks=%lu issues=%lu retires=%lu)\n",
            code, what, a, b, st_blocks, st_issues, st_retires);   /* run-realphp.py's SEAM_FAIL */
    fflush(stderr);
    abort();
}

/* THE ARM, NAMED OUT LOUD AND NEVER GUESSED. An absent PHP_CAP_MODE is a failure, not a
 * default: a run that silently chose an arm is a run whose result means nothing, and the
 * suite's own gate requires the two runs of a pair to report DIFFERENT mode lines. The one
 * exception is a build with ZEND_CAP_DEFAULT_MODE (the common application entry's image for
 * the generic runners), which takes that mode and SAYS so, and which run-realphp.py refuses. */
void zend_cap_set_mode(unsigned int mode)
{
    cap_mode = mode;
    cap_mode_set = 1;
    fprintf(stderr, "ZEND_CAP_MODE %u exact=%u revoke=%u\n", mode,
            (mode & ZEND_CAP_MODE_EXACT) ? 1u : 0u,
            (mode & ZEND_CAP_MODE_REVOKE) ? 1u : 0u);
    fflush(stderr);
}

unsigned int zend_cap_mode(void)
{
    if (!cap_mode_set) {
        const char *s = getenv("PHP_CAP_MODE");
        if (!s || !*s) {
#ifdef ZEND_CAP_DEFAULT_MODE
            fprintf(stderr, "ZEND_CAP_MODE_DEFAULTED %u\n", (unsigned int) ZEND_CAP_DEFAULT_MODE);
            fflush(stderr);
            zend_cap_set_mode((unsigned int) ZEND_CAP_DEFAULT_MODE);
            return cap_mode;
#else
            cap_fail(ZEND_CAP_FAIL_MODE, "PHP_CAP_MODE is unset: refusing to pick an arm", 0, 0);
#endif
        }
        zend_cap_set_mode((unsigned int) strtoul(s, (char **)0, 0));
    }
    return cap_mode;
}

/* ---- the exit report ---------------------------------------------------------------------
 * The runner reads the PHP_VIRTUAL SEAM line as its arm-ordering gate (a revoking arm must
 * revoke strictly more than its control). stderr, because by the time this runs PHP has shut
 * its own streams down; the runner reads both. Two routes, atexit and a destructor, because
 * the first attempt at this printed nothing: app/probe-atexit.c shows this platform running
 * both, and they are different mechanisms, so one failing does not silence the other. */
static int cap_reported;
static void cap_report_once(void)
{
    if (cap_reported) { return; }
    cap_reported = 1;
    fprintf(stderr, "PHP_VIRTUAL SEAM blocks=%lu frees=%lu issues=%lu retires=%lu revokes=%lu "
            "splits=0 reallocs=%lu live=%lu\n",
            st_blocks, st_frees, st_issues, st_retires, st_retires, st_reallocs,
            st_issues - st_retires);
    fflush(stderr);
}
__attribute__((destructor)) static void cap_report_dtor(void) { cap_report_once(); }

static void cap_arm_report(void)
{
    static int armed;
    if (armed) { return; }
    armed = 1;
    fprintf(stderr, "PHP_VIRTUAL SEAM armed rc=%d\n", atexit(cap_report_once));
    fflush(stderr);
}

void zend_cap_stats(unsigned long out[8])
{
    out[ZEND_CAP_STAT_BLOCKS]   = st_blocks;
    out[ZEND_CAP_STAT_FREES]    = st_frees;
    out[ZEND_CAP_STAT_ISSUES]   = st_issues;
    out[ZEND_CAP_STAT_RETIRES]  = st_retires;
    out[ZEND_CAP_STAT_REVOKES]  = st_retires;
    out[ZEND_CAP_STAT_SPLITS]   = 0;
    out[ZEND_CAP_STAT_REALLOCS] = st_reallocs;
    out[ZEND_CAP_STAT_LIVE]     = st_issues - st_retires;
}

/* ---- the seam calls ------------------------------------------------------------------------ */

/* ZEND_DO_MALLOC(bytes). A 16-grained block, at least a header plus a grain. */
void *zend_cap_block(size_t bytes)
{
    unsigned long want;
    void *p;
    cap_arm_report();
    if (bytes > ~0UL - ZEND_CAP_GRAIN) { return (void *)0; }   /* the rounding would wrap */
    want = ZEND_CAP_ROUND(bytes);
    if (want < 48UL + ZEND_CAP_GRAIN) { want = 48UL + ZEND_CAP_GRAIN; }
    p = malloc((size_t)want);
    if (p) { ++st_blocks; }
    return p;
}

/* The bound handed to the caller: stock PHP's REAL_SIZE rounding, or the request itself; never
 * zero, because neither CDERIVE nor a narrowed copy may be empty. */
static size_t cap_bound(unsigned int mode, size_t size)
{
    size_t b = (mode & ZEND_CAP_MODE_EXACT) ? size : ((size + 7UL) & ~7UL);
    return b ? b : 1UL;
}

/* ZEND_CAP_PAYLOAD(p, size). The caller's pointer over [p + hdr, p + hdr + bound): in the
 * revoking arms a child of the block with a lifetime of its own, otherwise a narrowed copy of the
 * block (the same bytes, the block's lifetime). The bound is the same in both, which is what
 * makes the temporal pair differ in revocation alone. A child per issue, fresh or from
 * AG(cache): the cache path matters as much as the fresh one, because a block handed back from
 * the cache must be re-issued under a FRESH lifetime, or the previous tenant's alias is still
 * live over the same bytes. The pointer's base is the payload, so the caller cannot reach PHP's
 * header: an under-read below the object is a bounds fault. */
void *zend_cap_issue(void *p, size_t hdr, size_t size)
{
    unsigned int mode = zend_cap_mode();
    size_t bound = cap_bound(mode, size);
    unsigned long start, extent;
    void *child;
    if (!p) { return (void *)0; }
    /* The request and the bound must fit the block. PHP's REAL_SIZE wraps on a request near
     * SIZE_MAX and its size field is 31 bits, so a hostile request can reach here with a block
     * that is the minimum and a size that is not; refused by name rather than derived past the
     * block (cause 29) or copied past it later by reblock. */
    extent = __builtin_capstone_cap_get_end(p) - __builtin_capstone_cap_get_base(p);
    if (hdr > extent || size > extent - hdr || bound > extent - hdr) {
        cap_fail(ZEND_CAP_FAIL_GEOMETRY, "request does not fit the block", size, extent);
    }
    ++st_issues;
    if (mode & ZEND_CAP_MODE_REVOKE) {
        child = __capstone_malloc_derive(p, hdr, bound);
        if (!child) { cap_fail(ZEND_CAP_FAIL_NODES, "the heap could not grow its node table for a child", bound, 0); }
        return child;
    }
    start = __builtin_capstone_cap_get_base(p) + hdr;
    return __builtin_capstone_cap_shrink((char *)p + hdr, start, start + bound);
}

/* The block that holds ptr, by the libc's slot record: ptr's base minus the header, the same
 * offset issue derived at. A pointer whose cursor is not at its payload's start is refused by
 * name (PHP freeing an interior pointer), and one that is in no block of this heap too (PHP
 * freeing what it did not emalloc): both are the instrument refusing, never a reading. */
static void *cap_block_of(void *ptr, size_t hdr, const char *who)
{
    unsigned long cur = __builtin_capstone_cap_get_cursor(ptr);
    unsigned long base = __builtin_capstone_cap_get_base(ptr);
    void *block;
    if (cur != base) { cap_fail(ZEND_CAP_FAIL_INTERIOR, who, cur, base); }
    block = __capstone_malloc_object(ptr, hdr);
    if (!block) { cap_fail(ZEND_CAP_FAIL_UNKNOWN, who, cur, 0); }
    return block;
}

/* ZEND_CAP_HEADER(ptr): _erealloc's and _mem_block_check's derivation. Nothing ends here. */
void *zend_cap_header(void *ptr, size_t hdr)
{
    if (!ptr) { return (void *)0; }
    return cap_block_of(ptr, hdr, "header of a pointer this heap did not issue, or not at its start");
}

/* ZEND_CAP_RECLAIM(ptr): _efree's derivation. In the revoking arms the caller's child ends in
 * the same call, inside the libc, through the block's own authority: a grandchild the caller
 * derived is cause 29 at cap_malloc_reclaim, a child already ended (a double efree) cause 25. */
void *zend_cap_reclaim(void *ptr, size_t hdr)
{
    void *block;
    if (!ptr) { return (void *)0; }
    block = cap_block_of(ptr, hdr, "free of a pointer this heap did not issue, or not at its start");
    if (zend_cap_mode() & ZEND_CAP_MODE_REVOKE) {
        __capstone_malloc_reclaim(ptr, hdr);
        ++st_retires;
    }
    return block;
}

/* ZEND_DO_FREE(p). The stock arm leaks the block on purpose: mallocng's own free revokes, and a
 * control that revokes is not a control. */
void zend_cap_release(void *p)
{
    ++st_frees;
    if (zend_cap_mode() & ZEND_CAP_MODE_RETURN) { free(p); }
}

/* ZEND_DO_REALLOC(p, bytes). A FRESH BLOCK ALWAYS, never in place: PHP's _erealloc is entitled
 * to a move, and moving is what the temporal axis wants -- the old payload's aliases die with the
 * old block when it is returned to the heap, and stay alive when it is leaked (mode 0). The copy
 * is `oldbytes` (the header plus the old request, which the macro reads from p->size before
 * _erealloc overwrites it), capped by the new block. On failure nothing has changed: the old
 * block, and the caller's pointer into it, are intact for _erealloc's ADD_POINTER_TO_LIST(orig). */
void *zend_cap_reblock(void *p, size_t bytes, size_t oldbytes)
{
    unsigned int mode = zend_cap_mode();
    void *fresh;
    if (!p) { return zend_cap_block(bytes); }
    unsigned long extent = __builtin_capstone_cap_get_end(p) - __builtin_capstone_cap_get_base(p);
    fresh = zend_cap_block(bytes);
    if (!fresh) { return (void *)0; }
    if (oldbytes > extent) { oldbytes = extent; }   /* never past the old block, whatever p->size says */
    memcpy(fresh, p, oldbytes < bytes ? oldbytes : bytes);
    ++st_reallocs;
    if (mode & ZEND_CAP_MODE_REVOKE) { ++st_retires; }   /* the old child ends with its block */
    zend_cap_release(p);
    return fresh;
}
