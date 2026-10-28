/* The arm, the counters and the three seam calls. Built only in seam mode (-DZEND_CAP_SUBLET);
 * see zend-cap-mode.h for what the seam is and why the three arms differ. */
#include "zend-cap-mode.h"
#include <malloc_capstone.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned int cap_mode, cap_mode_set;
static unsigned long st_blocks, st_frees, st_issues, st_retires;

/* Every way this layer can fail reports a distinct marker and aborts, so a runner classifies it
 * as infrastructure and refuses a verdict: PHP's own answer to a failed allocation is "FATAL:
 * emalloc(): Unable to allocate" and exit(1), which in a results table reads as the trigger
 * failing rather than as the instrument running out. */
#define ZEND_CAP_FAIL_MODE   1   /* no arm named */
#define ZEND_CAP_FAIL_NODES  2   /* the heap could not grow its node table for a child */
static void cap_fail(int code, const char *what)
{
    fflush(stdout);
    fprintf(stderr, "ZEND_CAP_FAIL code=%d %s\n", code, what);   /* run-realphp.py's SEAM_FAIL */
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
            cap_fail(ZEND_CAP_FAIL_MODE, "PHP_CAP_MODE is unset: refusing to pick an arm");
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
            "splits=0 reallocs=0 live=%lu\n",
            st_blocks, st_frees, st_issues, st_retires, st_retires, st_issues - st_retires);
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
    out[ZEND_CAP_STAT_REALLOCS] = 0;
    out[ZEND_CAP_STAT_LIVE]     = st_issues - st_retires;
}

/* ---- the three seam calls ----------------------------------------------------------------- */

/* ZEND_CAP_PAYLOAD(p, size). The bound is stock PHP's REAL_SIZE rounding, or the request in the
 * exact arm; never zero, because CDERIVE refuses an empty child. A child per issue, fresh or
 * from AG(cache): the cache path matters as much as the fresh one, because a block handed back
 * from the cache must be re-issued under a FRESH lifetime, or the previous tenant's alias is
 * still live over the same bytes. The child's base is the payload, so the caller cannot reach
 * PHP's header: an under-read below the object is a bounds fault here. */
void *zend_cap_issue(void *p, size_t hdr, size_t size)
{
    unsigned int mode = zend_cap_mode();
    size_t bound = (mode & ZEND_CAP_MODE_EXACT) ? size : ((size + 7UL) & ~7UL);   /* REAL_SIZE */
    void *child;
    cap_arm_report();
    if (!p) { return (void *)0; }
    if (!(mode & ZEND_CAP_MODE_REVOKE)) { return (char *)p + hdr; }
    child = __capstone_malloc_derive(p, hdr, bound ? bound : 1UL);
    if (!child) { cap_fail(ZEND_CAP_FAIL_NODES, "the heap could not grow its node table for a child"); }
    ++st_issues;
    return child;
}

/* ZEND_CAP_HEADER(ptr). Ends the caller's child and returns the block it lives in, cursor at
 * the header, through the libc's slot record. The identity check is CREVOKE's direct-parent
 * rule inside the libc: a grandchild the caller derived is cause 29, a child already ended
 * (a double efree) cause 25, both at cap_malloc_reclaim -- in the allocator, which the judge
 * attributes through the fault record's return address. */
void *zend_cap_reclaim(void *ptr, size_t hdr)
{
    void *block;
    if (!ptr) { return (void *)0; }
    if (!(zend_cap_mode() & ZEND_CAP_MODE_REVOKE)) { return (char *)ptr - hdr; }
    block = __capstone_malloc_reclaim(ptr, hdr);
    ++st_retires;
    return block;
}

/* ZEND_DO_FREE(p). The stock arm leaks the block on purpose: mallocng's own free revokes, and a
 * control that revokes is not a control. */
void zend_cap_release(void *p)
{
    ++st_frees;
    if (zend_cap_mode() & ZEND_CAP_MODE_RETURN) { free(p); }
}

/* ZEND_DO_MALLOC(size): plain malloc, counted, so the SEAM line's blocks= is the number of
 * blocks PHP took from the heap (the cache's hand-backs are issues without a block). */
void *zend_cap_block(size_t size)
{
    void *p = malloc(size);
    if (p) { ++st_blocks; }
    return p;
}
