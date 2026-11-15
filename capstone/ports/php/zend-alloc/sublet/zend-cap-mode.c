/* The arm, the counters, the block table and the nested layer calls. Built only in nested mode
 * (-DZEND_CAP_SUBLET); see zend-cap-mode.h for what the nested layer is and why the arms differ. */
#include "zend-cap-mode.h"
#include <heap_sublet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef ZEND_CAP_LAYOUT_TRACE
/* Test builds only (docs/plans/2026-11-15-php-port-layout-fix.md): counts the malloc, realloc and
 * free calls below, as the stock build counts ZEND_DO_MALLOC, ZEND_DO_REALLOC and ZEND_DO_FREE. */
# include "zend-layout-trace.h"
#endif

static unsigned int cap_mode, cap_mode_set;
static unsigned long st_blocks, st_frees, st_issues, st_retires, st_reallocs;
static unsigned long cap_table_used, cap_table_peak;   /* the block table's fill, below */

/* Every way this layer can fail reports a distinct marker and aborts, so a runner classifies it
 * as infrastructure and refuses a verdict: PHP's own answer to a failed allocation is "FATAL:
 * emalloc(): Unable to allocate" and exit(1), which in a results table reads as the trigger
 * failing rather than as the instrument running out. */
#define ZEND_CAP_FAIL_MODE      1   /* no arm named */
#define ZEND_CAP_FAIL_UNKNOWN   3   /* a pointer PHP frees or reallocs that is in no block of this heap */
#define ZEND_CAP_FAIL_INTERIOR  4   /* a pointer whose cursor is not at its payload's start */
#define ZEND_CAP_FAIL_GEOMETRY  5   /* a request or bound that does not fit the block (an overflowing size) */
#define ZEND_CAP_FAIL_TABLE     6   /* the block table is full */
static void cap_fail(int code, const char *what, unsigned long a, unsigned long b)
{
    fflush(stdout);
    fprintf(stderr, "ZEND_CAP_FAIL code=%d %s a=0x%lx b=0x%lx (blocks=%lu issues=%lu retires=%lu)\n",
            code, what, a, b, st_blocks, st_issues, st_retires);   /* run-realphp.py's NESTED_FAIL */
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
 * The runner reads the PHP_VIRTUAL NESTED line as its arm-ordering gate (a revoking arm must
 * revoke strictly more than its control). stderr, because by the time this runs PHP has shut
 * its own streams down. One route, a destructor: app/probe-atexit.c showed this platform runs
 * destructors at exit, and the gate refuses any run whose line is missing, so a silent route
 * cannot pass for a working one. (An atexit route beside it, with an "armed" line nothing read,
 * was dropped 2026-10-28.) */
__attribute__((destructor)) static void cap_report(void)
{
    fprintf(stderr, "PHP_VIRTUAL NESTED blocks=%lu frees=%lu issues=%lu retires=%lu revokes=%lu "
            "splits=0 reallocs=%lu live=%lu table_peak=%lu\n",
            st_blocks, st_frees, st_issues, st_retires, st_retires, st_reallocs,
            st_issues - st_retires, cap_table_peak);
    fflush(stderr);
}

void zend_cap_stats(unsigned long out[8])
{
    out[0] = st_blocks;  out[1] = st_frees;   out[2] = st_issues;   out[3] = st_retires;
    out[4] = st_retires; out[5] = 0;          out[6] = st_reallocs; out[7] = st_issues - st_retires;
}

/* ---- the block table ------------------------------------------------------------------------
 *
 * The caller's child covers its payload alone, so the layer cannot reach a block's header from it.
 * This table is where the layer keeps its authority instead: one slot per block, holding the
 * reference malloc returned (the cursor at PHP's header), keyed by the header address. It lives in
 * this file's .bss, outside every client bound, and never allocates, so malloc sees exactly PHP's
 * requests. Linear probing with backward-shift deletion keeps it free of tombstones however many
 * blocks come and go. 2^18 slots, filled to three quarters at most: the largest corpus case keeps
 * about 24,000 blocks at once on the stock arm, which leaks every block. A full table fails by
 * name rather than degrading; the exit line reports the peak (table_peak). */
#define CAP_TABLE_BITS  18
#define CAP_TABLE_SLOTS (1UL << CAP_TABLE_BITS)
#define CAP_TABLE_LIMIT (CAP_TABLE_SLOTS / 4UL * 3UL)
static void *cap_table[CAP_TABLE_SLOTS];

static unsigned long cap_home(unsigned long key)
{
    return ((key >> 4) * 0x9E3779B97F4A7C15UL) >> (64 - CAP_TABLE_BITS);
}

/* The slot that holds `key`, or the empty slot where it would go. */
static unsigned long cap_table_slot(unsigned long key)
{
    unsigned long i = cap_home(key);
    while (cap_table[i] && __builtin_capstone_cap_get_cursor(cap_table[i]) != key) {
        i = (i + 1UL) & (CAP_TABLE_SLOTS - 1UL);
    }
    return i;
}

static void *cap_table_find(unsigned long key)
{
    return cap_table[cap_table_slot(key)];
}

/* Enter a block, or replace the entry at its address (realloc in place: a new lifetime). */
static void cap_table_put(void *block)
{
    unsigned long i = cap_table_slot(__builtin_capstone_cap_get_cursor(block));
    if (!cap_table[i]) {
        if (cap_table_used >= CAP_TABLE_LIMIT) {
            cap_fail(ZEND_CAP_FAIL_TABLE, "the block table is full", cap_table_used, CAP_TABLE_SLOTS);
        }
        if (++cap_table_used > cap_table_peak) { cap_table_peak = cap_table_used; }
    }
    cap_table[i] = block;
}

/* Remove the block at `key`; 0 if there is none. Each entry after the hole moves back into it
 * unless its home lies cyclically in (hole, entry], which keeps every probe sequence unbroken. */
static int cap_table_drop(unsigned long key)
{
    unsigned long i = cap_table_slot(key), j = i, k;
    if (!cap_table[i]) { return 0; }
    for (;;) {
        j = (j + 1UL) & (CAP_TABLE_SLOTS - 1UL);
        if (!cap_table[j]) { break; }
        k = cap_home(__builtin_capstone_cap_get_cursor(cap_table[j]));
        if (i <= j ? (i < k && k <= j) : (i < k || k <= j)) { continue; }
        cap_table[i] = cap_table[j];
        i = j;
    }
    cap_table[i] = (void *)0;
    --cap_table_used;
    return 1;
}

/* ---- the block geometry ----------------------------------------------------------------------
 *
 * `bytes` is PHP's own request: the header plus REAL_SIZE(n). The widest child any arm issues
 * is B = heap_round(REAL_SIZE(n)), at least 1; mode 7's bound, heap_round(n), is never wider.
 * Malloc's reference covers heap_round(request) bytes from a start that mallocng aligns to
 * that length's granule, at least 16 bytes.
 *
 *   B below 16 KiB   B's granule is at most 16 bytes, so the payload at header + 48 carries
 *                    exact bounds: the request is PHP's own. Only emalloc(0), whose 48 bytes
 *                    leave no room for a one-byte child, asks for 49 (the same mallocng class).
 *   B of 16 KiB up   B's granule is 32 bytes or more, and no payload 48 bytes past an aligned
 *                    start can have it. The smallest slack in front of the header aligns the
 *                    payload: 16 bytes up to a 64-byte granule, granule - 48 beyond. */
struct cap_geometry { size_t request, slack; };

static int cap_geometry(size_t bytes, struct cap_geometry *g)
{
    size_t real, bound, granule, offset;
    /* Match malloc's ptrdiff-sized object limit before representable rounding. */
    if (bytes < ZEND_CAP_HEADER_BYTES || bytes > (size_t)-1 / 2) { return 0; }
    real = bytes - ZEND_CAP_HEADER_BYTES;
    bound = heap_round(real ? real : 1);
    granule = heap_granule(bound);
    if (granule <= 16UL) {
        g->slack = 0;
        g->request = bytes;
        if (ZEND_CAP_HEADER_BYTES + bound > heap_round(bytes)) {
            g->request = ZEND_CAP_HEADER_BYTES + bound;
        }
        return 1;
    }
    offset = granule < 64UL ? 64UL : granule;
    g->slack = offset - ZEND_CAP_HEADER_BYTES;
    g->request = offset + bound;
    return 1;
}

/* The reference malloc returned for `block`: the same lifetime and bounds, the cursor at the base. */
static void *cap_malloc_ref(void *block)
{
    return (char *)block - (__builtin_capstone_cap_get_cursor(block) -
                            __builtin_capstone_cap_get_base(block));
}

/* ---- the nested layer calls ------------------------------------------------------------------------ */

/* ZEND_DO_MALLOC(bytes). PHP's request to malloc, entered in the block table. */
void *zend_cap_block(size_t bytes)
{
    struct cap_geometry g;
    void *p, *block;
    if (!cap_geometry(bytes, &g)) { return (void *)0; }
    p = malloc(g.request);
    if (!p) { return (void *)0; }
    block = (char *)p + g.slack;
    cap_table_put(block);
    ++st_blocks;
    return block;
}

/* The bound handed to the caller: stock PHP's REAL_SIZE rounding, or the request itself; never
 * zero, because neither CDERIVE nor a narrowed copy may be empty. */
static size_t cap_bound(unsigned int mode, size_t size)
{
    size_t b = (mode & ZEND_CAP_MODE_EXACT) ? size : ((size + 7UL) & ~7UL);
    return heap_round(b);
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
    if (!p) { return (void *)0; }
    /* The request and the bound must fit the block. PHP's REAL_SIZE wraps on a request near
     * SIZE_MAX and its size field is 31 bits, so a hostile request can reach here with a block
     * that is the minimum and a size that is not; refused by name rather than derived past the
     * block (cause 29) or copied past it later by reblock. */
    extent = __builtin_capstone_cap_get_end(p) - __builtin_capstone_cap_get_cursor(p);
    if (hdr > extent || size > extent - hdr || bound > extent - hdr) {
        cap_fail(ZEND_CAP_FAIL_GEOMETRY, "request does not fit the block", size, extent);
    }
    ++st_issues;
    if (mode & ZEND_CAP_MODE_REVOKE) {
        return heap_issue((char *)p + hdr, bound);
    }
    start = __builtin_capstone_cap_get_cursor(p) + hdr;
    return __builtin_capstone_cap_shrink((char *)p + hdr, start, start + bound);
}

/* The block behind a caller's pointer. The pointer must start its object (an interior pointer is
 * refused by name) and must be live: heap_access's probe faults with cause 25 on a stale one,
 * before the table could hand back a block that a later tenant now holds at the same address.
 * The mapping reference heap_access returns is not used: the block comes from the table. CREVOKE
 * below names the child alone, so no instruction checks that the block issued it: the
 * start-of-object test and the table entry at payload - hdr are what remain. */
static void *cap_block_of(void *ptr, size_t hdr, const char *who)
{
    unsigned long cur = __builtin_capstone_cap_get_cursor(ptr);
    unsigned long base = __builtin_capstone_cap_get_base(ptr);
    void *block;
    if (cur != base) { cap_fail(ZEND_CAP_FAIL_INTERIOR, who, cur, base); }
    if (cur < hdr) { cap_fail(ZEND_CAP_FAIL_UNKNOWN, who, cur, hdr); }
    (void)heap_access(ptr);
    block = cap_table_find(cur - hdr);
    if (!block || __builtin_capstone_cap_get_end(block) <= cur) {
        cap_fail(ZEND_CAP_FAIL_UNKNOWN, who, cur, hdr);
    }
    return block;
}

/* ZEND_CAP_HEADER(ptr): _erealloc's and _mem_block_check's derivation. Nothing ends here. */
void *zend_cap_header(void *ptr, size_t hdr)
{
    if (!ptr) { return (void *)0; }
    return cap_block_of(ptr, hdr, "header of a pointer this heap did not issue, or not at its start");
}

/* ZEND_CAP_RECLAIM(ptr): _efree's derivation. In the revoking arms the caller's child ends in
 * the same call: a child already ended (a double efree) is cause 25. The release names the
 * object alone. A grandchild the caller derived at the same start therefore ends by itself and
 * leaves the issued child live, which this heap cannot tell from a correct free. */
void *zend_cap_reclaim(void *ptr, size_t hdr)
{
    void *block;
    if (!ptr) { return (void *)0; }
    block = cap_block_of(ptr, hdr, "free of a pointer this heap did not issue, or not at its start");
    if (zend_cap_mode() & ZEND_CAP_MODE_REVOKE) {
        heap_end(ptr);
        ++st_retires;
    }
    return block;
}

/* ZEND_DO_FREE(p). The block leaves the table and goes back to malloc. The stock arm leaks it on
 * purpose and keeps its entry: mallocng's own free revokes, and a control that revokes is not a
 * control. */
void zend_cap_release(void *p)
{
    if (!p) { return; }
    ++st_frees;
    if (!(zend_cap_mode() & ZEND_CAP_MODE_RETURN)) { return; }
    if (!cap_table_drop(__builtin_capstone_cap_get_cursor(p))) {
        cap_fail(ZEND_CAP_FAIL_UNKNOWN, "release of a block this heap did not take",
                 __builtin_capstone_cap_get_cursor(p), 0);
    }
    free(cap_malloc_ref(p));
}

/* ZEND_DO_REALLOC(p, bytes). In the arms that return blocks, realloc with PHP's request, as
 * stock PHP does: the block stays in place whenever mallocng keeps it there. `oldbytes` is the
 * header plus the old request, which the macro reads from p->size before _erealloc overwrites
 * it; header and payload up to that length must survive. When the slack changes (a block
 * crossing 16 or 64 KiB) they move inside the block: before realloc when the slack shrinks, so
 * that realloc's copy holds them, after it when the slack grows. mallocng's realloc ends the old
 * block lifetime in place or on a move, and with it the caller's old child, so the old pointer
 * is stale afterwards. On failure nothing has changed: the old block, and the caller's pointer
 * into it, are intact for _erealloc's ADD_POINTER_TO_LIST(orig).
 *
 * The stock arm never returns a block, so it cannot call realloc: a fresh block, the copy, and
 * the old block leaked with its table entry, as at efree. */
void *zend_cap_reblock(void *p, size_t bytes, size_t oldbytes)
{
    unsigned int mode = zend_cap_mode();
    struct cap_geometry g;
    unsigned long extent, slack;
    size_t keep;
    char *old, *fresh;
    if (!p) { return zend_cap_block(bytes); }
    extent = __builtin_capstone_cap_get_end(p) - __builtin_capstone_cap_get_cursor(p);
    if (oldbytes > extent) { oldbytes = extent; }   /* never past the old block, whatever p->size says */
    keep = oldbytes < bytes ? oldbytes : bytes;
    if (!(mode & ZEND_CAP_MODE_RETURN)) {
        void *copy = zend_cap_block(bytes);
        if (!copy) { return (void *)0; }
        memcpy(copy, p, keep);
        ++st_reallocs;
        zend_cap_release(p);
        return copy;
    }
    if (!cap_geometry(bytes, &g)) { return (void *)0; }
    slack = __builtin_capstone_cap_get_cursor(p) - __builtin_capstone_cap_get_base(p);
    old = cap_malloc_ref(p);
    if (g.slack < slack) { memmove(old + g.slack, old + slack, keep); }
    fresh = realloc(old, g.request);
    if (!fresh) {
        if (g.slack < slack) { memmove(old + slack, old + g.slack, keep); }
        return (void *)0;
    }
    if (g.slack > slack) { memmove(fresh + g.slack, fresh + slack, keep); }
    cap_table_drop(__builtin_capstone_cap_get_cursor(p));
    cap_table_put(fresh + g.slack);
    ++st_reallocs;
    if (mode & ZEND_CAP_MODE_REVOKE) { ++st_retires; }   /* the old child ends with the old block lifetime */
    return fresh + g.slack;
}
