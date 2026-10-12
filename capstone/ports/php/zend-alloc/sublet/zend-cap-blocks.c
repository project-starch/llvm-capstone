/* The seam: one Sublet-managed block per emalloc, split so the header outlives the payload.
 *
 * Read zend-cap-port.h first -- it carries the geometry and why the split is the design.
 *
 * WHAT EACH HOOK DOES, against the six sites the patch replaces in Zend/zend_alloc.c:
 *
 *   ZEND_DO_MALLOC   -> zend_cap_block_alloc   loan a block, take the SENIOR handle before any
 *                       (zend_alloc.c:182)     split, split at B+48, hand PHP the HEADER alias
 *   ZEND_DO_FREE     -> zend_cap_block_free    senior revoke, then return the loan
 *                       (:287, :491, :564)
 *   ZEND_DO_REALLOC  -> zend_cap_block_realloc a fresh block, copy through the ALLOCATOR's own
 *                       (:345)                 full-payload alias, then retire the old one
 *   p + 48           -> ZEND_CAP_PAYLOAD       take the payload region and shrink it to the
 *                       (:168, :216, :370)     CALLER's size
 *   ptr - 48         -> ZEND_CAP_HEADER        look the header alias up; never arithmetic
 *                       (:250, :320, :660)
 *   new, in _efree   -> ZEND_CAP_RETIRE        give the payload back: the caller's pointer dies
 *                       (after the size calc)  while PHP's header stays live
 *
 * The retire sits BEFORE _efree's branch, so it covers both the AG(cache) park and the real
 * free with one hunk. PHP's own ordering then holds unchanged: the cache park at :271-278 runs
 * with the header live, and REMOVE_POINTER_FROM_LIST at :281 walks neighbour headers that are
 * all still alive, so none of the physical port's unlink-before-revoke deviation is needed.
 *
 * _erealloc's OLD BUG IS GONE BY CONSTRUCTION. Physically it faulted because the lower level
 * copied a whole block's worth through an object-sized bound. Here the allocator never holds
 * the client's capability at all: ZEND_CAP_HEADER converts at the entry, and the copy uses
 * r->alias, the record's own full-payload alias. "Narrow going up, widen going down" stops
 * being a rule to remember.
 */
#include "zend-cap-port.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The nested-allocator loan ABI (runtime/virtual/heap.c:258,283) and the node reservation
 * (declared runtime/virtual/vm.h:23). Declared here rather than by including those headers:
 * this translation unit is compiled against PHP's include path, and the three signatures are
 * the whole of what it needs from the level below. */
unsigned long __capstone_sublet_malloc_linear(size_t n, sublet_cap *out);
void          __capstone_sublet_free_linear(unsigned long base);
long          __capstone_vm_nodes(unsigned long available);

static unsigned int cap_mode;
static int cap_mode_set;
static unsigned long st_blocks, st_frees, st_issues, st_retires, st_revokes, st_splits,
                     st_reallocs, st_live;

void zend_cap_fail(int code, const char *what, unsigned long a, unsigned long b)
{
    /* For a pointer this layer did not issue, say WHAT it is: an interior pointer into a block
     * we do own (the index model is wrong) or an address in no block at all (PHP is freeing
     * memory emalloc never produced). Without this the message is the same in both cases and
     * the next step is a guess. */
    if (code == ZEND_CAP_FAIL_UNKNOWN && a) {
        struct zend_cap_block *owner = zend_cap_find_containing(a);
        fflush(stdout);
        if (owner) {
            fprintf(stderr, "ZEND_CAP_INTERIOR addr=0x%lx is inside the block at 0x%lx "
                            "(+%lu of %lu, payload base 0x%lx, last size %lu)\n",
                    a, owner->base, a - owner->base, owner->bytes, owner->key, owner->size);
        } else {
            fprintf(stderr, "ZEND_CAP_FOREIGN addr=0x%lx is inside no block this layer owns: "
                            "PHP freed memory that did not come from emalloc\n", a);
        }
        fflush(stderr);
    }
    /* LOUD AND TERMINAL. PHP's own answer to a failed allocation is a fatal message and
     * exit(1), which in a results table is indistinguishable from the trigger failing. This
     * marker is distinguishable, and it aborts rather than limping on with a heap that is no
     * longer described by the index. */
    fflush(stdout);
    fprintf(stderr, "ZEND_CAP_FAIL code=%d %s a=0x%lx b=0x%lx "
                    "(blocks=%lu live=%lu issues=%lu retires=%lu)\n",
            code, what, a, b, st_blocks, st_live, st_issues, st_retires);
    fflush(stderr);
    abort();
}

/* THE ARM, NAMED OUT LOUD AND NEVER GUESSED. An absent PHP_CAP_MODE is a failure, not a
 * default: a run that silently chose an arm is a run whose result means nothing, and the
 * suite's own gate requires the two runs of a pair to report DIFFERENT mode lines. */
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
            zend_cap_fail(ZEND_CAP_FAIL_GEOMETRY,
                          "PHP_CAP_MODE is unset: refusing to pick an arm", 0, 0);
        }
        zend_cap_set_mode((unsigned int) strtoul(s, (char **)0, 0));
    }
    return cap_mode;
}

/* The bound handed to the caller: stock PHP's rounding, or the request itself. */
static unsigned long cap_bound(unsigned long size)
{
    if (zend_cap_mode() & ZEND_CAP_MODE_EXACT) { return size ? size : 1UL; }
    return ((size + 7UL) & ~7UL) ? ((size + 7UL) & ~7UL) : 1UL;      /* REAL_SIZE(size) */
}

/* heap.c reserves before every malloc (heap.c:48-56); the take/give cycles below mint nodes
 * without passing through it, so they reserve too. A cause-30 resource fault lands at an
 * arbitrary pc and is otherwise indistinguishable from a catch, so this must fail closed. */
static void reserve_nodes(void)
{
    if (__capstone_vm_nodes(256) < 0) {
        zend_cap_fail(ZEND_CAP_FAIL_NODES, "node table could not be reserved", 256, 0);
    }
}

/* Issue the payload if it is not already out. In REVOKE mode every retire gives the region
 * back, so this takes a FRESH node over the same bytes -- which is what leaves the previous
 * tenant's alias dead even though the bytes are live again. In the control mode the first take
 * is the only one, and the alias stays valid for the life of the block. */
static void cap_issue(struct zend_cap_block *r)
{
    if (r->issued) { return; }
    reserve_nodes();
    r->alias = sublet_take(&r->payload);
    if (!r->alias) {
        zend_cap_fail(ZEND_CAP_FAIL_GEOMETRY, "payload take produced nothing", r->key, 0);
    }
    r->issued = 1;
    ++st_issues;
}

/* End a block: the senior revoke kills the header alias, the payload region and every object
 * alias ever issued from it, in one operation. Then the loan goes back.
 *
 * The lower level revokes too -- __capstone_sublet_free_linear rotates the heap's own
 * authority, which is senior to the loan -- and it does so in BOTH arms. That is deliberate:
 * the level below is held constant so the temporal pair differs only in whether _efree
 * revokes. */
static void cap_retire_block(struct zend_cap_block *r)
{
    unsigned long base = r->base;
    unsigned int mode = zend_cap_mode();

    if (!(mode & ZEND_CAP_MODE_RETURN)) {
        /* THE STOCK COLUMN. Nothing is revoked and the loan is not returned, because returning
         * it would have mallocng revoke the block and catch the defect for us -- see the long
         * note on ZEND_CAP_MODE_RETURN in zend-cap-port.h. The record stays INDEXED too: stock
         * PHP leaves a freed block's header readable, and a trigger that frees the same pointer
         * twice must behave like stock PHP here rather than abort on a missing record. */
        ++st_frees;
        if (st_live) { --st_live; }
        return;
    }
    if (mode & ZEND_CAP_MODE_REVOKE) {
        sublet_give_to(&r->senior, &r->header);
        ++st_revokes;
    }
    zend_cap_unindex(r);
    zend_cap_record_free(r);
    __capstone_sublet_free_linear(base);
    ++st_frees;
    if (st_live) { --st_live; }
}

void *zend_cap_block_alloc(size_t bytes)
{
    struct zend_cap_block *r;
    unsigned long want, base;

    /* PHP asks for 48 + REAL_SIZE(n). The payload must be a whole number of capabilities or
     * sublet_give_to cannot write through it before init, so round the request up; 48 is itself
     * a multiple of the grain, so this rounds the payload and nothing else. */
    want = ZEND_CAP_ROUND(bytes);
    if (want < ZEND_CAP_HDR + ZEND_CAP_GRAIN) { want = ZEND_CAP_HDR + ZEND_CAP_GRAIN; }

    r = zend_cap_record_new();
    if (!r) { return (void *)0; }                 /* zend_cap_record_new already failed loudly */

    base = __capstone_sublet_malloc_linear(want, &r->header);
    if (!base) {
        zend_cap_record_free(r);
        zend_cap_fail(ZEND_CAP_FAIL_LOAN, "the heap would not lend a block", want, 0);
        return (void *)0;
    }
    if (base & (ZEND_CAP_GRAIN - 1UL)) {
        zend_cap_fail(ZEND_CAP_FAIL_ALIGN, "loan base is not capability-aligned", base, want);
    }

    /* THE SENIOR HANDLE IS TAKEN BEFORE ANY SPLIT. A handle taken before the split is senior to
     * both halves, so one revoke at block death ends the header, the payload region and every
     * alias below them. Taken after, it would cover only one half. */
    sublet_handle(&r->header, &r->senior);
    sublet_split(&r->header, base + ZEND_CAP_HDR, &r->payload);
    ++st_splits;

    r->base = base;
    r->bytes = want;
    r->key = base + ZEND_CAP_HDR;
    r->size = 0;
    r->header_alias = sublet_take(&r->header);
    if (!r->header_alias) {
        zend_cap_fail(ZEND_CAP_FAIL_GEOMETRY, "header take produced nothing", base, want);
    }
    zend_cap_index(r);
    ++st_blocks;
    ++st_live;
    return r->header_alias;
}

void zend_cap_block_free(void *p)
{
    struct zend_cap_block *r;
    if (!p) { return; }
    r = zend_cap_find(__builtin_capstone_cap_get_cursor(p) + ZEND_CAP_HDR);
    if (!r) {
        /* PHP is freeing a block this layer never issued. Loud: the alternative is to pass it
         * to the level below, which would either trap there (reading as a port artefact) or
         * succeed on a pointer we know nothing about. */
        zend_cap_fail(ZEND_CAP_FAIL_UNKNOWN, "free of a block this layer did not issue",
                      __builtin_capstone_cap_get_cursor(p), 0);
        return;
    }
    cap_retire_block(r);
}

void *zend_cap_block_realloc(void *p, size_t bytes)
{
    struct zend_cap_block *old, *fresh;
    void *header;
    unsigned long copy, capacity;

    if (!p) { return zend_cap_block_alloc(bytes); }
    old = zend_cap_find(__builtin_capstone_cap_get_cursor(p) + ZEND_CAP_HDR);
    if (!old) {
        zend_cap_fail(ZEND_CAP_FAIL_UNKNOWN, "realloc of a block this layer did not issue",
                      __builtin_capstone_cap_get_cursor(p), 0);
        return (void *)0;
    }

    /* A FRESH BLOCK ALWAYS, never in place. PHP's _erealloc is entitled to a move, and moving
     * is what the temporal axis wants: the old payload's aliases die with the old block. */
    header = zend_cap_block_alloc(bytes);
    if (!header) {
        /* The old block MUST survive a failed realloc: _erealloc's failure path does
         * ADD_POINTER_TO_LIST(orig), which dereferences the original header (:357). */
        return (void *)0;
    }
    fresh = zend_cap_find(__builtin_capstone_cap_get_cursor(header) + ZEND_CAP_HDR);
    if (!fresh) {
        zend_cap_fail(ZEND_CAP_FAIL_UNKNOWN, "the block just allocated is not indexed",
                      __builtin_capstone_cap_get_cursor(header), 0);
        return (void *)0;
    }

    /* The header's 48 bytes carry pNext, pLast, size and cached. _erealloc rewrites the list
     * links and the size straight afterwards, but libc realloc would have preserved them, so
     * they are preserved here: the copy is capability-grained, so the two list pointers arrive
     * as capabilities rather than as two halves each. */
    memcpy(header, p, ZEND_CAP_HDR);

    /* THE COPY GOES THROUGH THE ALLOCATOR'S OWN FULL-PAYLOAD ALIASES, not through anything the
     * caller holds. r->alias spans the whole payload region; what PHP gets is a shrunk copy of
     * it, made in zend_cap_payload. */
    cap_issue(old);
    cap_issue(fresh);
    capacity = (unsigned long)bytes > ZEND_CAP_HDR ? (unsigned long)bytes - ZEND_CAP_HDR : 0UL;
    copy = old->size < capacity ? old->size : capacity;
    if (copy) { memcpy(fresh->alias, old->alias, (size_t)copy); }

    cap_retire_block(old);
    ++st_reallocs;
    return header;
}

void *zend_cap_payload(void *p, size_t size)
{
    struct zend_cap_block *r;
    unsigned long key, end;

    if (!p) { return (void *)0; }
    key = __builtin_capstone_cap_get_cursor(p) + ZEND_CAP_HDR;
    r = zend_cap_find(key);
    if (!r) {
        zend_cap_fail(ZEND_CAP_FAIL_UNKNOWN, "payload of a block this layer did not issue",
                      key, (unsigned long)size);
        return (void *)0;
    }
    cap_issue(r);
    r->size = (unsigned long)size;

    /* Shrink to the caller's object: base at the PAYLOAD base, so the client capability cannot
     * reach PHP's header at all. The hierarchical mode had to keep the base at the block, to
     * let PHP's own ptr-48 arithmetic work; the seam replaces that arithmetic with a lookup, so
     * an under-read below the object is caught here and was not there. */
    end = key + cap_bound((unsigned long)size);
    if (end > r->base + r->bytes) {
        zend_cap_fail(ZEND_CAP_FAIL_GEOMETRY, "requested bound exceeds the block",
                      end, r->base + r->bytes);
    }
    return __builtin_capstone_cap_shrink(r->alias, key, end);
}

void *zend_cap_header(void *ptr)
{
    struct zend_cap_block *r;
    unsigned long key;
    if (!ptr) { return (void *)0; }

    /* Reading the CURSOR of a revoked capability does not trap -- measured, stage 4 of
     * zend-alloc/sublet/probe-sublet-split.c -- so a double efree arrives here rather than
     * faulting on the way in. That is why it is caught by name in zend_cap_retire below
     * instead of being left to fault inside this lookup, where the harness would refuse it as
     * a port artefact. */
    key = __builtin_capstone_cap_get_cursor(ptr);
    r = zend_cap_find(key);
    if (!r) {
        zend_cap_fail(ZEND_CAP_FAIL_UNKNOWN, "header of a pointer this layer did not issue",
                      key, 0);
        return (void *)0;
    }
    return r->header_alias;
}

void zend_cap_retire(void *p)
{
    struct zend_cap_block *r;
    if (!p) { return; }
    if (!(zend_cap_mode() & ZEND_CAP_MODE_REVOKE)) { return; }   /* the control arm */

    r = zend_cap_find(__builtin_capstone_cap_get_cursor(p) + ZEND_CAP_HDR);
    if (!r) {
        zend_cap_fail(ZEND_CAP_FAIL_UNKNOWN, "retire of a block this layer did not issue",
                      __builtin_capstone_cap_get_cursor(p), 0);
        return;
    }
    if (!r->issued) {
        /* The same pointer freed twice with no allocation in between: CWE-415, and a defect of
         * the program rather than of this layer. It is reported by NAME, at a labelled site,
         * because the alternative -- letting it fall through -- parks one block in AG(cache)
         * twice and hands it to two tenants. */
        fflush(stdout);
        fprintf(stderr, "ZEND_CAP_DOUBLE_FREE key=0x%lx size=%lu\n", r->key, r->size);
        fflush(stderr);
        abort();
    }
    sublet_give(&r->payload);
    ++st_revokes;
    r->issued = 0;
    r->alias = (void *)0;
    ++st_retires;
}

void zend_cap_stats(unsigned long out[8])
{
    out[ZEND_CAP_STAT_BLOCKS]   = st_blocks;
    out[ZEND_CAP_STAT_FREES]    = st_frees;
    out[ZEND_CAP_STAT_ISSUES]   = st_issues;
    out[ZEND_CAP_STAT_RETIRES]  = st_retires;
    out[ZEND_CAP_STAT_REVOKES]  = st_revokes;
    out[ZEND_CAP_STAT_SPLITS]   = st_splits;
    out[ZEND_CAP_STAT_REALLOCS] = st_reallocs;
    out[ZEND_CAP_STAT_LIVE]     = st_live;
}
