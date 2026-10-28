/* Does the nested layer's design still work after its old ABI was removed?
 *
 * WHY THIS EXISTS, AND WHY IT COMES BEFORE TOUCHING zend-cap-blocks.c. The 2026-10-25 merge
 * deleted __capstone_sublet_malloc_linear/_free_linear from runtime/virtual/heap.c -- the
 * Sublet loan this nested layer borrowed every PHP block through, and which this port was the only
 * consumer of. sublet.h itself survives, so the nested layer's SHAPE is unaffected: split at the
 * 48-byte header, retire the payload per _efree, never revoke the header. Only the block
 * PROVIDER is gone.
 *
 * The replacement is a different model, and a simpler one. capability.h's CDERIVE/CREVOKE take
 * a NON-LINEAR parent and cut children out of it directly:
 *   - capstone_cap_derive(parent, offset, size) -- offset from the parent's BASE, child inherits
 *     data rights and gains MANAGE; the parent is unchanged.
 *   - capstone_cap_revoke_child(parent, child) -- ends that subtree WITHOUT touching payload
 *     bytes, which is exactly what _efree needs while PHP still reads p->size and writes
 *     p->cached afterwards.
 * mallocng's block capability is non-linear and carries MANAGE -- heap.c revokes its own child
 * through it (heap.c:240) -- so __capstone_malloc_slot_start() should be a usable parent.
 *
 * SHOULD BE. Every sentence above is read off a header, and a header is not a machine. What this
 * probe settles, in ~200 lines rather than from a 5 MB PHP image whose fault lands somewhere in
 * Zend:
 *   stage 0  the positive control: carve a block, derive header and payload, write both, free.
 *            If this does not COMPLETE, nothing the other stages report means anything.
 *   stage 1  the geometry: are the two children bounded exactly where the nested layer assumes, with the
 *            header at the object and not at the block base? mallocng's slot prefix sits BELOW
 *            the object, so an offset taken from the parent's base is NOT the object's offset.
 *   stage 2  the whole point: after revoking the payload, is the HEADER still writable? This is
 *            the claim the design rests on, and the one that decides whether PHP's AG(cache)
 *            park at zend_alloc.c:271-278 can run unmodified or needs the physical port's
 *            unlink-before-revoke deviation.
 *   stage 3  does a revoked payload actually DENY? A silent survival here would score as a
 *            catch later, which is the worst failure this experiment can have.
 *   stage 4  re-derive over the same bytes: the cache-hit path. The new child must work and the
 *            PREVIOUS tenant's alias must stay dead.
 *   stage 5  the double-_efree hazard: ZEND_CAP_HEADER(ptr) reads the CURSOR of the caller's
 *            pointer, so a second _efree reads the cursor of a revoked capability. If that
 *            traps, it must trap at a named label here rather than inside the allocator, where
 *            the harness refuses it as a port artefact.
 *
 * ONE STAGE PER RUN, SELECTED BY ARGV, because most stages are expected to FAULT and a fault
 * ends the process. Each prints its marker BEFORE the operation under test, so the harness reads
 * "reached S, then faulted" instead of inferring it.
 */
#include <capstone/capability.h>
#include <malloc_capstone.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HDR   48UL    /* sizeof(zend_mem_header) on a 16-byte-pointer target */
#define BLOCK 256UL   /* header + a 208-byte payload; both ends 16-aligned   */

static void say(const char *s) { printf("%s\n", s); fflush(stdout); }

static unsigned long base_of(void *c) { return __builtin_capstone_cap_get_base(c); }
static unsigned long end_of(void *c)  { return __builtin_capstone_cap_get_end(c); }
static unsigned long cur_of(void *c)  { return __builtin_capstone_cap_get_cursor(c); }

/* The block, and the two children the nested layer would hand out. `parent` is the heap's own
 * capability over the block -- non-linear, carrying MANAGE -- and is never handed upward. */
struct carved {
    void *object;            /* what malloc returned                       */
    void *parent;            /* the block capability: the revocation parent */
    void *header;            /* [object, object+48)  -- never revoked      */
    void *payload;           /* [object+48, object+BLOCK) -- per-_efree    */
    unsigned long offset;    /* the object's offset within the parent      */
};

static int carve(struct carved *c)
{
    struct capstone_malloc_slot *r;
    c->object = malloc(BLOCK);
    if (!c->object) { say("PROBE_FAIL:malloc returned NULL"); return -1; }
    r = __capstone_malloc_find(c->object);
    if (!r) { say("PROBE_FAIL:the heap does not know the block it just returned"); return -1; }
    c->parent = __capstone_malloc_slot_start(r);
    if (!c->parent) { say("PROBE_FAIL:no block capability on the slot"); return -1; }
    /* From the parent's BASE, as CDERIVE requires -- and the object is NOT at the base:
     * mallocng's slot prefix lies below it. Taking the offset from the object's cursor is the
     * whole reason this is computed rather than assumed to be zero. */
    c->offset = cur_of(c->object) - base_of(c->parent);
    c->header  = capstone_cap_derive(c->parent, c->offset, HDR);
    c->payload = capstone_cap_derive(c->parent, c->offset + HDR, BLOCK - HDR);
    if (!c->header)  { say("PROBE_FAIL:header derive returned nothing"); return -1; }
    if (!c->payload) { say("PROBE_FAIL:payload derive returned nothing"); return -1; }
    return 0;
}

int main(int argc, char **argv)
{
    struct carved c;
    int stage = argc > 1 ? atoi(argv[1]) : 0;

    if (stage == 0) {
        say("PROBE_STAGE:0 control -- carve, write both children, free");
        if (carve(&c)) return 1;
        memset(c.header, 0xAA, HDR);
        memset(c.payload, 0x55, BLOCK - HDR);
        say("PROBE_WROTE:both children");
        capstone_cap_revoke_child(c.parent, c.header);
        capstone_cap_revoke_child(c.parent, c.payload);
        free(c.object);
        say("PROBE_OK:0");
        return 0;
    }

    if (stage == 1) {
        say("PROBE_STAGE:1 geometry");
        if (carve(&c)) return 1;
        printf("PROBE_GEOM object_off=%lu parent=[%lu,%lu) header=[%lu,%lu) payload=[%lu,%lu)\n",
               c.offset, base_of(c.parent), end_of(c.parent),
               base_of(c.header), end_of(c.header),
               base_of(c.payload), end_of(c.payload));
        fflush(stdout);
        /* The nested layer assumes exactly this, so it is asserted rather than eyeballed. */
        if (base_of(c.header) != cur_of(c.object))
            say("PROBE_FAIL:header does not start at the object");
        else if (end_of(c.header) != cur_of(c.object) + HDR)
            say("PROBE_FAIL:header is not 48 bytes");
        else if (base_of(c.payload) != cur_of(c.object) + HDR)
            say("PROBE_FAIL:payload does not start just above the header");
        else if (end_of(c.payload) != cur_of(c.object) + BLOCK)
            say("PROBE_FAIL:payload does not reach the end of the block");
        else
            say("PROBE_OK:1");
        return 0;
    }

    if (stage == 2) {
        say("PROBE_STAGE:2 the header survives the payload's revoke");
        if (carve(&c)) return 1;
        memset(c.payload, 0x55, BLOCK - HDR);
        capstone_cap_revoke_child(c.parent, c.payload);
        say("PROBE_REVOKED:payload");
        /* THE CLAIM THE WHOLE DESIGN RESTS ON. PHP writes p->cached here. */
        memset(c.header, 0xAA, HDR);
        say("PROBE_OK:2 header still writable after the payload died");
        return 0;
    }

    if (stage == 3) {
        say("PROBE_STAGE:3 a revoked payload denies");
        if (carve(&c)) return 1;
        capstone_cap_revoke_child(c.parent, c.payload);
        say("PROBE_ABOUT_TO_FAULT:3 storing through the revoked payload");
        memset(c.payload, 0x11, 16);
        say("PROBE_FAIL:3 the denial did not happen");
        return 1;
    }

    if (stage == 4) {
        void *again;
        say("PROBE_STAGE:4 re-derive over the same bytes (the AG(cache) hit)");
        if (carve(&c)) return 1;
        capstone_cap_revoke_child(c.parent, c.payload);
        again = capstone_cap_derive(c.parent, c.offset + HDR, BLOCK - HDR);
        if (!again) { say("PROBE_FAIL:4 re-derive returned nothing"); return 1; }
        memset(again, 0x22, BLOCK - HDR);
        say("PROBE_OK:4a the fresh child works");
        say("PROBE_ABOUT_TO_FAULT:4 storing through the PREVIOUS tenant's alias");
        memset(c.payload, 0x33, 16);
        say("PROBE_FAIL:4 the previous tenant's alias survived a re-derive");
        return 1;
    }

    if (stage == 5) {
        unsigned long seen;
        say("PROBE_STAGE:5 reading the CURSOR of a revoked capability");
        if (carve(&c)) return 1;
        capstone_cap_revoke_child(c.parent, c.payload);
        say("PROBE_ABOUT_TO_FAULT:5 reading the revoked payload's cursor");
        seen = cur_of(c.payload);
        printf("PROBE_OK:5 the cursor read did NOT trap, it gave %lu\n", seen);
        fflush(stdout);
        return 0;
    }

    say("PROBE_FAIL:no such stage");
    return 2;
}
