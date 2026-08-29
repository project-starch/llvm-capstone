/* WAMR EMS, PR 2279: a free coalesces BACKWARDS into a live object.
 *
 * gc_realloc_vo_internal splits a block and adds the remainder to the free list
 * without setting the new header's pinuse bit. A later free then believes the
 * PRECEDING block is free too and merges into it -- and the preceding block is
 * live. The allocator hands the same memory out twice.
 *
 * WHY IT IS A BLIND SPOT. Every byte of this lives inside `store[]`, one static
 * array that EMS carves internally. A capability derived from store covers all of
 * it, so purecap sees an in-bounds access and revocation has nothing to revoke:
 * nothing ever reached malloc or free at the system level. The oracle therefore
 * cannot be a crash. It is the ANSWER: two live allocations that overlap.
 *
 * The trigger sequence is upstream's own samples/mem-allocator/main.c, which is
 * the verified reproducer for this PR; what is added here is the question.
 *
 *   EMS=1  the live block is untouched and nothing overlaps it   -> correct
 *   EMS=2  a later allocation overlaps the live block, or its    -> the bug
 *          contents changed under it
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "mem_alloc.h"

#define LIVE 256
#define GROW 12

static char store[1000];

int
main(void)
{
    mem_allocator_t a = mem_allocator_create(store, sizeof(store));
    uint8_t *p, *q;
    int overlapped = 0, clobbered = 0, i;

    if (!a) { printf("EMS=0 create failed\n"); return 3; }

    p = mem_allocator_malloc(a, LIVE);
    if (!p) { printf("EMS=0 first malloc failed\n"); return 3; }
    p = mem_allocator_realloc(a, p, LIVE + GROW);
    if (!p) { printf("EMS=0 realloc failed\n"); return 3; }

    /* Upstream's forged header, verbatim: hmu at p+256 claiming HMU_FC/size. */
    *(uint32_t *)(p + LIVE) = (1u << 30) | 0x20u;
    *(uint32_t *)(p + LIVE + GROW - 4) = GROW;

    /* p stays LIVE from here on. Stamp it so a backward coalesce is visible. */
    memset(p, 0xA5, LIVE);

    for (i = 0; i < 2; i++) {
        q = mem_allocator_malloc(a, LIVE);
        if (!q) { printf("EMS=0 malloc %d failed\n", i); return 3; }
        /* THE QUESTION: q must not touch p, which is still live. */
        if (q < p + LIVE + GROW && q + LIVE > p)
            overlapped = 1;
        memset(q, 0x5A, LIVE);
        mem_allocator_free(a, q);
    }

    for (i = 0; i < LIVE; i++)
        if (p[i] != (uint8_t)0xA5) { clobbered = 1; break; }

    printf("EMS=%d overlap=%d clobber=%d\n",
           (overlapped || clobbered) ? 2 : 1, overlapped, clobbered);
    mem_allocator_free(a, p);
    return 0;
}
