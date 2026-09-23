/* FFmpeg's pools under the buffer-pool port's lifetime hooks, in the whole program.
 *
 * The pool arms (build-domain.sh FFAPP_POOL) build libavutil from prepare-source.sh --pool
 * and link the buffer-pool port's payload allocator and Capstone backend. The payload region
 * is the host's FOURTH shared region, program region 1 in hostcall.c (region 0 is the heap),
 * transferred linear; it must be handed over before FFmpeg's first pool get. The mode is the
 * buffer-pool port's: 0 bounds only, 1 revoke on backing free, 2 a lease per get, revoked on
 * return to the pool. Header-only, included by the domain entries of the pool arms. */
#ifndef FFAPP_POOL_H
#define FFAPP_POOL_H

#include <stdio.h>
#include "trace.h"

#ifndef FFAPP_POOL_REGION_BYTES
#error "FFAPP_POOL_REGION_BYTES must equal the region the guest host shares"
#endif

void *__capstone_region(unsigned index);

static void ffapp_pool_init(void)
{
    ff2_set_mode(FFAPP_POOL_MODE);
    ff2_payload_init(__capstone_region(1), FFAPP_POOL_REGION_BYTES);
}

/* the pool's own Sublet counts, beside the heap's FFAPP-HEAP line */
static void ffapp_pool_report(void)
{
    struct ff2_header h = {0};
    ff2_memory_report(&h);
    printf("FFAPP-POOL mode=%d payload-used=%llu split=%llu mrev=%llu delin=%llu revoke=%llu init=%llu\n",
           FFAPP_POOL_MODE, (unsigned long long)h.payload_used, (unsigned long long)h.split,
           (unsigned long long)h.mrev, (unsigned long long)h.delin,
           (unsigned long long)h.revoke, (unsigned long long)h.init);
}

#endif
