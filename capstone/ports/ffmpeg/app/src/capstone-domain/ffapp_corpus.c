/* The FFmpeg pool bug corpus (capstone/bug-corpora/ffmpeg/pool-repros) in this port's domain.
 *
 * Each case.c runs UNCHANGED against FFmpeg's own libavutil as the pool arms build it. On
 * poolsublet that is the Sublet port of the pools, where a buffer's return to its pool is a
 * revoke. On poolstock it is upstream's pools, the one-macro control. The corpus's other protected
 * arm (the buffer-pool port's probe cases 36-38) runs FFmpeg's buffer.c with the payloads served
 * by that port's own allocator and its leases; here the pools are FFmpeg's own, ported.
 *
 * This file stands in for the corpus's shared/driver.c. The case and its arm (upstream's defect,
 * or the fix applied) are compile-time, like the port's safety fixtures: one image per case and
 * arm, fixture 40 + 2 * case + fixed (build-domain.sh, FFAPP_CORPUS_DIR), because a fault ends the
 * domain. The case prints its own verdict line, exactly as natively. */
#include <stdio.h>
#include <stdlib.h>

#include "corpus.h"

#ifndef FFAPP_FIXTURE
#error "FFAPP_FIXTURE: 40 + 2 * case + fixed"
#endif
#ifndef FFAPP_CORPUS_FIXED
#error "FFAPP_CORPUS_FIXED: 0 runs upstream's defect, 1 the fix"
#endif

AVBufferPool *g_pool;
/* The side-table pool, mirroring the corpus's shared/driver.c. A case that does not use
 * it is unaffected by its existence; case 3 is the first that does, and without this the
 * link fails with "undefined symbol: g_refpool". */
AVRefStructPool *g_refpool;

/* The corpus's contract: an infrastructure failure is never a verdict. */
_Noreturn void ff2_fail(unsigned code)
{
    printf("CONTROL-FAILED %u\n", code);
    fflush(stdout);
    exit(75);
}

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("FFAPP-FIX %d begin\n", FFAPP_FIXTURE);
    g_pool = av_buffer_pool_init(POOL_BYTES, NULL);   /* the driver's pool, as natively */
    if (!g_pool)
        ff2_fail(605);
    g_refpool = av_refstruct_pool_alloc(TAB_BYTES, 0); /* same codes as the corpus driver */
    if (!g_refpool)
        ff2_fail(606);
    printf("case=%d arm=%s\n", ff2_case_number, FFAPP_CORPUS_FIXED ? "fixed" : "buggy");
    fflush(stdout);
    int rc = ff2_case_run(FFAPP_CORPUS_FIXED);
    av_buffer_pool_uninit(&g_pool);
    av_refstruct_pool_uninit(&g_refpool);
    printf("FFAPP-FIX %d mark=%x\n", FFAPP_FIXTURE, rc);
    fflush(stdout);
    return rc;
}
