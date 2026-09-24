/* Domain entry. musl-capstone's runtime (hostcall.c) calls capstone_main from domain_main;
 * its return value reaches the host as the DONE result.
 *
 * Three things the runtime does not do for a program, and FFmpeg needs:
 *  - __environ is never set, and log.c calls getenv(). A NULL environ makes getenv
 *    dereference NULL. libc-test's domain entry sets the same empty environment
 *    (musl-capstone/libc-test/libc_test_domain.c).
 *  - There is no argv, so the input path and the stop stage are compile-time. One image
 *    per milestone keeps every run returning a result (ffapp_decode.h).
 *  - stdout is never flushed when capstone_main RETURNS (runtime/hostcall.c domain_main calls
 *    no exit path), and musl switches stdout to FULL buffering on its first flush, because the
 *    TIOCGWINSZ ioctl fails (ENOTTY). Found 2026-09-23: exactly the first line of every run
 *    reached the host and the rest was lost. So stdout is set LINE-buffered here (each line
 *    is one hostcall round, and a wedge loses at most a partial line), and flushed before
 *    returning. */
#include <stdio.h>

#include "ffapp_decode.h"

#ifndef FFAPP_INPUT
#define FFAPP_INPUT "/mnt/host/input.mkv"
#endif
#ifndef FFAPP_STOP_AT
#define FFAPP_STOP_AT FFAPP_M5_ALL
#endif

extern char **__environ;
#ifdef FFAPP_SUBLET_HEAP
void __capstone_sublet_heap_stats(unsigned long out[9]);
#endif
#ifdef FFAPP_POOL_MODE
#include "ffapp_pool.h"
#endif
static char *ffapp_empty_environ[1] = { 0 };

int capstone_main(void)
{
    __environ = ffapp_empty_environ;
    setvbuf(stdout, NULL, _IOLBF, 0);
#ifdef FFAPP_POOL_MODE
    ffapp_pool_init();
#endif
    int status = ffapp_run(FFAPP_INPUT, FFAPP_STOP_AT);
#ifdef FFAPP_POOL_MODE
    ffapp_pool_report();
#endif
#ifdef FFAPP_SUBLET_HEAP
    /* What the revoking heap spent: split + mrev is the revocation-node count, which silicon
       caps at 65,532 for the life of the machine (runtime/sublet_heap.c). */
    unsigned long hs[9];
    __capstone_sublet_heap_stats(hs);
    printf("FFAPP-HEAP alloc=%lu free=%lu merge=%lu peak-live=%lu split=%lu mrev=%lu delin=%lu revoke=%lu init=%lu\n",
           hs[0], hs[1], hs[2], hs[3], hs[4], hs[5], hs[6], hs[7], hs[8]);
#endif
    fflush(stdout);
    return status;
}
