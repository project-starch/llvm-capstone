/* Complete decoder entry; argv and environment come from the application SDK. */
#include <stdio.h>

#include "ffapp_decode.h"

#ifndef FFAPP_INPUT
#define FFAPP_INPUT "/mnt/host/input.mkv"
#endif
#ifndef FFAPP_STOP_AT
#define FFAPP_STOP_AT FFAPP_M5_ALL
#endif

#ifdef FFAPP_SUBLET_HEAP
void __capstone_sublet_heap_stats(unsigned long out[9]);
#endif
#ifdef FFAPP_POOL_MODE
#include "ffapp_pool.h"
#endif
#ifdef FFAPP_SUBLET_POOLS
void ff_sublet_report(void);
#endif

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    setvbuf(stdout, NULL, _IOLBF, 0);
#ifdef FFAPP_POOL_MODE
    ffapp_pool_init();
#endif
    int status = ffapp_run(argc > 1 ? argv[1] : FFAPP_INPUT, FFAPP_STOP_AT);
#ifdef FFAPP_POOL_MODE
    ffapp_pool_report();
#endif
#ifdef FFAPP_SUBLET_POOLS
    /* the pools' own counts: blocks taken from the heap and ended, entries, takes, gives */
    ff_sublet_report();
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
