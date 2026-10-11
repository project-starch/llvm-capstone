/* The FFmpeg app port's stand-in for the buffer-pool port's event RECORDER.
 *
 * The buffer-pool port's patch 0001 threads event hooks through libavutil's pools and adds
 * ff2_record.o and ff2_observe.o to libavutil's Makefile. In the replay, those record every
 * pool event to a trace file and check the pool's invariants. A whole program has no trace
 * to write, and its pools are driven by the decoder, not replayed, so here:
 *  - the lock is a no-op: the app is built --disable-pthreads and runs one thread;
 *  - events go nowhere;
 *  - a failure the pool code reports (ff2_fail: a stale or foreign authority offered back,
 *    an allocator the port cannot take) is PRINTED and ends the program with its code, so a
 *    run still returns a result that names what went wrong.
 * prepare-source.sh --pool copies this file into libavutil/ as ff2_record.c. */
#include <stdio.h>
#include <stdlib.h>

#include "trace.h"

void ff2_lock(void) {}
void ff2_unlock(int *guard) { (void)guard; }
void ff2_sink(const struct ff2_event *event) { (void)event; }
void ff2_finish(void) {}
void ff2_reset(void) {}

/* pool-allocator.c reports it; the metadata here is the program heap's, counted there.
   Declared here: its header (metadata-allocator.h) belongs to the replay's allocator. */
size_t ff2_metadata_used(void);
size_t ff2_metadata_used(void) { return 0; }

_Noreturn void ff2_fail(unsigned code)
{
    printf("FFAPP-POOL fail %u\n", code);
    fflush(stdout);
    exit((int)code);
}
