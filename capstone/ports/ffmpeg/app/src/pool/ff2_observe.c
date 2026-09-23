/* The FFmpeg app port's stand-in for the buffer-pool port's event OBSERVER: every hook is a
 * no-op. The replay's observer keeps lifetime tables (1,024 pools, 8,192 blocks, never
 * reused) and treats an exhausted table or a NULL pool get as fatal; none of that is a
 * property of the pools being protected, and a whole program would outgrow the tables. The
 * protection itself is in the payload hooks (patch 0002, pool-allocator.c), untouched.
 * prepare-source.sh --pool copies this file into libavutil/ as ff2_observe.c. */
#include "trace.h"

uint64_t ff2_begin(unsigned op, unsigned kind, void *pool, size_t size, uint64_t flags,
                   void *object)
{
    (void)op; (void)kind; (void)pool; (void)size; (void)flags; (void)object;
    return 0;
}
void ff2_end(uint64_t call, void *result) { (void)call; (void)result; }
void ff2_new(unsigned kind, void *pool, void *object) { (void)kind; (void)pool; (void)object; }
void ff2_drop(unsigned kind, void *pool, void *object) { (void)kind; (void)pool; (void)object; }
uint64_t ff2_callback(unsigned kind, void *pool, void *object, unsigned type)
{
    (void)kind; (void)pool; (void)object; (void)type;
    return 0;
}
