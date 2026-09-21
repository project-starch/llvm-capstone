/* What cache.c and cache.h need, freestanding, in place of <stdlib.h>,
 * <string.h>, <inttypes.h>, <assert.h> and <pthread.h>. queue.h, the BSD list
 * macros cache.h includes, is self-contained and stays upstream's. The four
 * allocation calls cache.c makes are not declared here on purpose: the port's
 * second patch replaces every one, and an unreplaced call must fail to compile
 * rather than quietly reach a libc. */
#ifndef MC_CACHE_SHIM_H
#define MC_CACHE_SHIM_H
#include <stddef.h>
#include <stdint.h>
#include "mc_pthread_shim.h"

/* memcached ships with -DNDEBUG (Makefile.am:92); this is the same binary.
 * Without it cache.c keeps redzones around every object and raises SIGABRT,
 * a debugging build the port does not carry. */
#ifndef NDEBUG
#error "memcached builds with -DNDEBUG (Makefile.am:92); pass it here too"
#endif
#define assert(x) ((void)0)
#endif
