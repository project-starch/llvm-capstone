/* What a domain gives the runtime's files, for the native tests that compile one
 * of them directly (level0.c, mmap_shm_level0.c). Included before the runtime file,
 * it lets that file build as it ships, capability operations included, so a native
 * test exercises the code an application runs and not a variant made for the host:
 *  - a pointer is its address and carries no bounds: the cursor is the address, the
 *    end is the top of the address space, and narrowing hands the pointer back.
 *    The code around the bounds runs; enforcing them is the domain's, and so is
 *    testing that (run-heap.py);
 *  - one native thread needs no runtime lock (capstone/lock.h);
 *  - the unserved report (hostcall.c) is a counter the test can read. */
#ifndef CAPSTONE_TESTS_NATIVE_RUNTIME_H
#define CAPSTONE_TESTS_NATIVE_RUNTIME_H

#include <stdint.h>
#include <capstone/lock.h>

#define __builtin_capstone_cap_get_cursor(p) ((unsigned long)(uintptr_t)(p))
#define __builtin_capstone_cap_get_end(p) ((void)(p), ~0UL)
#define __builtin_capstone_cap_shrink(p, lo, hi) ((void)(lo), (void)(hi), (void *)(p))

void capstone_lock(volatile int *word) { (void)word; }
void capstone_unlock(volatile int *word) { (void)word; }

int native_unserved;
void __capstone_hc_note_unserved(long n) { (void)n; ++native_unserved; }

#endif
