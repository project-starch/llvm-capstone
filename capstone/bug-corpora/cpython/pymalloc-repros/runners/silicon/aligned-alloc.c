/* The silicon runner's aligned_alloc, linked into every case image by bug-corpora/tools/run-silicon-cases.py.
 *
 * The B0 runtime's heap is level0, which has no aligned_alloc. musl's own depends on oldmalloc's internals, so linking
 * it pulls oldmalloc's malloc.o in beside level0, and the link fails on duplicate __libc_malloc_impl, __libc_realloc
 * and __libc_free.
 *
 * The replay (ports/cpython/pymalloc/src/native/main.c) makes two aligned requests: 16 MiB and 64 MiB, each at 16 KiB.
 * Under CAPSTONE_LEVEL0_LIFETIMES, level0 places a request of 4 KiB or more on the granule of its padded size, so these
 * land on 32 KiB and 128 KiB boundaries. The alignment is checked, never assumed: a block that misses it is freed and
 * NULL returned, and the replay reports that as exit 4 before the case runs. */
#include <stdint.h>
#include <stdlib.h>

void *aligned_alloc(size_t alignment, size_t size) {
  void *p = malloc(size);
  if (p && ((uintptr_t)p & (alignment - 1)) != 0) {
    free(p);
    return NULL;
  }
  return p;
}
