/* posix_memalign and aligned_alloc for corpus programs on the silicon ABI, over level0's malloc.
 *
 * level0 (ports/musl-capstone/runtime/level0.c) defines malloc, free, calloc, realloc and musl's internal
 * __libc_* names, but no aligned allocation. A program that calls posix_memalign -- the FFmpeg buffer-pool port's
 * av_malloc, for its SIMD alignment -- would pull musl's own allocator out of the archive and its __libc_* names
 * would collide with level0's at link time. This file answers the two calls from level0 instead.
 *
 * The alignment asked for is NOT honoured beyond level0's 16 bytes: a block cannot start anywhere but at its
 * header, or free would not find it. The corpora that link this (the FFmpeg pool and sub-object cases) do not
 * depend on it; the recorded capstone-application runs of the sub-object corpus mapped aligned_alloc to malloc for
 * the same reason (its record.json cc_args). Linked by tools/run-silicon-cases.py --extra-src. */
#include <errno.h>
#include <stdlib.h>

int posix_memalign(void **out, size_t align, size_t n) {
  (void)align;
  void *p = malloc(n ? n : 1);
  if (!p)
    return ENOMEM;
  *out = p;
  return 0;
}

void *aligned_alloc(size_t align, size_t n) {
  (void)align;
  return malloc(n);
}
