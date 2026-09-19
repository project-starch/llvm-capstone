/* Direct client of the extracted FFmpeg pool library, without a trace driver. */
#include "libavutil/buffer.h"
#include "libavutil/refstruct.h"
#include "trace.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef FFPOOL_POISONCAP
#include <sys/mman.h>
#endif

_Noreturn void ff2_fail(unsigned code) {
  fprintf(stderr, "FFPOOL example failed: %u\n", code);
  exit(1);
}
/* A client may collect these events; this example checks payloads directly. */
void ff2_sink(const struct ff2_event *event) { (void)event; }
/* This extracted example is single-threaded, like the replay. */
void ff2_lock(void) {}
void ff2_unlock(int *guard) { (void)guard; }
int main(void) {
  void *metadata = aligned_alloc(64, FF2_META_BYTES);
#ifdef FFPOOL_POISONCAP
  void *payload = mmap(NULL, FF2_PAYLOAD_BYTES, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANON, -1, 0);
  if (payload == MAP_FAILED)
    return 1;
#else
  void *payload = aligned_alloc(64, FF2_PAYLOAD_BYTES);
#endif
  if (!metadata || !payload)
    return 1;
  ff2_memory_init(metadata, FF2_META_BYTES);
  ff2_payload_init(payload, FF2_PAYLOAD_BYTES);
#ifdef FFPOOL_POISONCAP
  ff2_set_mode(2);
#else
  ff2_set_mode(0);
#endif
  ff2_reset();
  AVBufferPool *pool = av_buffer_pool_init(64, NULL);
  AVBufferRef *a = av_buffer_pool_get(pool);
  AVBufferRef *b = av_buffer_pool_get(pool);
  if (!a || !b)
    return 1;
  memset(a->data, 17, 64);
  memset(b->data, 41, 64);
  AVBufferRef *alias = av_buffer_ref(a);
  av_buffer_unref(&a);
  if (alias->data[0] != 17 || b->data[63] != 41)
    return 1;
  av_buffer_unref(&alias);
  a = av_buffer_pool_get(pool);
  a->data[0] = 61;
  av_buffer_pool_uninit(&pool); /* outstanding references remain valid */
  if (a->data[0] != 61 || b->data[0] != 41)
    return 1;
  av_buffer_unref(&a);
  av_buffer_unref(&b);
  ff2_finish();
  free(metadata);
#ifdef FFPOOL_POISONCAP
  munmap(payload, FF2_PAYLOAD_BYTES);
#else
  free(payload);
#endif
  printf("ALLOCATOR_EXAMPLE ffmpeg PASS pointer_bytes=%zu\n", sizeof(void *));
  return 0;
}
