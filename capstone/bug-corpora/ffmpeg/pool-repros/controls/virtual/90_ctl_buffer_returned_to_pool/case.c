/* CONTROL, not a defect: a buffer handed back to its AVBufferPool, then read through the pointer
 * that held it. The pool keeps the entry for its next get, so no system free happens: on
 * `virtual-malloc` (stock pools, each entry one virtual-mallocng object) the read must COMPLETE;
 * on `virtual-nested-pools` (FFPOOL_SUBLET, patch 0003) the release revokes the buffer and the read
 * must FAULT at read_probe. Run as `buggy 90`. */
#include "corpus.h"

__attribute__((noinline)) unsigned read_probe(const volatile unsigned char *p) {
  return *p;
}

FF2_CASE(90) {
  AVBufferRef *b = av_buffer_pool_get(g_pool);
  CHECK(b, 710);
  memset(b->data, 0xA7, POOL_BYTES);
  const volatile unsigned char *held = b->data;
  av_buffer_unref(&b); /* back to the pool, not to malloc */
  unsigned seen = fixed ? 0xA7 : read_probe(held);
  FF2_VERDICT(!fixed && seen == 0xA7, fixed, "control: the returned buffer was read through its old pointer",
              "control: no fixed sequence");
  return 0;
}
