/* What extent does a pool-issued buffer actually carry?
 *
 * WHY THIS EXISTS. The corpus cannot answer it. Modes 0 and 1 of the
 * capability backend both report 0 of 4 on this corpus, and so does the native
 * pointer backend, so three arms are indistinguishable by their verdicts --
 * and the claim "mode 0 gives each lease its own bounds" would rest on reading
 * ff2_payload_issue_pointer rather than on a measurement. A silence that three
 * different mechanisms produce is exactly where a reader is entitled to ask
 * for the mechanism to be shown directly.
 *
 * WHAT IT PRINTS. One line per mode, for a buffer the pool hands out:
 *
 *   BOUNDS mode=0 requested=64 length=64 base=0x... PER-OBJECT
 *
 * `length` is the issued capability's own extent, read with LCC through a
 * slot, not a size the program remembers. PER-OBJECT means that extent equals
 * what the lease asked for; WHOLE-BLOCK means it is the block's, which is what
 * an unshrunk interior pointer carries and what the native backend hands out.
 *
 * Built and run exactly like a case of this corpus, against the same library,
 * so the thing measured is the thing the cases ran on. It is an instrument,
 * not a case: it has no fix/defect pair and no verdict.
 */
#include "corpus.h"
#include "payload-backend.h"

#include <capstone/capability.h>

AVBufferPool *g_pool;
AVRefStructPool *g_refpool;

/* The corpus's shared driver supplies these for a case; this instrument is its
 * own program, so it supplies them itself. */
_Noreturn void ff2_fail(unsigned code) {
  fprintf(stderr, "CONTROL-FAILED %u\n", code);
  exit(75);
}
void ff2_sink(const struct ff2_event *event) { (void)event; }
void ff2_lock(void) {}
void ff2_unlock(int *guard) { (void)guard; }

int main(int argc, char **argv) {
  unsigned mode = argc > 1 ? (unsigned)atoi(argv[1]) : 0;
  if (mode > 2) {
    fprintf(stderr, "CONTROL-FAILED mode %s is not 0, 1 or 2\n", argv[1]);
    return 75;
  }
  void *metadata = aligned_alloc(64, FF2_META_BYTES);
  if (!metadata)
    ff2_fail(604);
  ff2_memory_init(metadata, FF2_META_BYTES);
#ifdef FFPOOL_BORROW_LINEAR
  ff2_payload_borrow(FF2_PAYLOAD_BYTES);
#else
  void *payload = aligned_alloc(64, FF2_PAYLOAD_BYTES);
  if (!payload)
    ff2_fail(604);
  ff2_payload_init(payload, FF2_PAYLOAD_BYTES);
  if (mode)
    ff2_fail(607);
#endif
  ff2_set_mode(mode);
  ff2_reset();
  g_pool = av_buffer_pool_init(POOL_BYTES, NULL);
  if (!g_pool)
    ff2_fail(605);

  AVBufferRef *b = av_buffer_pool_get(g_pool);
  if (!b || !b->data)
    ff2_fail(608);
  /* Read the extent off the capability itself. capstone_cap_store puts the
   * register's capability into a slot; the metadata readers restore it. */
  capstone_cap_slot slot;
  capstone_cap_store(&slot, b->data);
  unsigned long base = capstone_cap_base(&slot);
  unsigned long end = capstone_cap_end(&slot);
  unsigned long length = end - base;
  printf("BOUNDS mode=%u requested=%u length=%lu base=%#lx %s\n", mode,
         (unsigned)POOL_BYTES, length, base,
         length == (unsigned long)POOL_BYTES
             ? "PER-OBJECT"
             : (length >= FF2_PAYLOAD_BYTES ? "WHOLE-PAYLOAD" : "WIDER-THAN-REQUESTED"));
  av_buffer_unref(&b);
  av_buffer_pool_uninit(&g_pool);
  free(metadata);
  return 0;
}
