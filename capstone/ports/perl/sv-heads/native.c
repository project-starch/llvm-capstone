/* Native test backend for the SV-head adapter. It is not a study arm.
 *
 * PERL_SVH_NATIVE_MODE=0 publishes a released head at once, as upstream
 * does. PERL_SVH_NATIVE_MODE=1, also the default when the variable is absent
 * (upstream tests start child perls with a cleared environment), keeps
 * released heads out of reuse in a
 * 4,096-entry FIFO and, under AddressSanitizer, poisons them for that time.
 * A patched Perl built with -fsanitize=address and this backend therefore
 * reports any read of a released head that bypasses the adapter's record,
 * which is exactly what the protected Capstone and CheriBSD arms would turn
 * into a fault. Heads come from one aligned reservation, as on both targets. */
#define SVH_PLATFORM "native"
#define SVH_BACKEND_SLOT
#define SVH_HEAD_ALIGN sizeof(void *)
#define SVH_REPORT_OPTIONAL 1
#include "sv-heads.h"

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define SVH_ASAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#define SVH_ASAN 1
#endif
#ifdef SVH_ASAN
#include <sanitizer/asan_interface.h>
#define SVH_POISON(p, n) ASAN_POISON_MEMORY_REGION((p), (n))
#define SVH_UNPOISON(p, n) ASAN_UNPOISON_MEMORY_REGION((p), (n))
#else
#define SVH_POISON(p, n) ((void)(p), (void)(n))
#define SVH_UNPOISON(p, n) ((void)(p), (void)(n))
#endif

#define NATIVE_REGION_BYTES (64UL << 20)
#define NATIVE_DELAY 4096u
static unsigned char *native_region;
static uint32_t native_fifo[NATIVE_DELAY];
static size_t native_head, native_count, native_published;

static void svh_backend_init(size_t bytes, uint64_t *base, size_t *capacity) {
  const char *mode = getenv("PERL_SVH_NATIVE_MODE");
  if (!mode)
    mode = "1";
  if ((mode[0] != '0' && mode[0] != '1') || mode[1])
    svh_fail(851, "select PERL_SVH_NATIVE_MODE=0 or 1");
  svh.mode = (unsigned)(mode[0] - '0');
  native_region = aligned_alloc(4096, NATIVE_REGION_BYTES);
  if (!native_region)
    svh_fail(852, "reservation");
  *base = (uint64_t)(uintptr_t)native_region;
  *capacity = NATIVE_REGION_BYTES / bytes;
}

static int svh_backend_carve(size_t g, struct svh_slot *s) {
  (void)s;
  SVH_UNPOISON(native_region + g * svh.bytes, svh.bytes);
  return 1;
}

static void *svh_backend_issue(size_t g, struct svh_slot *s) {
  (void)s;
  unsigned char *head = native_region + g * svh.bytes;
  SVH_UNPOISON(head, svh.bytes);
  return head;
}

static void native_publish_oldest(void) {
  size_t g = native_fifo[native_head];
  native_head = (native_head + 1) % NATIVE_DELAY;
  --native_count;
  ++native_published;
  SVH_UNPOISON(native_region + g * svh.bytes, svh.bytes);
  memset(native_region + g * svh.bytes, 0, svh.bytes);
  svh_publish(g);
}

static void svh_backend_release(size_t g, struct svh_slot *s) {
  (void)s;
  if (!svh.mode) {
    svh_publish(g);
    return;
  }
  if (native_count == NATIVE_DELAY)
    native_publish_oldest();
  SVH_POISON(native_region + g * svh.bytes, svh.bytes);
  native_fifo[(native_head + native_count++) % NATIVE_DELAY] = (uint32_t)g;
}

static int svh_backend_current(struct svh_slot *s, const void *head) {
  return s->client == head;
}

static uint64_t svh_backend_address(const void *head) {
  return (uint64_t)(uintptr_t)head;
}

static void svh_backend_teardown(void) {}

static void svh_backend_report(struct svh_line *l) {
  svh_put(l, "delayed", native_count);
  svh_put(l, "published_after_delay", native_published);
#ifdef SVH_ASAN
  svh_put(l, "asan", 1);
#else
  svh_put(l, "asan", 0);
#endif
}
