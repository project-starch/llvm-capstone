/* Capstone backend of the SV-head adapter.
 *
 * The heads live in the program's region 1 (experiments/applications/
 * regions.c hands the SDK's single grant over as that index), which this
 * backend holds linearly and carves in order, one 16-byte-aligned slot per
 * head, as a page of heads is first needed. Each slot keeps its revocation
 * handle in the sidecar (s->region) and Perl receives the delinearized alias.
 *
 * PERL_SUBLET_MODE=0 is the spatial control: every head has its own bounds,
 * and a released head's alias stays valid when the slot is reissued.
 * PERL_SUBLET_MODE=1 is Sublet: a release revokes the slot (sublet_give) and
 * takes a fresh alias for the next lease, so every stale copy of the old one
 * is dead before the slot can be issued again. One image carries both, so
 * the two arms differ in the mode alone. Reuse order is identical in both,
 * because Sublet revokes synchronously instead of deferring reuse. */
#include <capstone/capability.h>
#include <sublet/sublet.h>

#define SVH_PLATFORM "capstone"
#define SVH_BACKEND_SLOT capstone_cap_slot region; void *alias;
#include "sv-heads.h"

void *__capstone_region(unsigned index);

static capstone_cap_slot remaining;
static unsigned long region_base, region_end;

static void svh_backend_init(size_t bytes, uint64_t *base, size_t *capacity) {
  const char *mode = getenv("PERL_SUBLET_MODE");
  if (!mode || (mode[0] != '0' && mode[0] != '1') || mode[1])
    svh_fail(821, "select PERL_SUBLET_MODE=0 or 1");
  svh.mode = (unsigned)(mode[0] - '0');
  void *region = __capstone_region(1);
  if (!region)
    svh_fail(822, "no program region 1; link with regions.c and grant a heap");
  capstone_cap_store(&remaining, region);
  if (capstone_cap_type(&remaining) != CAPSTONE_CAP_LINEAR)
    svh_fail(823, "region 1 is not linear");
  region_base = capstone_cap_base(&remaining);
  region_end = capstone_cap_end(&remaining);
  if ((region_base & 15) || region_end <= region_base)
    svh_fail(824, "region 1 alignment");
  *base = region_base;
  *capacity = (region_end - region_base) / bytes;
}

/* Slots are carved strictly in order, so each carve takes the prefix. */
static int svh_backend_carve(size_t g, struct svh_slot *s) {
  unsigned long end = region_base + (unsigned long)(g + 1) * svh.bytes;
  if (end > region_end || capstone_cap_base(&remaining) != end - svh.bytes)
    return 0;
  sublet_carve(&remaining, end, &s->region);
  s->alias = sublet_take(&s->region);
  return 1;
}

static void *svh_backend_issue(size_t g, struct svh_slot *s) {
  (void)g;
  return s->alias;
}

static void svh_backend_release(size_t g, struct svh_slot *s) {
  if (svh.mode) {
    sublet_give(&s->region);
    s->alias = sublet_take(&s->region);
  }
  svh_publish(g);
}

/* Both capabilities' stored words, compared: a stale alias carries the
 * revoked node and differs from the slot's current one. Only the stored
 * bits are read; no metadata query is issued on a possibly revoked alias. */
static int svh_backend_current(struct svh_slot *s, const void *head) {
  capstone_cap_slot x, y;
  capstone_cap_store(&x, (void *)head);
  capstone_cap_store(&y, s->client);
  const volatile uint64_t *xx = (const volatile uint64_t *)&x;
  const volatile uint64_t *yy = (const volatile uint64_t *)&y;
  return xx[0] == yy[0] && xx[1] == yy[1];
}

static uint64_t svh_backend_address(const void *head) {
  return (uint64_t)(uintptr_t)head;
}

static void svh_backend_teardown(void) {}

static void svh_backend_report(struct svh_line *l) {
  svh_put(l, "region_bytes", region_end - region_base);
  svh_put(l, "split", sublet_stats.split);
  svh_put(l, "mrev", sublet_stats.mrev);
  svh_put(l, "delin", sublet_stats.delin);
  svh_put(l, "revoke", sublet_stats.revoke);
  svh_put(l, "init", sublet_stats.init);
}
