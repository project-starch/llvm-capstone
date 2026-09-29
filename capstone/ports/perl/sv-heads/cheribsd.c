/* CheriBSD PoisonCap backend of the SV-head adapter.
 *
 * Heads come from one anonymous mapping, the same 32 MiB reservation the
 * Capstone arm is granted, carved in upstream's page order. The adapter keeps
 * the mapping's capability, which carries PoisonCap's POISON and SW_VMEM
 * permissions; Perl receives each head bounded exactly and with both
 * permissions removed, so its copies are revocable.
 *
 * PERL_POISONCAP_MODE=0 is the spatial control: a released head is published
 * at once, as upstream does. PERL_POISONCAP_MODE=1 is PoisonCap: a released
 * head is poisoned and queued, and it becomes issuable only after a
 * revocation sweep, when every stale copy has been invalidated. The queue
 * transfers the published SQLite policy (common/include/
 * poisoncap-quarantine-policy.h): 4,096 entries, or held spans of at least
 * 16 MiB with at least a quarter quarantined. A full queue is swept before
 * the next entry is added (the corrected full-queue drain). One binary
 * carries both modes. */
#include <cheri/cheric.h>
#include <cheri/revoke.h>
#include <sys/mman.h>
#include "poisoncap-quarantine-policy.h"

#ifndef CHERI_PERM_POISON
#error "Use the published PoisonCap SDK and matching kernel"
#endif

#define SVH_PLATFORM "cheribsd-poisoncap"
#define SVH_BACKEND_SLOT
#include "sv-heads.h"

#define REGION_BYTES (32UL << 20)
static unsigned char *region;
static uint32_t queue[POISONCAP_QUARANTINE_ENTRIES];
static size_t queued, peak_queued;
static uint64_t sweeps, full_drains, threshold_drains, teardown_drains;
static uint64_t poison_bytes, clear_bytes, zero_bytes;

static void svh_backend_init(size_t bytes, uint64_t *base, size_t *capacity) {
  const char *mode = getenv("PERL_POISONCAP_MODE");
  if (!mode || (mode[0] != '0' && mode[0] != '1') || mode[1])
    svh_fail(831, "select PERL_POISONCAP_MODE=0 or 1");
  if (!feature_present("cheri_caprevoke_poison"))
    svh_fail(832, "poison feature absent");
  svh.mode = (unsigned)(mode[0] - '0');
  void *p = mmap(NULL, REGION_BYTES, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANON, -1, 0);
  if (p == MAP_FAILED)
    svh_fail(833, "mmap");
  region = p;
  if (!cheri_gettag(region) || cheri_getlen(region) < REGION_BYTES ||
      ((uintptr_t)region & 15) ||
      (cheri_getperm(region) & (CHERI_PERM_POISON | CHERI_PERM_SW_VMEM)) !=
          (CHERI_PERM_POISON | CHERI_PERM_SW_VMEM))
    svh_fail(834, "region authority or alignment");
  *base = cheri_getaddress(region);
  *capacity = REGION_BYTES / bytes;
}

static int svh_backend_carve(size_t g, struct svh_slot *s) {
  (void)g;
  (void)s;
  return 1;
}

/* The adapter's own pointer to a head: exact bounds, full permissions. */
static unsigned char *manager(size_t g) {
  unsigned char *m = cheri_setboundsexact(region + g * svh.bytes, svh.bytes);
  if (!cheri_gettag(m) || cheri_getlen(m) != svh.bytes)
    svh_fail(835, "head bounds");
  return m;
}

static void *svh_backend_issue(size_t g, struct svh_slot *s) {
  (void)s;
  return cheri_clearperm(manager(g), CHERI_PERM_POISON | CHERI_PERM_SW_VMEM);
}

static void drain(void) {
  if (!queued)
    return;
  struct cheri_revoke_syscall_info info = {0};
  if (cheri_revoke(CHERI_REVOKE_LAST_PASS | CHERI_REVOKE_IGNORE_START |
                   CHERI_REVOKE_TAKE_STATS, 0, &info))
    svh_fail(836, "revocation failed");
  ++sweeps;
  /* Publish in release order: the last head queued is issued first, as it
     would be from upstream's LIFO list had it been freed now. */
  for (size_t j = 0; j < queued; ++j) {
    unsigned char *m = manager(queue[j]);
    for (size_t offset = 0; offset < svh.bytes; offset += 16) {
      void *word = m + offset;
      __asm__ volatile("cclearpoison %0, 0(%0)" : : "C"(word) : "memory");
    }
    clear_bytes += svh.bytes;
    memset(m, 0, svh.bytes);
    zero_bytes += svh.bytes;
    svh_publish(queue[j]);
  }
  queued = 0;
}

static void svh_backend_release(size_t g, struct svh_slot *s) {
  (void)s;
  if (!svh.mode) {
    svh_publish(g);
    return;
  }
  if (queued == POISONCAP_QUARANTINE_ENTRIES) {
    ++full_drains;
    drain();
  }
  unsigned char *m = manager(g);
  for (size_t offset = 0; offset < svh.bytes; offset += 16) {
    void *word = m + offset;
    __asm__ volatile("cpoison %0, 0(%0)" : : "C"(word) : "memory");
  }
  poison_bytes += svh.bytes;
  queue[queued++] = (uint32_t)g;
  if (queued > peak_queued)
    peak_queued = queued;
  size_t held = (size_t)(svh.live + queued) * svh.bytes;
  if (poisoncap_quarantine_threshold(held, queued * svh.bytes)) {
    ++threshold_drains;
    drain();
  }
}

static int svh_backend_current(struct svh_slot *s, const void *head) {
  return cheri_gettag(head) && __builtin_cheri_equal_exact(head, s->client);
}

static uint64_t svh_backend_address(const void *head) {
  return cheri_getaddress(head);
}

/* The process is ending: sweep what is still queued, so the report closes
 * with nothing quarantined. */
static void svh_backend_teardown(void) {
  if (queued) {
    ++teardown_drains;
    drain();
  }
}

static void svh_backend_report(struct svh_line *l) {
  svh_put(l, "region_bytes", REGION_BYTES);
  svh_put(l, "sweeps", sweeps);
  svh_put(l, "full_drains", full_drains);
  svh_put(l, "threshold_drains", threshold_drains);
  svh_put(l, "teardown_drains", teardown_drains);
  svh_put(l, "queued", queued);
  svh_put(l, "peak_queued", peak_queued);
  svh_put(l, "poison_bytes", poison_bytes);
  svh_put(l, "clear_bytes", clear_bytes);
  svh_put(l, "zero_bytes", zero_bytes);
  svh_put(l, "policy", 1);
  svh_put(l, "queue_limit", POISONCAP_QUARANTINE_ENTRIES);
  svh_put(l, "minimum_held", POISONCAP_MIN_HELD_BYTES);
  svh_put(l, "queue_metadata_bytes", sizeof queue);
}
