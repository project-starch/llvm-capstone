/* Published SQLite quarantine thresholds transferred to FFmpeg pool leases.
 * AVRefStructPool retains initialized state while an entry is idle, so its
 * payload needs an external capability-preserving snapshot. AVBufferPool does
 * not promise contents across leases and can reuse detoxed storage directly.
 */
#include "payload-backend.h"
#include <cheri/cheric.h>
#include <cheri/revoke.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../../../../common/include/poisoncap-quarantine-policy.h"

#ifndef CHERI_PERM_POISON
#error "PoisonCap requires its matching SDK and poison-enabled kernel"
#endif

static unsigned char *arena;
static unsigned long returned, swept;
static size_t sweeps, poison_bytes, clear_bytes, copied_bytes;
static size_t snapshot_bytes, snapshot_peak;
static size_t held_bytes, quarantine_bytes, quarantine_entries;
static size_t peak_held, peak_quarantine, full_drains, threshold_drains, teardown_drains;
static size_t legacy_reuse_drains;

static void sweep(void) {
  struct cheri_revoke_syscall_info info = {0};
  if (!quarantine_entries || quarantine_bytes > held_bytes) ff2_fail(337);
  if (cheri_revoke(CHERI_REVOKE_LAST_PASS | CHERI_REVOKE_IGNORE_START |
                   CHERI_REVOKE_TAKE_STATS, 0, &info) != 0)
    ff2_fail(333);
  swept = returned;
  ++sweeps;
  held_bytes -= quarantine_bytes;
  quarantine_bytes = quarantine_entries = 0;
}

int ff2_poisoncap_reusable(const struct payload_block *b, unsigned mode) {
#ifndef FFPOOL_APP_QUARANTINE
  (void)b;
  (void)mode;
  return 1;
#else
  return mode == 0 || b->poison_epoch <= swept;
#endif
}

void ff2_poisoncap_teardown(struct payload_block *b, unsigned mode) {
  /* FFmpeg destructor callbacks may read initialized RefStruct state. This
   * necessary port extension is separate from the published batch triggers;
   * ordinary lease requests must never enter it. */
  if (!ff2_poisoncap_reusable(b, mode)) {
    ++teardown_drains;
    sweep();
  }
}

static void check_mode(unsigned mode) {
  if (mode != 0 && mode != 2)
    ff2_fail(330);
}

void ff2_payload_init(void *p, size_t n) {
  if (!feature_present("cheri_caprevoke_poison") || ((uintptr_t)p & 15) ||
      !(cheri_getperm(p) & CHERI_PERM_POISON))
    ff2_fail(331);
  arena = p;
  ff2_pool_init_region((uintptr_t)p, n);
}

void ff2_payload_carve(struct payload_block *b, size_t offset) {
  b->region.c = arena + offset; /* retain the wider manager authority */
}

void ff2_payload_prepare_backing(struct payload_block *b, unsigned mode) {
  check_mode(mode);
  (void)b;
}

void *ff2_payload_issue_pointer(struct payload_block *b, unsigned mode) {
  check_mode(mode);
  if (mode == 2 && b->poison_epoch) {
    /* Selection must skip quarantined entries. Never sweep to make this
     * particular application request reuse its preferred block. */
    if (b->poison_epoch > swept) {
#ifdef FFPOOL_APP_QUARANTINE
      ff2_fail(338);
#else
      /* Preserve the separately documented legacy extraction API. Complete
       * application comparisons always compile FFPOOL_APP_QUARANTINE. */
      ++legacy_reuse_drains;
      sweep();
#endif
    }
    for (size_t i = 0; i < b->rounded; i += 16) {
      void *word = (unsigned char *)b->region.c + i;
      __asm__ volatile("cclearpoison %0, 0(%0)" : : "C"(word) : "memory");
    }
    clear_bytes += b->rounded;
    if (b->outer.c) {
      memcpy(b->region.c, b->outer.c, b->rounded);
      copied_bytes += b->rounded;
    }
    b->poison_epoch = 0;
  }
  if (mode == 2) {
    held_bytes += b->rounded;
    if (held_bytes > peak_held) peak_held = held_bytes;
  }
  /* A previous bounded alias may have been revoked. Always derive anew. */
  void *p = cheri_setbounds(b->region.c, b->requested);
  if (!cheri_gettag(p) || cheri_getbase(p) != cheri_getaddress(b->region.c) ||
      cheri_getlen(p) < b->requested || cheri_getlen(p) > b->rounded)
    ff2_fail(334);
  b->full_alias = cheri_clearperm(p, CHERI_PERM_POISON | CHERI_PERM_SW_VMEM);
  return b->full_alias;
}

int ff2_payload_same_authority(const void *a, const void *b) {
  return __builtin_cheri_equal_exact(a, b);
}

void ff2_payload_return_lease(struct payload_block *b, unsigned mode) {
  check_mode(mode);
  if (mode == 0)
    return;
  if (b->poison_epoch || returned == ~0UL)
    ff2_fail(335);
  if (quarantine_entries == POISONCAP_QUARANTINE_ENTRIES) {
    /* Keep the published capacity trigger, correct its unswept drain. */
    ++full_drains;
    sweep();
  }
  if (b->meta) {
    if (!b->outer.c) {
      b->outer.c = aligned_alloc(16, b->rounded);
      if (!b->outer.c)
        ff2_fail(332);
      snapshot_bytes += b->rounded;
      if (snapshot_bytes > snapshot_peak)
        snapshot_peak = snapshot_bytes;
    }
    memcpy(b->outer.c, b->region.c, b->rounded);
    copied_bytes += b->rounded;
  }
  /* Poison padding too: every exposed compressed bound must be covered. */
  unsigned char *bounded = cheri_setboundsexact(b->region.c, b->rounded);
  if (!cheri_gettag(bounded))
    ff2_fail(336);
  for (size_t i = 0; i < b->rounded; i += 16) {
    void *word = bounded + i;
    __asm__ volatile("cpoison %0, 0(%0)" : : "C"(word) : "memory");
  }
  poison_bytes += b->rounded;
  b->poison_epoch = ++returned;
  ++quarantine_entries;
  quarantine_bytes += b->rounded;
  if (quarantine_bytes > peak_quarantine) peak_quarantine = quarantine_bytes;
  if (poisoncap_quarantine_threshold(held_bytes, quarantine_bytes)) {
    ++threshold_drains;
    sweep();
  }
}

void ff2_payload_free_backing(struct payload_block *b, unsigned mode) {
  /* The final owner is gone. Its initialized state will never be reissued. */
  if (b->outer.c) {
    free(b->outer.c);
    b->outer.c = NULL;
    snapshot_bytes -= b->rounded;
  }
  b->meta = NULL;
  ff2_payload_return_lease(b, mode);
}

void ff2_payload_report_stats(struct ff2_header *report) {
  report->reserved[0] = sweeps;
  report->reserved[1] = poison_bytes;
  report->reserved[2] = snapshot_bytes;
  report->reserved[3] = sizeof(void *);
#ifdef FFPOOL_APP_MEMORY
  fprintf(stderr, "FF2_POISONCAP sweeps=%zu poison_bytes=%zu clear_bytes=%zu "
#else
  printf("FF2_POISONCAP sweeps=%zu poison_bytes=%zu clear_bytes=%zu "
#endif
         "snapshot_bytes=%zu snapshot_peak=%zu copied_bytes=%zu "
#ifdef FFPOOL_APP_QUARANTINE
         "policy=1 "
#else
         "policy=0 "
#endif
         "held=%zu peak_held=%zu quarantine=%zu peak_quarantine=%zu "
         "qcount=%zu full_drains=%zu threshold_drains=%zu teardown_drains=%zu "
         "quarantine_limit=%u minimum_held=%lu legacy_reuse_drains=%zu\n",
         sweeps, poison_bytes, clear_bytes, snapshot_bytes, snapshot_peak,
         copied_bytes, held_bytes, peak_held, quarantine_bytes, peak_quarantine,
         quarantine_entries, full_drains, threshold_drains, teardown_drains,
         POISONCAP_QUARANTINE_ENTRIES, POISONCAP_MIN_HELD_BYTES, legacy_reuse_drains);
}
