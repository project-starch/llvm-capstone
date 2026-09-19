/* Experimental trusted adapter: poison on return, sweep before reissue.
 * Pool contents survive idle periods, unlike malloc's freed payload. Keep a
 * capability-preserving snapshot outside the poisoned storage and account
 * for it explicitly. This is a conservative policy, not optimized quarantine.
 */
#include "payload-backend.h"
#include <cheri/cheric.h>
#include <cheri/revoke.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef CHERI_PERM_POISON
#error "PoisonCap requires its matching SDK and poison-enabled kernel"
#endif

static unsigned char *arena;
static unsigned long returned, swept;
static size_t sweeps, poison_bytes, clear_bytes, copied_bytes, snapshot_bytes;

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
  if (mode == 2 && !b->outer.c) {
    b->outer.c = aligned_alloc(16, b->rounded);
    if (!b->outer.c)
      ff2_fail(332);
    snapshot_bytes += b->rounded;
  }
}

void *ff2_payload_issue_pointer(struct payload_block *b, unsigned mode) {
  check_mode(mode);
  if (mode == 2 && b->poison_epoch) {
    if (b->poison_epoch > swept) {
      struct cheri_revoke_syscall_info info = {0};
      /* Never clear poison or release storage after a failed sweep. */
      if (cheri_revoke(CHERI_REVOKE_LAST_PASS | CHERI_REVOKE_IGNORE_START |
                       CHERI_REVOKE_TAKE_STATS, 0, &info) != 0)
        ff2_fail(333);
      swept = returned;
      ++sweeps;
    }
    for (size_t i = 0; i < b->rounded; i += 16) {
      void *word = (unsigned char *)b->region.c + i;
      __asm__ volatile("cclearpoison %0, 0(%0)" : : "C"(word) : "memory");
    }
    clear_bytes += b->rounded;
    memcpy(b->region.c, b->outer.c, b->rounded);
    copied_bytes += b->rounded;
    b->poison_epoch = 0;
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
  if (b->poison_epoch || !b->outer.c || returned == ~0UL)
    ff2_fail(335);
  memcpy(b->outer.c, b->region.c, b->rounded);
  copied_bytes += b->rounded;
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
}

void ff2_payload_free_backing(struct payload_block *b, unsigned mode) {
  ff2_payload_return_lease(b, mode);
}

void ff2_payload_report_stats(struct ff2_header *report) {
  report->reserved[0] = sweeps;
  report->reserved[1] = poison_bytes;
  report->reserved[2] = snapshot_bytes;
  report->reserved[3] = sizeof(void *);
  printf("FF2_POISONCAP sweeps=%zu poison_bytes=%zu clear_bytes=%zu "
         "snapshot_bytes=%zu copied_bytes=%zu\n", sweeps, poison_bytes,
         clear_bytes, snapshot_bytes, copied_bytes);
}
