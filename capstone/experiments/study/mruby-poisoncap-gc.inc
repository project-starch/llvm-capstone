/* Included by the pinned mruby gc.c adaptation, after mrb_heap_page exists.
 * Both modes use the same mmap-backed page geometry and GC observer. Mode 1
 * poisons dead RVALUE slots immediately, keeps their free links out of band,
 * and revokes/clears the batch before any slot can be reissued. */
#include <cheri/cheric.h>
#include <cheri/revoke.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "poisoncap-quarantine-policy.h"

#ifndef CHERI_PERM_POISON
#error "The mruby GC study requires the PoisonCap SDK"
#endif

static struct {
  int mode;
  uint64_t issues, releases, reissues, gap15, gap_bins[32];
  uint64_t pages, peak_pages, quarantine, peak_quarantine;
  uint64_t discarded_quarantine;
  uint64_t sweeps, poison_bytes, clear_bytes, zero_bytes;
  uint64_t full_drains, threshold_drains;
} mrb_gc_study = { .mode = -1 };

static void
mrb_gc_study_fail(const char *why)
{
  fprintf(stderr, "MRB_GC_STUDY refusal=%s\n", why);
  _exit(126);
}

static void
mrb_gc_study_init(void)
{
  if (mrb_gc_study.mode >= 0) return;
  const char *value = getenv("MRB_GC_POISONCAP");
  if (!value || (value[0] != '0' && value[0] != '1') || value[1])
    mrb_gc_study_fail("select MRB_GC_POISONCAP=0 or 1");
  if (!feature_present("cheri_caprevoke_poison"))
    mrb_gc_study_fail("poison feature absent");
  mrb_gc_study.mode = value[0] - '0';
}

static mrb_heap_page *
mrb_gc_study_new_page(void)
{
  mrb_gc_study_init();
  mrb_heap_page *page = mmap(NULL, sizeof(mrb_heap_page),
                            PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (page == MAP_FAILED) return NULL;
  if (!cheri_gettag(page) ||
      (cheri_getperm(page) & (CHERI_PERM_POISON | CHERI_PERM_SW_VMEM)) !=
          (CHERI_PERM_POISON | CHERI_PERM_SW_VMEM) ||
      ((uintptr_t)page->objects & 15) || (sizeof(RVALUE) & 15))
    mrb_gc_study_fail("page authority or slot alignment");
  ++mrb_gc_study.pages;
  if (mrb_gc_study.pages > mrb_gc_study.peak_pages)
    mrb_gc_study.peak_pages = mrb_gc_study.pages;
  return page;
}

static void
mrb_gc_study_free_page(mrb_heap_page *page)
{
  uint64_t discarded = 0;
  for (size_t i = 0; i < sizeof(page->study_quarantine); ++i)
    discarded += (unsigned)__builtin_popcount((unsigned)page->study_quarantine[i]);
  if (discarded > mrb_gc_study.quarantine)
    mrb_gc_study_fail("quarantine accounting");
  mrb_gc_study.quarantine -= discarded;
  mrb_gc_study.discarded_quarantine += discarded;
  if (munmap(page, sizeof(mrb_heap_page)))
    mrb_gc_study_fail("munmap");
  --mrb_gc_study.pages;
}

static size_t
mrb_gc_study_index(mrb_heap_page *page, RVALUE *p)
{
  uintptr_t start = cheri_getaddress(page->objects);
  uintptr_t addr = cheri_getaddress(p);
  if (addr < start || addr - start >= sizeof(page->objects) ||
      (addr - start) % sizeof(RVALUE))
    mrb_gc_study_fail("slot index");
  return (addr - start) / sizeof(RVALUE);
}

static int
mrb_gc_study_quarantined(mrb_heap_page *page, RVALUE *p)
{
  size_t i = mrb_gc_study_index(page, p);
  return (page->study_quarantine[i >> 3] >> (i & 7)) & 1;
}

static void mrb_gc_study_reclaim(mrb_gc *gc);

static void
mrb_gc_study_release(mrb_gc *gc, mrb_heap_page *page, RVALUE *p)
{
  size_t i = mrb_gc_study_index(page, p);
  if (!((page->study_seen[i >> 3] >> (i & 7)) & 1) ||
      mrb_gc_study_quarantined(page, p))
    mrb_gc_study_fail("release state");
  if (mrb_gc_study.mode &&
      mrb_gc_study.quarantine == POISONCAP_QUARANTINE_ENTRIES) {
    /* Published capacity trigger; correct its unswept full-queue drain. */
    ++mrb_gc_study.full_drains;
    mrb_gc_study_reclaim(gc);
  }
  ++mrb_gc_study.releases;
  page->study_release_at[i] = mrb_gc_study.issues;
  if (!mrb_gc_study.mode) return;
  /* obj_free has run. No in-band free-list link is written until revocation. */
  void *bounded = cheri_setboundsexact(p, sizeof(RVALUE));
  if (!cheri_gettag(bounded)) mrb_gc_study_fail("poison slot bounds");
  for (size_t offset = 0; offset < sizeof(RVALUE); offset += 16) {
    void *word = (unsigned char *)bounded + offset;
    __asm__ volatile("cpoison %0, 0(%0)" : : "C"(word) : "memory");
  }
  mrb_gc_study.poison_bytes += sizeof(RVALUE);
  page->study_quarantine[i >> 3] |= 1u << (i & 7);
  ++mrb_gc_study.quarantine;
  if (mrb_gc_study.quarantine > mrb_gc_study.peak_quarantine)
    mrb_gc_study.peak_quarantine = mrb_gc_study.quarantine;
  size_t held = (mrb_gc_study.issues - mrb_gc_study.releases +
                 mrb_gc_study.quarantine) * sizeof(RVALUE);
  if (poisoncap_quarantine_threshold(held,
                                    mrb_gc_study.quarantine * sizeof(RVALUE))) {
    ++mrb_gc_study.threshold_drains;
    mrb_gc_study_reclaim(gc);
  }
}

static void
mrb_gc_study_reclaim(mrb_gc *gc)
{
  if (!mrb_gc_study.mode || !mrb_gc_study.quarantine) return;
  struct cheri_revoke_syscall_info info = {0};
  if (cheri_revoke(CHERI_REVOKE_LAST_PASS | CHERI_REVOKE_IGNORE_START |
                   CHERI_REVOKE_TAKE_STATS, 0, &info))
    mrb_gc_study_fail("revocation failed");
  ++mrb_gc_study.sweeps;
  for (mrb_heap_page *page = gc->heaps; page; page = page->next) {
    for (size_t i = 0; i < MRB_HEAP_PAGE_SIZE; ++i) {
      if (!(page->study_quarantine[i >> 3] & (1u << (i & 7)))) continue;
      RVALUE *p = page->objects + i;
      for (size_t offset = 0; offset < sizeof(RVALUE); offset += 16) {
        void *word = (unsigned char *)p + offset;
        __asm__ volatile("cclearpoison %0, 0(%0)" : : "C"(word) : "memory");
      }
      mrb_gc_study.clear_bytes += sizeof(RVALUE);
      memset(p, 0, sizeof(RVALUE));
      mrb_gc_study.zero_bytes += sizeof(RVALUE);
      p->as.free.tt = MRB_TT_FREE;
      p->as.free.next = page->freelist;
      page->freelist = p;
      page->study_quarantine[i >> 3] &= ~(1u << (i & 7));
      --mrb_gc_study.quarantine;
    }
  }
  if (mrb_gc_study.quarantine)
    mrb_gc_study_fail("unaccounted quarantine");
  gc->free_heaps = NULL;
  for (mrb_heap_page *page = gc->heaps; page; page = page->next)
    if (page->freelist) {
      page->free_next = gc->free_heaps;
      gc->free_heaps = page;
    }
}

static struct RBasic *
mrb_gc_study_issue(mrb_heap_page *page, RVALUE *p)
{
  size_t i = mrb_gc_study_index(page, p);
  if (mrb_gc_study_quarantined(page, p))
    mrb_gc_study_fail("issued quarantined slot");
  ++mrb_gc_study.issues;
  if (page->study_seen[i >> 3] & (1u << (i & 7))) {
    uint64_t gap = mrb_gc_study.issues - page->study_release_at[i];
    if (!page->study_release_at[i] || !gap)
      mrb_gc_study_fail("reissue without release");
    ++mrb_gc_study.reissues;
    if (gap <= 15) ++mrb_gc_study.gap15;
    unsigned bin = 63u - (unsigned)__builtin_clzll(gap);
    if (bin > 31) bin = 31;
    ++mrb_gc_study.gap_bins[bin];
  }
  page->study_seen[i >> 3] |= 1u << (i & 7);
  void *client = cheri_setboundsexact(p, sizeof(RVALUE));
  if (!cheri_gettag(client) || cheri_getlen(client) != sizeof(RVALUE))
    mrb_gc_study_fail("slot bounds");
  return cheri_clearperm(client, CHERI_PERM_POISON | CHERI_PERM_SW_VMEM);
}

void
mrb_gc_study_report(const char *phase)
{
  mrb_gc_study_init();
  fprintf(stderr, "MRB_GC_STUDY phase=%s mode=%d issues=%llu releases=%llu "
          "reissues=%llu gap15=%llu pages=%llu peak_pages=%llu "
          "quarantine=%llu peak_quarantine=%llu sweeps=%llu "
          "poison_bytes=%llu clear_bytes=%llu zero_bytes=%llu "
          "slot_bytes=%zu metadata_bytes=%zu observer_bytes=%zu "
          "page_payload_bytes=%zu page_metadata_bytes=%zu mapped_page_bytes=%zu "
          "discarded_quarantine=%llu policy=1 full_drains=%llu threshold_drains=%llu "
          "quarantine_limit=%u minimum_held=%lu\n",
          phase, mrb_gc_study.mode,
          (unsigned long long)mrb_gc_study.issues,
          (unsigned long long)mrb_gc_study.releases,
          (unsigned long long)mrb_gc_study.reissues,
          (unsigned long long)mrb_gc_study.gap15,
          (unsigned long long)mrb_gc_study.pages,
          (unsigned long long)mrb_gc_study.peak_pages,
          (unsigned long long)mrb_gc_study.quarantine,
          (unsigned long long)mrb_gc_study.peak_quarantine,
          (unsigned long long)mrb_gc_study.sweeps,
          (unsigned long long)mrb_gc_study.poison_bytes,
          (unsigned long long)mrb_gc_study.clear_bytes,
          (unsigned long long)mrb_gc_study.zero_bytes,
          sizeof(RVALUE), sizeof(mrb_heap_page) - sizeof(((mrb_heap_page *)0)->objects),
          sizeof(((mrb_heap_page *)0)->study_seen) +
              sizeof(((mrb_heap_page *)0)->study_release_at),
          sizeof(((mrb_heap_page *)0)->objects),
          sizeof(mrb_heap_page) - sizeof(((mrb_heap_page *)0)->objects),
          (sizeof(mrb_heap_page) + 4095) & ~(size_t)4095,
          (unsigned long long)mrb_gc_study.discarded_quarantine,
          (unsigned long long)mrb_gc_study.full_drains,
          (unsigned long long)mrb_gc_study.threshold_drains,
          POISONCAP_QUARANTINE_ENTRIES, POISONCAP_MIN_HELD_BYTES);
  if (!strcmp(phase, "exit")) {
    fputs("MRB_GC_GAPS", stderr);
    for (unsigned i = 0; i < 32; ++i)
      fprintf(stderr, " b%u=%llu", i,
              (unsigned long long)mrb_gc_study.gap_bins[i]);
    fputc('\n', stderr);
  }
}
