/* Shared Capstone mruby GC-slot observer. Included in a pinned gc.c build.
 * Only integer slot identities and issue numbers are retained; never an alias
 * to a freed object. Both the spatial and Sublet GC builds use these hooks. */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

static uint64_t gc_gap_issues, gc_gap_releases, gc_gap_reissues, gc_gap_near;
static uint64_t gc_gap_bins[32], gc_gap_pages, gc_gap_peak_pages;

static void gc_gap_page_new(void) {
  ++gc_gap_pages;
  if (gc_gap_pages > gc_gap_peak_pages) gc_gap_peak_pages = gc_gap_pages;
}
static void gc_gap_page_free(void) {
  if (!gc_gap_pages) abort();
  --gc_gap_pages;
}
static void gc_gap_issue(mrb_heap_page *page, unsigned i) {
  if (i >= MRB_HEAP_PAGE_SIZE || gc_gap_issues == UINT64_MAX) abort();
  ++gc_gap_issues;
  if (page->study_seen[i >> 3] & (1u << (i & 7))) {
    uint64_t released = page->study_release_at[i];
    if (!released || released >= gc_gap_issues) abort();
    uint64_t gap = gc_gap_issues - released;
    unsigned bin = 63u - (unsigned)__builtin_clzll(gap);
    if (bin > 31) bin = 31;
    ++gc_gap_bins[bin];
    ++gc_gap_reissues;
    if (gap <= 15) ++gc_gap_near;
    page->study_release_at[i] = 0;
  }
  page->study_seen[i >> 3] |= 1u << (i & 7);
}
static void gc_gap_release(mrb_heap_page *page, unsigned i) {
  if (i >= MRB_HEAP_PAGE_SIZE || !gc_gap_issues ||
      !(page->study_seen[i >> 3] & (1u << (i & 7))) ||
      page->study_release_at[i]) abort();
  page->study_release_at[i] = gc_gap_issues;
  ++gc_gap_releases;
}
void mrb_gc_study_report(const char *phase) {
  char line[1100];
  int n = snprintf(line, sizeof line, "MRB_GC_STUDY phase=%s mode=%d ", phase,
#ifdef MRB_CAPSTONE_GC_SUBLET
    1
#else
    0
#endif
  );
  if (n <= 0 || (size_t)n >= sizeof line) return;
  n += snprintf(line+n, sizeof line-(size_t)n,
    "issues=%llu releases=%llu reissues=%llu gap15=%llu ",
    (unsigned long long)gc_gap_issues, (unsigned long long)gc_gap_releases,
    (unsigned long long)gc_gap_reissues, (unsigned long long)gc_gap_near);
  if (n <= 0 || (size_t)n >= sizeof line) return;
  n += snprintf(line+n, sizeof line-(size_t)n,
    "pages=%llu peak_pages=%llu slot_bytes=%zu observer_bytes=%zu ",
    (unsigned long long)gc_gap_pages,
    (unsigned long long)gc_gap_peak_pages,
    sizeof(RVALUE),
    sizeof(((mrb_heap_page *)0)->study_release_at) +
    sizeof(((mrb_heap_page *)0)->study_seen));
  if (n <= 0 || (size_t)n >= sizeof line) return;
#ifdef MRB_CAPSTONE_GC_SUBLET
  size_t page_metadata = sizeof(mrb_heap_page);
#else
  size_t page_metadata = sizeof(mrb_heap_page) - sizeof(((mrb_heap_page *)0)->objects);
#endif
  n += snprintf(line+n, sizeof line-(size_t)n,
    "page_payload_bytes=%zu page_metadata_bytes=%zu\n",
    sizeof(RVALUE) * MRB_HEAP_PAGE_SIZE, page_metadata);
  if (n > 0 && (size_t)n < sizeof line) write(2, line, (size_t)n);
  if (phase[0] != 'e' || phase[1] != 'x') return;
  n = snprintf(line, sizeof line, "MRB_GC_GAPS");
  for (unsigned i = 0; i < 32 && n > 0 && (size_t)n < sizeof line; ++i)
    n += snprintf(line+n, sizeof line-(size_t)n, " b%u=%llu", i,
                  (unsigned long long)gc_gap_bins[i]);
  if (n > 0 && (size_t)n + 1 < sizeof line) {
    line[n++] = '\n';
    write(2, line, (size_t)n);
  }
}
