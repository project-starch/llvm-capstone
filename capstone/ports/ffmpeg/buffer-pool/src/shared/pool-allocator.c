/* Exact-size payload classes and out-of-band metadata. All comparison arms
 * share the port and layout; only the lifetime discipline changes at runtime.
 * Mode 0: spatial bounds. Mode 1: backing-allocation revocation. Mode 2:
 * Sublet, also revoking each last return to a pool, before reissuing that same
 * storage.
 */
#include "libavutil/mem.h"
#include "metadata-allocator.h"
#include "payload-backend.h"
#ifdef FFPOOL_STUDY_GAPS
#include <stdio.h>
#endif
#ifdef FFPOOL_APP_QUARANTINE
/* Match this provision in both platforms. The published queue alone can hold
 * 4096 unavailable entries; the old extraction's 2048 records cannot fit it. */
#define PAYLOAD_BLOCKS 8192
#else
#define PAYLOAD_BLOCKS 2048
#endif
static struct payload_block payload_blocks[PAYLOAD_BLOCKS];
static unsigned mode, nblocks;
static uintptr_t payload_base;
static size_t payload_capacity, payload_used;

#ifdef FFPOOL_STUDY_GAPS
/* An in-process observer over the real pool leases. It stores indices and
 * allocation numbers, never aliases to released application payloads. */
static uint64_t gap_issues, gap_reuses, gap_last_release[PAYLOAD_BLOCKS], gap_bins[32];
static unsigned gap_index(const struct payload_block *b) {
  ptrdiff_t i = b - payload_blocks;
  if (i < 0 || i >= PAYLOAD_BLOCKS)
    ff2_fail(307);
  return (unsigned)i;
}
static void gap_issue(const struct payload_block *b) {
  unsigned index = gap_index(b);
  if (gap_issues == UINT64_MAX)
    ff2_fail(308);
  gap_issues++;
  if (gap_last_release[index]) {
    uint64_t distance = gap_issues - gap_last_release[index];
    unsigned bucket = 0;
    if (!distance)
      ff2_fail(309);
    while (distance > 1) { distance >>= 1; bucket++; }
    if (bucket >= 32)
      ff2_fail(310);
    gap_bins[bucket]++;
    gap_reuses++;
    gap_last_release[index] = 0;
  }
}
static void gap_release(const struct payload_block *b) {
  if (!gap_issues)
    ff2_fail(311);
  gap_last_release[gap_index(b)] = gap_issues;
}
void ff2_reuse_report(void) {
#if defined(FFPOOL_CHERI) || defined(FFPOOL_STUDY_GAPS_STDERR)
#define GAP_PRINT(...) fprintf(stderr, __VA_ARGS__)
#else
#define GAP_PRINT(...) printf(__VA_ARGS__)
#endif
  GAP_PRINT("FF2-GAP-TOTAL issues=%llu reuses=%llu observer=%llu\n",
            (unsigned long long)gap_issues, (unsigned long long)gap_reuses,
            (unsigned long long)(sizeof(gap_issues)+sizeof(gap_reuses)+
                                 sizeof(gap_last_release)+sizeof(gap_bins)));
  for (unsigned i = 0; i < 32; i += 2)
    GAP_PRINT("FF2-GAP pair=%u a=%llu b=%llu\n", i/2,
              (unsigned long long)gap_bins[i],
              (unsigned long long)gap_bins[i+1]);
#undef GAP_PRINT
}
#else
void ff2_reuse_report(void) { }
#endif

void ff2_set_mode(unsigned value) {
  if (value > 2)
    ff2_fail(301);
  mode = value;
}
void ff2_pool_init_region(uintptr_t base, size_t capacity) {
  payload_base = base;
  payload_capacity = capacity;
}
static void *issue(struct payload_block *b, int observed) {
  void *p = ff2_payload_issue_pointer(b, mode);
  b->alias = p;
  b->idle = 0;
#ifdef FFPOOL_STUDY_GAPS
  b->observed_lease = observed;
  if (observed) gap_issue(b);
#else
  (void)observed;
#endif
  return p;
}
static int reusable(const struct payload_block *b) {
  (void)b;
  return 1;
}
static struct payload_block *by_address(uintptr_t address) {
  for (unsigned i = 0; i < nblocks; i++)
    if (payload_blocks[i].alive && payload_blocks[i].address == address)
      return &payload_blocks[i];
  ff2_fail(303);
}
static struct payload_block *by_authority(const void *p) {
  struct payload_block *b = by_address((uintptr_t)p);
  if (b->idle || !ff2_payload_same_authority(p, b->alias))
    ff2_fail(304);
  return b;
}
void *ff2_payload_alloc(size_t size) {
  if (!size || size > payload_capacity || size > SIZE_MAX - 63)
    return NULL;
  size_t rounded = (size + 63) & ~(size_t)63;
  size_t alignment = 64;
#ifdef FFPOOL_CHERI
  /* Compressed bounds may need additional alignment and tail padding.
   * Keep the allocator policy, but account for the actual CHERI geometry. */
  size_t cheri_alignment =
      ~__builtin_cheri_representable_alignment_mask(size) + 1;
  if (cheri_alignment > alignment)
    alignment = cheri_alignment;
  size_t representable = __builtin_cheri_round_representable_length(size);
  if (representable > rounded)
    rounded = representable;
#endif
  struct payload_block *b = NULL;
  for (unsigned i = 0; i < nblocks; i++)
    if (!payload_blocks[i].alive && payload_blocks[i].rounded == rounded &&
        reusable(&payload_blocks[i])) {
      b = &payload_blocks[i];
      break;
    }
  if (!b) {
    size_t padding = (-(size_t)(payload_base + payload_used)) & (alignment - 1);
    if (nblocks == PAYLOAD_BLOCKS || padding > payload_capacity - payload_used ||
        rounded > payload_capacity - payload_used - padding)
      return NULL;
    payload_used += padding;
    b = &payload_blocks[nblocks++];
    b->address = payload_base + payload_used;
    b->rounded = rounded;
    ff2_payload_carve(b, payload_used);
    payload_used += rounded;
  }
  b->requested = size;
  b->alive = 1;
  b->idle = 1;
  ff2_payload_prepare_backing(b, mode);
  return issue(b, 1);
}
void ff2_payload_return(void *p) {
  struct payload_block *b = by_authority(p);
  ff2_payload_return_lease(b, mode);
  b->idle = 1;
#ifdef FFPOOL_STUDY_GAPS
  if (b->observed_lease) gap_release(b);
  b->observed_lease = 0;
#endif
}
int ff2_payload_reusable(uintptr_t address) {
  struct payload_block *b = by_address(address);
  return b->idle && reusable(b);
}
static void *teardown(struct payload_block *b) {
  return issue(b, 0);
}
void *ff2_payload_teardown(uintptr_t address) {
  struct payload_block *b = by_address(address);
  if (!b->idle) ff2_fail(305);
  return teardown(b);
}
void *ff2_payload_issue(uintptr_t address) {
  struct payload_block *b = by_address(address);
  if (!b->idle)
    ff2_fail(305);
  return issue(b, 1);
}
void ff2_payload_free(void *p) {
  if (!p)
    return;
  struct payload_block *b = by_authority(p);
  ff2_payload_free_backing(b, mode);
#ifdef FFPOOL_STUDY_GAPS
  if (b->observed_lease) gap_release(b);
  b->observed_lease = 0;
#endif
  b->alive = 0;
  b->idle = 1;
}
void *ff2_ref_alloc(size_t size, size_t metadata_size) {
  void *meta = av_mallocz(metadata_size);
  if (!meta)
    return NULL;
  void *p = ff2_payload_alloc(size);
  if (!p) {
    av_free(meta);
    return NULL;
  }
  by_authority(p)->meta = meta;
  return meta;
}
static struct payload_block *by_meta(void *meta) {
  for (unsigned i = 0; i < nblocks; i++)
    if (payload_blocks[i].alive && payload_blocks[i].meta == meta)
      return &payload_blocks[i];
  ff2_fail(306);
}
void *ff2_ref_meta(const void *p) { return by_authority(p)->meta; }
void *ff2_ref_data(void *meta) {
  struct payload_block *b = by_meta(meta);
  /* Trusted free-entry callbacks may inspect persistent fields in an idle
   * entry. Give them fresh authority, never revive the application's alias. */
  /* The manager exposes an idle entry to its own callback or initializer. */
  return b->idle ? teardown(b) : b->alias;
}
int ff2_ref_reusable(void *meta) {
  struct payload_block *b = by_meta(meta);
  return b->idle && reusable(b);
}
void *ff2_ref_issue(void *meta) {
  struct payload_block *b = by_meta(meta);
  return ff2_payload_issue(b->address);
}
void ff2_ref_return(void *meta) {
  struct payload_block *b = by_meta(meta);
  ff2_payload_return(b->alias);
}
void ff2_ref_free(void *meta) {
  if (!meta)
    return;
  struct payload_block *b = by_meta(meta);
  /* Reacquire solely for trusted teardown; this is not an application lease. */
  void *p = b->idle ? teardown(b) : b->alias;
  ff2_payload_free(p);
  b->meta = NULL;
  av_free(meta);
}
void ff2_memory_report(struct ff2_header *h) {
  h->metadata_used = ff2_metadata_used();
  h->payload_used = payload_used;
  ff2_payload_report_stats(h);
}
