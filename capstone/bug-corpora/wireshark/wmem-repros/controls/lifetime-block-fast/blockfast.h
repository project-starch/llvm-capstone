/* What the block_fast lifetime controls share: the block_fast allocator's own
 * reference to its block, the reset that releases A, and the successor
 * sequence.
 *
 * These are Wireshark's row of the pre-registration
 * docs/plans/2026-10-12-nested-lifetime-controls-extended.md. They run on a
 * fresh wmem_allocator_new(WMEM_ALLOCATOR_BLOCK_FAST). block_fast's
 * wmem_free is a no-op on both builds, so the only release it has is the
 * reset, wmem_free_all; every control here releases A that way. The sibling
 * control has no meaning on block_fast (the reset ends every object at once)
 * and is not run: it is reported as not applicable.
 *
 * The labelled accesses, the patterns, the build check (wm_ctl_arm) and the
 * printed lines are the 2026-10-11 controls' own, from ../lifetime/lifetime.h,
 * unchanged. That header's struct restatements are of the BLOCK allocator and
 * are not used here.
 *
 * WHICH REFERENCE IS "THE PARENT". block_fast keeps its blocks on
 * allocator->block_list as wmem_block_fast_hdr_t pointers: the capability
 * the system allocator returned for the 2 MiB block (wmem_block_fast_new_block,
 * `block = wmem_alloc(NULL, WMEM_BLOCK_SIZE)`). Upstream's alloc carves every
 * chunk from it by address arithmetic (`block_list + block_list->pos`), and
 * upstream's reset rewinds `pos` through it. That is the reference control 00
 * uses first, on both builds.
 *
 * Patch 0001 keeps two more in the block header (wmem_sublet_block_t): `life`,
 * CDERIVEd from the block, and `generation`, CDERIVEd from life. Every object
 * is a CDERIVE child of `generation`. The reset keeps the first block and
 * renews it (wmem_sublet_block_renew): CREVOKE(life, generation), which ends
 * the generation and every object below it, then a NEW generation CDERIVEd
 * from life. So after the reset, `life` is the revoked generation's parent
 * (the CREVOKE source), and `generation` is the new generation the next object
 * will be derived from. Control 00 uses both, on the protected build only.
 *
 * The layouts below restate wmem_allocator_block_fast.c at the 4.6.8 pin
 * (wmem_block_fast_hdr_t, wmem_block_fast_chunk_t,
 * wmem_block_fast_allocator_t) and patch 0001's addition to the block header.
 */
#ifndef WM_BLOCK_FAST_CONTROLS_H
#define WM_BLOCK_FAST_CONTROLS_H

#include "../lifetime/lifetime.h"

#define WM_BF_BLOCK_SIZE (2UL * 1024 * 1024)

struct wm_bf_hdr {
  struct wm_bf_hdr *next;
  int32_t pos;
#if WM_CTL_PROTECTED
  wmem_sublet_block_t sublet; /* patch 0001: life, generation */
#endif
};
struct wm_bf_chunk {
  uint32_t len;
};
struct wm_bf_allocator {
  struct wm_bf_hdr *block_list;
  void *jumbo_list;
};
#define WM_BF_ALIGN(n) (((n) + WM_CTL_ALIGN - 1) & ~(WM_CTL_ALIGN - 1))
#define WM_BF_HEADER_SIZE WM_BF_ALIGN(sizeof(struct wm_bf_hdr))
#define WM_BF_CHUNK_HEADER WM_BF_ALIGN(sizeof(struct wm_bf_chunk))

/* The block_fast allocator's own reference to its (only) block. */
static inline struct wm_bf_hdr *wm_bf_block(wmem_allocator_t *al) {
  struct wm_bf_allocator *bf = (struct wm_bf_allocator *)al->private_data;
  CHECK(bf->block_list && !bf->block_list->next, 0xbf100001);
  CHECK(!bf->jumbo_list, 0xbf100002);
  return bf->block_list;
}

/* A fresh BLOCK_FAST allocator, A as its first object and a second object
 * after it, both filled. A is the block's first chunk, so its data starts at
 * the block header plus one chunk header; the second object moves `pos` past
 * A, so an address equal to A's afterwards can only come from the reset. */
static inline unsigned char *wm_bf_setup(wmem_allocator_t **out,
                                         struct wm_bf_hdr **block_out) {
  wmem_allocator_t *al = wmem_allocator_new(WMEM_ALLOCATOR_BLOCK_FAST);
  CHECK(al, 0xbf200001);
  unsigned char *a = wmem_alloc(al, WM_CTL_N);
  CHECK(a, 0xbf200002);
  wm_ctl_arm(a, WM_CTL_N);
  unsigned char *second = wmem_alloc(al, WM_CTL_N);
  CHECK(second, 0xbf200003);
  struct wm_bf_hdr *block = wm_bf_block(al);
  unsigned long offset = WM_CTL_CURSOR(a) - WM_CTL_CURSOR(block);
  CHECK(offset == WM_BF_HEADER_SIZE + WM_BF_CHUNK_HEADER, 0xbf200004);
  CHECK(WM_CTL_CURSOR(second) == WM_CTL_CURSOR(a) + WM_CTL_N + WM_BF_CHUNK_HEADER,
        0xbf200005);
  wm_ctl_fill(a, WM_CTL_N, wm_ctl_pattern_a);
  wm_ctl_fill(second, WM_CTL_N, wm_ctl_pattern_b);
  wm_ctl_show("A", a);
  wm_ctl_show("second", second);
  wm_ctl_show("block", block);
#if WM_CTL_PROTECTED
  wm_ctl_show("life", block->sublet.life);
  wm_ctl_show("generation-before-reset", block->sublet.generation);
#endif
  printf("WM_CTL A_offset_in_block=%#lx block_header=%lu chunk_header=%lu "
         "pos_before_reset=%d\n",
         offset, (unsigned long)WM_BF_HEADER_SIZE,
         (unsigned long)WM_BF_CHUNK_HEADER, (int)block->pos);
  *out = al;
  *block_out = block;
  return a;
}

/* The release: wmem_free_all. Patch 0001: CREVOKE(life, generation), which
 * ends A and the second object, then a new generation from life; upstream's
 * reset rewinds pos to the header. */
static inline void wm_bf_release(wmem_allocator_t *al) {
  wmem_free_all(al);
  printf("WM_CTL released A via wmem_free_all\n");
}

/* LIFETIME CONTROL 3's sequence, shared by its two programs so they differ
 * only in what follows it: A released by the reset, then C allocated at A's
 * size until wmem returns A's start address, at most WM_CTL_TRIES times,
 * every miss kept live. C is written and read back through C. Returns C;
 * *stale is the old A. */
static unsigned char *wm_bf_successor(unsigned char **stale) {
  wmem_allocator_t *al;
  struct wm_bf_hdr *block;
  unsigned char *a = wm_bf_setup(&al, &block);
  unsigned long a_at = WM_CTL_CURSOR(a);

  wm_bf_release(al);

  unsigned char *c = NULL;
  int tries = 0;
  while (tries < WM_CTL_TRIES) {
    c = wmem_alloc(al, WM_CTL_N);
    CHECK(c, 0xbf300001);
    tries++;
    printf("WM_CTL alloc try=%d addr=%#lx\n", tries, WM_CTL_CURSOR(c));
    if (WM_CTL_CURSOR(c) == a_at)
      break;
  }
  int same = c && WM_CTL_CURSOR(c) == a_at;
  printf("WM_CTL successor A=%#lx C=%#lx tries=%d equal=%s\n", a_at,
         c ? WM_CTL_CURSOR(c) : 0UL, tries, same ? "yes" : "no");
  CHECK(same, 0xbf300002); /* without equality the control says nothing */
  wm_ctl_show("C", c);
#if WM_CTL_PROTECTED
  wm_ctl_show("generation-after-reset", block->sublet.generation);
#endif

  unsigned char got[WM_CTL_N];
  wm_ctl_write("C", "C", c, WM_CTL_N, wm_ctl_pattern_c, WM_CTL_AT("blockfast.h"));
  wm_ctl_read("C", "C", c, WM_CTL_N, got, WM_CTL_AT("blockfast.h"));
  wm_ctl_expect("C-written", got, 0, WM_CTL_N, wm_ctl_pattern_c);
  *stale = a;
  return c;
}

#endif
