/* LIFETIME CONTROL 2, a sibling stays usable. A and B are allocated from the
 * same block of a fresh BLOCK allocator, adjacent; A is released with
 * wmem_free; B is then read through B's own pointer, checked byte for byte,
 * overwritten and read back.
 *
 * On the protected build A and B are sibling CDERIVE children of the block's
 * generation, and the release is CREVOKE(generation, A): it must end A and
 * nothing beside it. Upstream's free does not merge A into B (B is in use),
 * so the allocator itself does not write B either.
 *
 * Predicted on both builds: every access completes and B is intact. */
#include "../lifetime.h"

WM_CASE(1) {
  wmem_allocator_t *al = wmem_allocator_new(WMEM_ALLOCATOR_BLOCK);
  CHECK(al, 1);
  unsigned char *a = wmem_alloc(al, WM_CTL_N);
  unsigned char *b = wmem_alloc(al, WM_CTL_N);
  CHECK(a && b, 2);
  wm_ctl_arm(b, WM_CTL_N);
  struct wm_ctl_block_hdr *block = wm_ctl_block(al); /* A and B share it */
  CHECK(WM_CTL_CURSOR(b) == WM_CTL_CURSOR(a) + WM_CTL_N + WM_CTL_CHUNK_HEADER, 3);
  wm_ctl_fill(a, WM_CTL_N, wm_ctl_pattern_a);
  wm_ctl_fill(b, WM_CTL_N, wm_ctl_pattern_b);
  wm_ctl_show("A", a);
  wm_ctl_show("B", b);
  wm_ctl_show("block", block);

  wmem_free(al, a); /* patch 0001: CREVOKE(generation, A), then upstream's free */
  printf("WM_CTL released A via wmem_free\n");

  unsigned char got[WM_CTL_N];
  wm_ctl_read("B", "B", b, WM_CTL_N, got, WM_CTL_AT("case.c"));
  wm_ctl_expect("B-intact-after-A-released", got, 0, WM_CTL_N, wm_ctl_pattern_b);
  wm_ctl_write("B", "B", b, WM_CTL_N, wm_ctl_pattern_w, WM_CTL_AT("case.c"));
  wm_ctl_read("B", "B", b, WM_CTL_N, got, WM_CTL_AT("case.c"));
  wm_ctl_expect("B-written", got, 0, WM_CTL_N, wm_ctl_pattern_w);
}
