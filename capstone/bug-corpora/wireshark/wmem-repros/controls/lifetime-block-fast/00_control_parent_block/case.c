/* LIFETIME CONTROL 1 on block_fast, the parent stays usable. A is the first
 * object of a fresh BLOCK_FAST allocator, a second object follows it, and
 * wmem_free_all releases both. A's former range is then read and written
 * through the allocator's own reference to the block that holds it
 * (blockfast.h says which reference that is, and why).
 *
 *   via=block       the wmem_block_fast_hdr_t pointer on
 *                   allocator->block_list, the system allocator's capability
 *                   for the 2 MiB block, moved to A's offset: the pointer
 *                   upstream's alloc carves chunks from. Both builds.
 *   via=generation  patch 0001's block->sublet.generation AFTER the reset:
 *                   the new generation the reset derived, from which the next
 *                   object is derived. Protected build only.
 *   via=life        patch 0001's block->sublet.life: the parent of the
 *                   revoked generation, the reset's CREVOKE source.
 *                   Protected build only.
 *
 * The read first checks what the allocator itself left there. block_fast's
 * reset only rewinds `pos` (and, on the protected build, renews the
 * generation): the block header's pos is back at the header size, A's chunk
 * header still carries A's length, and all 64 bytes still hold A's pattern.
 * Then the range is overwritten and read back. Nothing is allocated
 * afterwards.
 *
 * Predicted on both builds: every access completes. */
#include "../blockfast.h"

WM_CASE(0) {
  wmem_allocator_t *al;
  struct wm_bf_hdr *block;
  unsigned char *a = wm_bf_setup(&al, &block);
  unsigned long offset = WM_CTL_CURSOR(a) - WM_CTL_CURSOR(block);

  wm_bf_release(al);

  unsigned char *parent = (unsigned char *)block + offset;
  unsigned char got[WM_CTL_N];
  int32_t pos;
  wm_ctl_read("block-pos", "block", &block->pos, sizeof pos, (unsigned char *)&pos,
              WM_CTL_AT("case.c"));
  printf("WM_CTL block-pos pos=%d%s\n", (int)pos,
         (unsigned long)pos == WM_BF_HEADER_SIZE ? " ok: the reset rewound pos to the header"
                                                 : " UNEXPECTED");
  struct wm_bf_chunk header;
  wm_ctl_read("A-chunk-header", "block", parent - WM_BF_CHUNK_HEADER, sizeof header,
              (unsigned char *)&header, WM_CTL_AT("case.c"));
  printf("WM_CTL A-chunk-header len=%u%s\n", (unsigned)header.len,
         header.len == WM_CTL_N ? " ok: the reset left A's chunk header" : " UNEXPECTED");
  wm_ctl_read("A-former-range", "block", parent, WM_CTL_N, got, WM_CTL_AT("case.c"));
  wm_ctl_expect("A-former-range-after-reset", got, 0, WM_CTL_N, wm_ctl_pattern_a);
  wm_ctl_write("A-former-range", "block", parent, WM_CTL_N, wm_ctl_pattern_w,
               WM_CTL_AT("case.c"));
  wm_ctl_read("A-former-range", "block", parent, WM_CTL_N, got, WM_CTL_AT("case.c"));
  wm_ctl_expect("A-former-range-written", got, 0, WM_CTL_N, wm_ctl_pattern_w);

#if WM_CTL_PROTECTED
  void *generation = block->sublet.generation;
  wm_ctl_show("generation-after-reset", generation);
  unsigned char *direct = (unsigned char *)generation +
                          (WM_CTL_CURSOR(parent) - WM_CTL_CURSOR(generation));
  wm_ctl_read("A-former-range", "generation", direct, WM_CTL_N, got, WM_CTL_AT("case.c"));
  wm_ctl_expect("A-former-range-via-generation", got, 0, WM_CTL_N, wm_ctl_pattern_w);
  wm_ctl_write("A-former-range", "generation", direct, WM_CTL_N, wm_ctl_pattern_v,
               WM_CTL_AT("case.c"));
  wm_ctl_read("A-former-range", "block", parent, WM_CTL_N, got, WM_CTL_AT("case.c"));
  wm_ctl_expect("A-former-range-written-via-generation", got, 0, WM_CTL_N,
                wm_ctl_pattern_v);

  void *life = block->sublet.life;
  wm_ctl_show("life", life);
  unsigned char *source = (unsigned char *)life +
                          (WM_CTL_CURSOR(parent) - WM_CTL_CURSOR(life));
  wm_ctl_read("A-former-range", "life", source, WM_CTL_N, got, WM_CTL_AT("case.c"));
  wm_ctl_expect("A-former-range-via-life", got, 0, WM_CTL_N, wm_ctl_pattern_v);
  wm_ctl_write("A-former-range", "life", source, WM_CTL_N, wm_ctl_pattern_b,
               WM_CTL_AT("case.c"));
  wm_ctl_read("A-former-range", "block", parent, WM_CTL_N, got, WM_CTL_AT("case.c"));
  wm_ctl_expect("A-former-range-written-via-life", got, 0, WM_CTL_N, wm_ctl_pattern_b);
#else
  printf("WM_CTL generation absent: the release build has no patch 0001\n");
#endif
}
