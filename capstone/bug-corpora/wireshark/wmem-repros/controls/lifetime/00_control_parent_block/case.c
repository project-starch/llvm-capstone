/* LIFETIME CONTROL 1, the parent stays usable. A is allocated from a fresh
 * BLOCK allocator, released with wmem_free, and A's former range is then read
 * and written through the allocator's own reference to the block that holds
 * it (lifetime.h says which reference that is, and why).
 *
 *   via=block       the wmem_block_hdr_t pointer on allocator->block_list, the
 *                   system allocator's capability for the 8 MiB block, moved
 *                   to A's offset: the pointer upstream's consumer functions
 *                   and patch 0001's release use. Both builds.
 *   via=generation  patch 0001's block->sublet.generation, A's direct parent
 *                   lifetime (CDERIVE parent, CREVOKE'd from at wmem_free).
 *                   Protected build only.
 *
 * The read first checks what the allocator itself left there: A's chunk
 * header marked free with A's length, the free-list links naming A's own
 * chunk (the recycler holds only it), and bytes 32..63 still A's, since the
 * allocator writes only the links into a free chunk. Then the range is
 * overwritten and read back. Nothing is allocated afterwards: the overwrite
 * destroys the links on purpose.
 *
 * Predicted on both builds: every access completes. */
#include "../lifetime.h"

WM_CASE(0) {
  wmem_allocator_t *al = wmem_allocator_new(WMEM_ALLOCATOR_BLOCK);
  CHECK(al, 1);
  unsigned char *a = wmem_alloc(al, WM_CTL_N);
  CHECK(a, 2);
  wm_ctl_arm(a, WM_CTL_N);
  /* A guard after A, so the free cannot merge A into the master's tail. */
  unsigned char *guard = wmem_alloc(al, WM_CTL_N);
  CHECK(guard, 3);
  wm_ctl_fill(a, WM_CTL_N, wm_ctl_pattern_a);
  wm_ctl_fill(guard, WM_CTL_N, wm_ctl_pattern_b);
  struct wm_ctl_block_hdr *block = wm_ctl_block(al);
  unsigned long offset = WM_CTL_CURSOR(a) - WM_CTL_CURSOR(block);
  CHECK(offset >= WM_CTL_CHUNK_HEADER && offset < (8UL << 20), 4);
  wm_ctl_show("A", a);
  wm_ctl_show("guard", guard);
  wm_ctl_show("block", block);
  printf("WM_CTL A_offset_in_block=%#lx chunk_header=%lu\n", offset,
         (unsigned long)WM_CTL_CHUNK_HEADER);

  wmem_free(al, a); /* patch 0001: CREVOKE(generation, A), then upstream's free */
  printf("WM_CTL released A via wmem_free\n");

  unsigned char *parent = (unsigned char *)block + offset;
  unsigned char got[WM_CTL_N];
  struct wm_ctl_chunk header;
  wm_ctl_read("A-chunk-header", "block", parent - WM_CTL_CHUNK_HEADER,
              sizeof header, (unsigned char *)&header, WM_CTL_AT("case.c"));
  printf("WM_CTL A-chunk-header used=%u jumbo=%u len=%u prev=%u%s\n",
         (unsigned)header.used, (unsigned)header.jumbo, (unsigned)header.len,
         (unsigned)header.prev,
         !header.used && header.len == WM_CTL_N + WM_CTL_CHUNK_HEADER
             ? " ok: free, A's length"
             : " UNEXPECTED");
  /* The links are pointers; read them as the allocator does, through the same
   * reference, and compare their addresses with A's chunk. */
  printf("WM_CTL access read A-free-links via=block addr=%#lx len=%lu "
         "probe=pointer-load site=%s:%d\n",
         WM_CTL_CURSOR(parent), (unsigned long)(2 * sizeof(void *)), WM_CTL_AT("case.c"));
  void *const *links = (void *const *)parent;
  unsigned long link_prev = WM_CTL_CURSOR(links[0]),
                link_next = WM_CTL_CURSOR(links[1]);
  printf("WM_CTL done read A-free-links\n");
  unsigned long chunk_at = WM_CTL_CURSOR(parent) - WM_CTL_CHUNK_HEADER;
  printf("WM_CTL A-free-links prev=%#lx next=%#lx chunk=%#lx%s\n", link_prev,
         link_next, chunk_at,
         link_prev == chunk_at && link_next == chunk_at
             ? " ok: the recycler holds A's chunk alone"
             : " UNEXPECTED");
  wm_ctl_read("A-former-range", "block", parent, WM_CTL_N, got, WM_CTL_AT("case.c"));
  wm_ctl_expect("A-former-range-after-links", got, 2 * sizeof(void *), WM_CTL_N,
                wm_ctl_pattern_a);
  wm_ctl_write("A-former-range", "block", parent, WM_CTL_N, wm_ctl_pattern_w,
               WM_CTL_AT("case.c"));
  wm_ctl_read("A-former-range", "block", parent, WM_CTL_N, got, WM_CTL_AT("case.c"));
  wm_ctl_expect("A-former-range-written", got, 0, WM_CTL_N, wm_ctl_pattern_w);

#if WM_CTL_PROTECTED
  void *generation = ((struct wm_ctl_block_hdr *)block)->sublet.generation;
  wm_ctl_show("generation", generation);
  unsigned char *direct = (unsigned char *)generation +
                          (WM_CTL_CURSOR(parent) - WM_CTL_CURSOR(generation));
  wm_ctl_read("A-former-range", "generation", direct, WM_CTL_N, got, WM_CTL_AT("case.c"));
  wm_ctl_expect("A-former-range-via-generation", got, 0, WM_CTL_N,
                wm_ctl_pattern_w);
  wm_ctl_write("A-former-range", "generation", direct, WM_CTL_N,
               wm_ctl_pattern_v, WM_CTL_AT("case.c"));
  wm_ctl_read("A-former-range", "block", parent, WM_CTL_N, got, WM_CTL_AT("case.c"));
  wm_ctl_expect("A-former-range-written-via-generation", got, 0, WM_CTL_N,
                wm_ctl_pattern_v);
#else
  printf("WM_CTL generation absent: the release build has no patch 0001\n");
#endif
  /* The guard was never released and nothing above touched it. */
  wm_ctl_read("guard", "guard", guard, WM_CTL_N, got, WM_CTL_AT("case.c"));
  wm_ctl_expect("guard-intact", got, 0, WM_CTL_N, wm_ctl_pattern_b);
}
