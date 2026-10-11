/* row25 / sqlite-mem5-design -- MEMSYS5 stores freelist metadata INSIDE freed
 * blocks (src/mem5.c: memsys5Link/Unlink write Mem5Link{next,prev} into the freed
 * block; CTRL_FREE is out-of-band in aCtrl[]). A dangling pointer into a block
 * freed back to the pool then reads/overwrites allocator metadata. Design-inherent,
 * never patched (paper: PoisonCap). Present in 3.22.0 (mem5.c:143 MEM5LINK).
 *
 * CONTROL arm: allocate two adjacent blocks, keep a dangling pointer to the first,
 * free it, force a neighbour free so memsys5 links/coalesces (writing in-band
 * metadata), then READ through the dangling pointer. The bytes read are now
 * allocator freelist indices, not the caller's data -- the aliasing the paper
 * names. On unprotected Capstone (whole pool is one allocation, invisible to ASan)
 * this READ SUCCEEDS and the domain RETURNS. Sublet would revoke the freed block.
 */
#include "repro322_common.h"

static int run_case(void) {
  if (repro_init()) return 1;

  /* Two blocks from memsys5. p is the one we will dangle into. */
  volatile unsigned int *p = (unsigned int *)sqlite3_malloc(128);
  void *q = sqlite3_malloc(128);
  if (!p || !q) return FAILRC("mem5 alloc", SQLITE_NOMEM);

  /* caller's data pattern */
  for (int i = 0; i < 32; i++) p[i] = 0xA5A50000u | (unsigned)i;
  unsigned int before = p[0];

  /* Free p back to the pool. memsys5 marks CTRL_FREE in aCtrl[] and writes
   * Mem5Link{next,prev} into p's first bytes (the in-band metadata). */
  sqlite3_free((void *)p);
  /* Free q too, adjacent, so memsys5 coalesces -- link/unlink rewrite the
   * in-band metadata of the merged free block, i.e. through *p's storage. */
  sqlite3_free(q);

  /* Dangling read: p still points into the pool; the bytes are now freelist
   * metadata, not 0xA5A5.... On unprotected Capstone this does not trap. */
  /* Reachability probe. The stale read is in THIS source, not in sqlite3.c, so the
   * probe belongs here. It asks memsys5 whether p's block is free at this instant --
   * the one question neither the exit code nor CHERI can answer under an arena
   * allocator, because the capability is the arena's and the arena is still live. */
  LB_HIT(1, (const void *)p, 4);
  unsigned int after = p[0];

  out_text("mem5design before="); out_uint(before);
  out_text(" after="); out_uint(after);
  out_text(after != before ? " (in-band metadata overwrote caller data)\n"
                           : " (unchanged)\n");
  out_text("mem5design NOTRAP done\n");
  return 0;
}

REPRO322_MAIN("mem5design")
