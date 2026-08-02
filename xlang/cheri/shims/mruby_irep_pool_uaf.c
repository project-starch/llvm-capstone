/* Row 9 — mruby#3829, UAF in mrb_gc_mark via a shared irep-pool string slice.
 * asan.txt: heap-use-after-free READ of size 4, mrb_gc_mark gc.c:721,
 *           0 bytes inside a 48-byte region.
 * Freed by: obj_free -> mrb_irep_free -> mrb_free -> free, under mrb_full_gc
 *           (an explicit GC.start). GC-driven, but it IS a real free() —
 *           see rows.tsv: mruby's arena does not hide this from revocation.
 */
#include "../mock-mruby/mock_mruby.h"
#include <stdint.h>

static volatile uint32_t sink;

int main(void) {
  mrb_state *mrb = mrb_open(1024);

  /* new_lit -> mrb_str_pool: the literal's buffer in the dynamic irep pool. */
  char *pool_str = (char *)mrb_gc_alloc(mrb, 48);

  /* "..."[1..-2] makes an FSHARED substring: a second object pointing into
   * the SAME buffer. That shared pointer is what outlives the free. */
  volatile char *shared_slice = pool_str;

  /* p = nil; GC.start  ->  the irep is swept and its pool string released. */
  mrb_full_gc(mrb);

  sink = *(volatile uint32_t *)shared_slice; /* READ of size 4 at +0 */

  mock_report("mruby_irep_pool_uaf", "use-after-free-survived");
  mrb_close(mrb);
  return 0;
}
