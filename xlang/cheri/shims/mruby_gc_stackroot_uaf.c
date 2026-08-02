/* Row 14 — mruby#3596, UAF in mark_context_stack (GC stack-root scanner).
 * asan.txt: heap-use-after-free READ of size 4, mark_context_stack gc.c:556,
 *           49152 bytes inside a 49200-byte region.
 * Freed by: incremental_sweep_phase -> mrb_free -> free, under mrb_full_gc
 *           driven by mrb_obj_alloc (the row's build sets MRB_GC_STRESS).
 *
 * Note the region is a whole 49,200-byte GC heap page, and releasing it is a
 * real free() — the audit finding that mruby's arena does not make CHERI
 * blind here. Registers above the shrunken stack limit are not cleared on
 * return, so a later growth re-covers stale slots that the scanner then reads.
 */
#include "../mock-mruby/mock_mruby.h"
#include <stdint.h>

#define PAGE_BYTES 49200
#define STALE_OFF 49152

static volatile uint32_t sink;

int main(void) {
  mrb_state *mrb = mrb_open(1024);

  char *heap_page = (char *)mrb_gc_alloc(mrb, PAGE_BYTES); /* add_heap */

  /* An uncleared register slot above the shrunken stack limit still holds a
   * raw pointer into the page. */
  volatile char *stale_slot = heap_page + STALE_OFF;

  mrb_full_gc(mrb); /* sweep releases the page */

  sink = *(volatile uint32_t *)stale_slot; /* READ of size 4 */

  mock_report("mruby_gc_stackroot_uaf", "use-after-free-survived");
  mrb_close(mrb);
  return 0;
}
