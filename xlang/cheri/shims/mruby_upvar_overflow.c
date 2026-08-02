/* Row 11 — CVE-2018-10191, OP_GETUPVAR scope-level truncation.
 * SPATIAL, not temporal: the environment it would "revoke" is alive and
 * intact. The temporal path was tested and closed — envadjust() rewrites
 * REnv::stack on every realloc, and mruby closes an environment when a Proc
 * escapes. Bounds, not revocation, are what stop this.
 *
 * asan.txt: heap-buffer-overflow READ of size 16, mrb_vm_exec vm.c:1208,
 *           528 bytes AFTER a 4096-byte region.
 *
 * Mechanism: MKOP_ABC(OP_GETUPVAR, cursp(), idx, lv) is emitted with no range
 * check. C (scope level) is 7 bits, so nesting 129 truncates to 129 & 0x7f
 * == 1: uvenv() walks ONE scope instead of 129 and lands on the wrong
 * environment. B (local index) is 9 bits and still carries the full index, so
 * *regs_a = e->stack[b] reads far past that environment's register array.
 *
 * Both trigger parameters are load-bearing and are enforced statically:
 *   - nesting must be 129..254. Below 129 the level fits in 7 bits and
 *     resolution is correct; 255+ is rejected by codegen ("too complex
 *     expression").
 *   - outer locals must be >= ~80. Below that the stray read stays IN BOUNDS
 *     and quietly returns the wrong value (the trigger prints the
 *     instance_eval receiver instead of the intended local) — the same bug
 *     with no sanitizer-visible symptom, which would read as a spurious MISS.
 */
#include "../mock-mruby/mock_mruby.h"
#include <stdint.h>
#include <string.h>

#ifndef NESTING
#define NESTING 129      /* trigger.rb: 129 nested blocks */
#endif
#ifndef OUTER_LOCALS
#define OUTER_LOCALS 80  /* trigger.rb: v0..v79 */
#endif

_Static_assert(NESTING >= 129 && NESTING <= 254,
               "nesting outside 129..254: below 129 the scope level is not "
               "truncated (no defect); 255+ is rejected by codegen.");
_Static_assert(OUTER_LOCALS >= 80,
               "outer-local count is in the IN-BOUNDS regime: the stray read "
               "would stay inside the allocation and return a wrong value "
               "silently, so the row would measure the trigger, not CHERI.");
_Static_assert((NESTING & 0x7f) != NESTING,
               "the 7-bit scope level must actually truncate");

/* Geometry as ASan reports it for the pinned trigger. */
#define ENV_REGION_BYTES 4096
#define OOB_OFF 528 /* bytes past the end of that region */

int main(void) {
  mrb_state *mrb = mrb_open(1024);

  /* The environment uvenv() should have reached, sized for its locals. */
  char *env_stack = (char *)mrb_malloc(mrb, ENV_REGION_BYTES);

  /* uvenv() walks (NESTING & 0x7f) == 1 scope, so the read is indexed off the
   * wrong environment and runs off the end of this live, correctly-sized
   * register array. */
  volatile char *upvar = env_stack + ENV_REGION_BYTES + OOB_OFF;

  unsigned char buf[16];
  memcpy(buf, (const void *)upvar, sizeof buf); /* READ of size 16, OOB */

  mock_report("mruby_upvar_overflow", buf[0] ? "overflow-survived" : "overflow-survived");
  mrb_free(mrb, env_stack);
  mrb_close(mrb);
  return 0;
}
