/* Case 11 at the 4.0.0-rc2 pin, through the C API: a closure escapes a proc that
 * mruby-task runs in a temporary task, the task's stack is freed with the closure's
 * env still pointing into it, and the next collection marks that env.
 *
 * At the pin no Ruby script reaches this (trigger.rb's Task#close postdates the pin;
 * see case.json harness_limit). mrb_execute_proc_synchronously is the C API that
 * upstream's own test reaches through its C helper TaskTest.run_sync (cb51fce92,
 * tasktest.c): it runs the proc in a temporary task and frees the task's stack and
 * callinfo with mrb_free (task.c, step 6), without detaching the envs on that stack.
 *
 *   capi             the defective sequence; prints CASE11 ready once the stack is freed
 *
 * The defective access is the GC's read of e->stack[i] in gc_mark_children's
 * MRB_TT_ENV case (gc.c), on the freed stack; upstream's report shows the crash in
 * mrb_gc_mark under gc_mark_children. Output: CASE11 ready, then CASE11 completed
 * with the closure's value (30 when nothing went wrong), or a fault in between. */
#include <stdio.h>
#include <unistd.h>
#include <mruby.h>
#include <mruby/compile.h>
#include <mruby/variable.h>
#include "task.h"

int main(void) {
  mrb_state *mrb = mrb_open();
  if (!mrb) return 75;
  setvbuf(stdout, NULL, _IONBF, 0);
  /* upstream's run_sync test, minus the helper: the proc's env escapes via $esc */
  mrb_value proc = mrb_load_string(mrb,
      "-> { c1 = 10; c2 = 20; $esc = -> { c1 + c2 }; 'sync-result' }");
  if (mrb->exc) { mrb_print_error(mrb); return 75; }
  mrb_value r = mrb_execute_proc_synchronously(mrb, proc, 0, NULL);
  /* The task's stack is freed now. Any collection from here on marks $esc's env
   * and reads e->stack[i] on the freed stack, so the mark comes first, and is a
   * bare write: an allocation here can already run a collection step. */
  static const char ready[] = "CASE11 ready\n";
  if (write(1, ready, sizeof ready - 1) < 0) return 75;
  mrb_p(mrb, r);
  mrb_full_gc(mrb);                       /* marks $esc's env: e->stack[i] on the freed stack */
  mrb_load_string(mrb, "j = []; 200.times { j << 'z' * 48 }; GC.start");   /* reuse, as upstream's test */
  mrb_value v = mrb_load_string(mrb, "$esc.call");
  if (mrb->exc) { mrb_print_error(mrb); return 1; }
  printf("CASE11 completed %lld\n", (long long)mrb_integer(v));
  mrb_close(mrb);
  return 0;
}
