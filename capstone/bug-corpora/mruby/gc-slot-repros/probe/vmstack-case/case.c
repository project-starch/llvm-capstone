/* realloc-vmstack, as a C reduction against the real interpreter.
 *
 * Every mruby row in this shape has one form: a raw mrb_value* into the VM data
 * stack, held across something that calls mrb_stack_extend(), then stored through.
 * Three Ruby reproducers failed to place the reallocation at the one instruction
 * that matters (CANDIDATES.md records them); C places it, because C holds the
 * pointer itself and can say how much stack the callee must demand.
 *
 *   0  the bare shape: stale WRITE after an extend that moves the stack
 *   1  the bare shape: stale READ
 *   2  the destination address fixed BEFORE the call that moves the stack
 *   3  c52faebb7  mrb_funcall_argv(), the operator fallback's own call
 *   4  7b503f3a3  mrb_ary_splat(), with a user-defined to_a, as its clusterfuzz
 *                 testcase describes
 *   5  e8d075045  mrb_hash_delete_key(), reached through the key's own hash/eql?
 *
 * Usage: case <0..5> [grow.rb] [depth]
 *
 * The depth is how deep Grow#grow recurses, and therefore how many stack slots the
 * callee demands: each frame holds 150 locals, so depth d asks for roughly 150*d.
 * The case prints the stack's size and address before and after, so a run where the
 * stack did NOT move is visible rather than silently uninteresting -- that case is
 * INVALID, not a MISS, which is the distinction xlang/capstone/rows.tsv insists on.
 *
 * Built against the pin's libmruby. Under -fsanitize=address the stale access is a
 * heap-use-after-free on the old stbase; in a plain build every case completes and
 * says nothing, which is the row's whole point.
 */
#include <stdio.h>
#include <stdlib.h>
#include <mruby.h>
#include <mruby/value.h>
#include <mruby/compile.h>
#include <mruby/array.h>
#include <mruby/hash.h>
#include <mruby/variable.h>
#include <mruby/string.h>

static const char *GROW_RB = "grow.rb";

static mrb_int stack_slots(mrb_state *mrb)
{
  return (mrb_int)(mrb->c->stend - mrb->c->stbase);
}

static void set_depth(mrb_state *mrb, int d)
{
  mrb_gv_set(mrb, mrb_intern_cstr(mrb, "$depth"), mrb_fixnum_value(d));
}

static void define_grow(mrb_state *mrb, const char *path)
{
  FILE *f = fopen(path, "r");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
  mrb_load_file(mrb, f);
  fclose(f);
  if (mrb->exc) {
    mrb_value m = mrb_funcall(mrb, mrb_obj_value(mrb->exc), "inspect", 0);
    fprintf(stderr, "%s raised: %s\n", path, RSTRING_PTR(m));
    exit(2);
  }
}

static mrb_value grow_obj(mrb_state *mrb)
{
  return mrb_gv_get(mrb, mrb_intern_cstr(mrb, "$grow"));
}

/* stands in for whatever the VM calls that can grow the stack, for cases 0-2 */
static mrb_value grow_then_answer(mrb_state *mrb, mrb_int room)
{
  mrb_stack_extend(mrb, room);
  return mrb_fixnum_value(0x5eed);
}

int main(int argc, char **argv)
{
  int which = argc > 1 ? atoi(argv[1]) : 0;
  const char *rbpath = argc > 2 ? argv[2] : GROW_RB;
  int depth = argc > 3 ? atoi(argv[3]) : 8;

  mrb_state *mrb = mrb_open();
  if (!mrb) { fputs("mrb_open failed\n", stderr); return 2; }

  if (which >= 3) { define_grow(mrb, rbpath); set_depth(mrb, depth); }

  mrb_value *regs = mrb->c->ci->stack;   /* what `#define regs (ci->stack)` is */
  volatile mrb_value sink;
  mrb_int before_slots = stack_slots(mrb);
  void *before_base = (void*)mrb->c->stbase;

  switch (which) {
  case 0:
    grow_then_answer(mrb, 1 << 16);
    regs[1] = mrb_fixnum_value(42);                      /* stale WRITE */
    break;
  case 1:
    regs[1] = mrb_fixnum_value(7);
    grow_then_answer(mrb, 1 << 16);
    sink = regs[1];                                      /* stale READ */
    break;
  case 2:
    regs[1] = grow_then_answer(mrb, 1 << 16);            /* destination fixed first */
    break;
  case 3: {                                             /* c52faebb7 */
    mrb_value arg = mrb_fixnum_value(1);
    regs[1] = mrb_funcall_argv(mrb, grow_obj(mrb), mrb_intern_cstr(mrb, "+"), 1, &arg);
    break;
  }
  case 4:                                               /* 7b503f3a3 */
    regs[1] = mrb_ary_splat(mrb, grow_obj(mrb));
    break;
  case 5: {                                             /* e8d075045 */
    mrb_value h = mrb_hash_new(mrb), k = grow_obj(mrb);
    mrb_hash_set(mrb, h, k, mrb_fixnum_value(7));
    regs = mrb->c->ci->stack;                            /* re-read: the set may grow */
    before_slots = stack_slots(mrb); before_base = (void*)mrb->c->stbase;
    regs[1] = mrb_hash_delete_key(mrb, h, k);
    break;
  }
  default:
    fputs("usage: case <0..5> [grow.rb] [depth]\n", stderr);
    mrb_close(mrb); return 2;
  }

  sink = regs[1];                        /* read back: no store here is dead */
  (void)sink;
  void *after_base = (void*)mrb->c->stbase;
  printf("case=%d depth=%d slots %d -> %d  stbase %p -> %p  %s\n",
         which, depth, (int)before_slots, (int)stack_slots(mrb),
         before_base, after_base,
         before_base == after_base ? "STACK DID NOT MOVE (run is INVALID)"
                                   : "stack moved (the access above was stale)");
  mrb_close(mrb);
  return 0;
}
