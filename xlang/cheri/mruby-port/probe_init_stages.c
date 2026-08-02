/* Stage 4: fault is inside mrb_init_core. That is a flat list of per-subsystem
 * init calls (src/init.c), so replicate it with a marker after each one. The
 * last marker printed names the exact subsystem that faults. */
#include <mruby.h>
#include <string.h>
#include <stdio.h>

void mrb_gc_init(mrb_state *, mrb_gc *);
MRB_API void *mrb_default_allocf(mrb_state *, void *, size_t, void *);
#define D(f) void f(mrb_state *);
D(mrb_init_symtbl) D(mrb_init_class) D(mrb_init_object) D(mrb_init_kernel)
D(mrb_init_comparable) D(mrb_init_enumerable) D(mrb_init_symbol)
D(mrb_init_string) D(mrb_init_exception) D(mrb_init_proc) D(mrb_init_array)
D(mrb_init_hash) D(mrb_init_numeric) D(mrb_init_range) D(mrb_init_gc)
D(mrb_init_version) D(mrb_init_mrblib)

#define STEP(f) do { f(mrb); printf("MRBPROBE ok_%s\n", #f); fflush(stdout); } while (0)

int main(void) {
  mrb_state *mrb = (mrb_state *)mrb_default_allocf(NULL, NULL, sizeof(mrb_state), NULL);
  memset(mrb, 0, sizeof(mrb_state));
  mrb->allocf = mrb_default_allocf;
  mrb->allocf_ud = NULL;
  mrb->atexit_stack_len = 0;
  mrb_gc_init(mrb, &mrb->gc);
  mrb->c = (struct mrb_context *)mrb_malloc(mrb, sizeof(struct mrb_context));
  memset(mrb->c, 0, sizeof(struct mrb_context));
  mrb->root_c = mrb->c;
  printf("MRBPROBE prelude_ok\n"); fflush(stdout);

  STEP(mrb_init_symtbl);   STEP(mrb_init_class);     STEP(mrb_init_object);
  STEP(mrb_init_kernel);   STEP(mrb_init_comparable);STEP(mrb_init_enumerable);
  STEP(mrb_init_symbol);   STEP(mrb_init_string);    STEP(mrb_init_exception);
  STEP(mrb_init_proc);     STEP(mrb_init_array);     STEP(mrb_init_hash);
  STEP(mrb_init_numeric);  STEP(mrb_init_range);     STEP(mrb_init_gc);
  STEP(mrb_init_version);  STEP(mrb_init_mrblib);
  printf("MRBPROBE ALL_OK\n"); fflush(stdout);
  return 0;
}
