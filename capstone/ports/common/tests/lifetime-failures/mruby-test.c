#include <mruby.h>
#include <mruby/gc.h>
#include <mruby/error.h>
#include <stdio.h>
#include <string.h>
unsigned long life_calls, life_fail;
static mrb_value allocate(mrb_state *mrb, void *data)
{
  (void)data;
  return mrb_obj_value(mrb_obj_alloc(mrb,MRB_TT_OBJECT,mrb->object_class));
}
int main(int argc, char **argv)
{
  mrb_state *mrb=mrb_open();
  if (!mrb) return 10;
  mrb->gc.disabled=TRUE;
  int ai=mrb_gc_arena_save(mrb);
  if (argc>1 && !strcmp(argv[1],"stale")) {
    struct RBasic *p=mrb_obj_alloc(mrb,MRB_TT_OBJECT,mrb->object_class);
    mrb_gc_arena_restore(mrb,ai);
    mrb->gc.disabled=FALSE;
    mrb_full_gc(mrb);
    puts("TRAPTEST: mruby stale armed");
    return ((volatile struct RBasic *)p)->tt;
  }
  for (unsigned long n=1;n<=3;n++) {
    life_calls=0; life_fail=n;
    for (unsigned long k=1;k<=n;k++) {
      size_t before=mrb->gc.live;
      mrb_bool error=FALSE;
      mrb_protect_error(mrb,allocate,NULL,&error);
      if (k==n) {
        if (!error || mrb->gc.live!=before) return 11;
      } else if (error || mrb->gc.live!=before+1) return 12;
    }
    life_fail=0;
    if (!mrb_obj_alloc(mrb,MRB_TT_OBJECT,mrb->object_class)) return 13;
  }
  mrb_close(mrb);
  puts("TRAPTEST: mruby injection PASS");
  return 0;
}
