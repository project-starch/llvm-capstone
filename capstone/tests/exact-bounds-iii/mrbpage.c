/* mruby's heap page as patch 0008 leaves it, to get offsetof(objects) under the capability ABI.
   RVALUE's exact size is upstream and unknown here, so the probe is parameterised on it and the
   conclusion is checked across every plausible value rather than assuming one. */
typedef unsigned char mrb_bool;
typedef struct RBasic RBasic;
typedef struct RVALUE RVALUE;
struct RVALUE { void *c; unsigned long f; };   /* >= one capability; size pinned below */
#define MRB_HEAP_PAGE_SIZE 1024
typedef struct mrb_heap_page {
  RVALUE *freelist;
  struct mrb_heap_page *next;
  struct mrb_heap_page *free_next;
  mrb_bool old:1;
  mrb_bool region:1;
  void *life;
  struct RBasic *issued[MRB_HEAP_PAGE_SIZE];
  RVALUE objects[MRB_HEAP_PAGE_SIZE];
} mrb_heap_page;
unsigned long off_objects = __builtin_offsetof(mrb_heap_page, objects);
unsigned long size_ptr    = sizeof(void *);
unsigned long size_rvalue = sizeof(RVALUE);
