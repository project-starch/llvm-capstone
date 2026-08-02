/* mock-mruby lifecycle harness — see mock_mruby.h for what this models. */
#include "mock_mruby.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- allocator layer, named as in mruby's gc.c / state.c ---------------- */

static void *mrb_default_allocf(void *p, size_t size) {
  if (size == 0) {
    free(p);
    return NULL;
  }
  return realloc(p, size);
}

static void *mrb_realloc_simple(void *p, size_t size) {
  return mrb_default_allocf(p, size);
}

static void *mrb_realloc(void *p, size_t size) { return mrb_realloc_simple(p, size); }

void *mrb_malloc(mrb_state *mrb, size_t n) {
  (void)mrb;
  void *p = mrb_realloc(NULL, n);
  if (!p) abort();
  return p;
}

void mrb_free(mrb_state *mrb, void *p) {
  (void)mrb;
  free(p);
}

/* --- VM register stack -------------------------------------------------- */

static void stack_init(mrb_state *mrb, size_t stack_bytes) {
  mrb->c->stack = (mrb_value *)calloc(1, stack_bytes);
  if (!mrb->c->stack) abort();
  mrb->c->stack_size = stack_bytes;
  /* The triggers push strings and arrays per recursion frame with the stated
   * intent "prevent in-place reallocations of the stack". One adjacent live
   * block reproduces that effect deterministically. */
  mrb->c->wedge = malloc(64);
  if (!mrb->c->wedge) abort();
}

static mrb_value *stack_extend_alloc(mrb_state *mrb, size_t new_bytes) {
  return (mrb_value *)mrb_realloc(mrb->c->stack, new_bytes);
}

void mrb_stack_extend(mrb_state *mrb, size_t new_bytes) {
  mrb_value *old = mrb->c->stack;
  mrb_value *ns = stack_extend_alloc(mrb, new_bytes);
  if (!ns) abort();
  if (ns == old) {
    /* Not a defect in the row: without a move there is no dangling pointer,
     * so the shim would report a MISS that says nothing about CHERI. */
    fprintf(stderr, "SHIM-INVALID: realloc grew the VM stack in place; "
                    "no stale pointer exists to measure\n");
    exit(66);
  }
  mrb->c->stack = ns;
  mrb->c->stack_size = new_bytes;
}

void mrb_funcall_cb(mrb_state *mrb, mrb_callback cb) { cb(mrb); }

/* --- GC path ------------------------------------------------------------ */

void *mrb_gc_alloc(mrb_state *mrb, size_t n) {
  mrb->gc_obj = mrb_malloc(mrb, n);
  mrb->gc_obj_size = n;
  memset(mrb->gc_obj, 0, n);
  return mrb->gc_obj;
}

static void obj_free(mrb_state *mrb, void *p) { mrb_free(mrb, p); }

static void incremental_sweep_phase(mrb_state *mrb) {
  if (mrb->gc_obj) {
    obj_free(mrb, mrb->gc_obj);
    mrb->gc_obj = NULL;
  }
}

void mrb_full_gc(mrb_state *mrb) { incremental_sweep_phase(mrb); }

/* --- lifecycle ---------------------------------------------------------- */

mrb_state *mrb_open(size_t stack_bytes) {
  mrb_state *mrb = (mrb_state *)calloc(1, sizeof *mrb);
  if (!mrb) abort();
  mrb->c = (mrb_context *)calloc(1, sizeof *mrb->c);
  if (!mrb->c) abort();
  stack_init(mrb, stack_bytes);
  return mrb;
}

void mrb_close(mrb_state *mrb) {
  if (!mrb) return;
  free(mrb->c->stack);
  free(mrb->c->wedge);
  free(mrb->c);
  free(mrb);
}

void mock_report(const char *row, const char *what) {
  printf("MOCK %s %s\n", row, what);
  fflush(stdout);
}
