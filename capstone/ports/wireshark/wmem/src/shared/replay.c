#include "port.h"
#include "wmem_core.h"
#include <string.h>
struct object {
  unsigned char *p;
  size_t size;
  unsigned char value;
  unsigned allocator, live;
};
static struct object objects[WM_OBJECTS];
static wmem_allocator_t *pools[WM_ALLOCATORS];
static void check_object(const struct object *o) {
  for (size_t j = 0; j < o->size; ++j)
    if (o->p[j] != o->value)
      wm_fail(301);
}
static void check_pool(unsigned a) {
  for (unsigned i = 0; i < WM_OBJECTS; ++i)
    if (objects[i].live && objects[i].allocator == a)
      check_object(&objects[i]);
}
static void retire_pool(unsigned a) {
  check_pool(a);
  for (unsigned i = 0; i < WM_OBJECTS; ++i)
    if (objects[i].live && objects[i].allocator == a)
      objects[i].live = 0;
}
void wm_replay(const struct wm_header *in, struct wm_header *out) {
  unsigned mode = out->mode;
  memset(out, 0, sizeof *out);
  out->magic = WM_MAGIC;
  out->mode = mode;
  if (in->magic != WM_MAGIC || !in->count ||
      in->count > (WM_TRACE_BYTES - sizeof *in) / sizeof(struct wm_event))
    wm_fail(302);
  out->count = in->count;
  wmem_init();
  const struct wm_event *events = (const void *)(in + 1);
  for (size_t k = 0; k < in->count; ++k) {
    const struct wm_event *e = &events[k];
    if (e->op == WM_END) {
      if (k + 1 != in->count || e->allocator != out->live_allocators ||
          e->object || e->size || e->type || e->arg)
        wm_fail(303);
      for (unsigned a = 0; a < WM_ALLOCATORS; ++a)
        if (pools[a])
          check_pool(a);
      ++out->completed;
      wm_backing_stats(out);
      return;
    }
    if (e->allocator >= WM_ALLOCATORS || e->object >= WM_OBJECTS ||
        e->size > WM_PAYLOAD_BYTES)
      wm_fail(304);
    unsigned a = e->allocator;
    struct object *o = &objects[e->object];
    switch (e->op) {
    case WM_NEW:
      if (pools[a] || e->type > 3 || e->object || e->size || e->arg)
        wm_fail(305);
      pools[a] = wmem_allocator_new((wmem_allocator_type_t)e->type);
      if (!pools[a])
        wm_fail(306);
      ++out->news;
      ++out->live_allocators;
      break;
    case WM_ALLOC:
      if (!pools[a] || o->live || !e->size || e->arg > 255 || e->type)
        wm_fail(307);
      o->p = wmem_alloc(pools[a], e->size);
      if (!o->p)
        wm_fail(308);
      o->size = e->size;
      o->value = (unsigned char)e->arg;
      o->allocator = a;
      o->live = 1;
      memset(o->p, o->value, o->size);
      ++out->allocs;
      break;
    case WM_FREE:
      if (!pools[a] || !o->live || o->allocator != a || e->size || e->type ||
          e->arg)
        wm_fail(309);
      check_object(o);
      wmem_free(pools[a], o->p);
      o->live = 0;
      ++out->frees;
      break;
    case WM_REALLOC: {
      if (!pools[a] || !o->live || o->allocator != a || !e->size ||
          e->arg > 255 || e->type)
        wm_fail(310);
      check_object(o);
      unsigned char *q = wmem_realloc(pools[a], o->p, e->size);
      if (!q)
        wm_fail(311);
      size_t keep = e->size < o->size ? e->size : o->size;
      for (size_t j = 0; j < keep; ++j)
        if (q[j] != o->value)
          wm_fail(312);
      o->p = q;
      o->size = e->size;
      o->value = (unsigned char)e->arg;
      memset(o->p, o->value, o->size);
      ++out->reallocs;
      break;
    }
    case WM_FREE_ALL:
    case WM_GC:
    case WM_DESTROY:
      if (!pools[a] || e->object || e->size || e->type || e->arg)
        wm_fail(313);
      if (e->op == WM_GC) {
        check_pool(a);
        wmem_gc(pools[a]);
        ++out->gcs;
      } else if (e->op == WM_FREE_ALL) {
        retire_pool(a);
        wmem_free_all(pools[a]);
        ++out->free_alls;
      } else {
        retire_pool(a);
        wmem_destroy_allocator(pools[a]);
        pools[a] = NULL;
        --out->live_allocators;
        ++out->destroys;
      }
      break;
    default:
      wm_fail(314);
    }
    out->checksum = out->checksum * 33 ^ (e->op + 7 * e->allocator +
                                          13 * e->object + 17 * e->size +
                                          e->type + e->arg);
    ++out->completed;
  }
  wm_fail(315);
}
