#include "ggml.h"
#include "port.h"
#include <string.h>
struct object {
  unsigned char *p;
  size_t size;
  unsigned char value;
  unsigned next;
};
static struct object objects[WG_OBJECTS];
static unsigned heads[WG_BUFFERS], free_head;
static struct ggml_context *contexts[WG_CONTEXTS];
static unsigned ctx_buffers[WG_CONTEXTS], owned[WG_CONTEXTS], bound[WG_BUFFERS];
static void check_buffer(unsigned id) {
  for (unsigned i = heads[id]; i; i = objects[i].next)
    for (size_t j = 0; j < objects[i].size; ++j)
      if (objects[i].p[j] != objects[i].value)
        wg_fail(301);
}
static void retire(unsigned id) {
  check_buffer(id);
  while (heads[id]) {
    unsigned i = heads[id];
    heads[id] = objects[i].next;
    objects[i].next = free_head;
    free_head = i;
  }
}
void wg_replay(const struct wg_header *in, struct wg_header *out) {
  unsigned mode = out->mode;
  memset(out, 0, sizeof *out);
  out->magic = WG_MAGIC;
  out->mode = mode;
  if (in->magic != WG_MAGIC || !in->count ||
      in->count > (WG_TRACE_BYTES - sizeof *in) / sizeof(struct wg_event))
    wg_fail(302);
  out->count = in->count;
  out->object_header_size = wg_object_header_size();
  if (in->object_header_size != 32 ||
      out->object_header_size < in->object_header_size)
    wg_fail(313);
  for (unsigned i = 1; i + 1 < WG_OBJECTS; ++i)
    objects[i].next = i + 1;
  free_head = 1;
  const struct wg_event *events = (const void *)(in + 1);
  for (size_t k = 0; k < in->count; ++k) {
    const struct wg_event *e = &events[k];
    if (e->op == WG_END) {
      if (k + 1 != in->count || e->ctx != out->live_contexts || e->buffer ||
          e->size || e->type || e->arg)
        wg_fail(303);
      for (unsigned b = 0; b < WG_BUFFERS; ++b)
        check_buffer(b);
      ++out->completed;
      wg_backing_stats(out);
      return;
    }
    if (e->ctx >= WG_CONTEXTS || e->buffer >= WG_BUFFERS ||
        e->size > WG_PAYLOAD_BYTES || e->type > 2)
      wg_fail(304);
    unsigned c = e->ctx, b = e->buffer;
    if (e->op == WG_INIT) {
      if (contexts[c] || bound[b] || e->arg > 1 || e->type || !e->size)
        wg_fail(305);
      retire(b);
      wg_select_buffer(b);
      /* Preserve the recorded spare capacity when capability pointers enlarge
       * each object header. Payload bytes remain recorded native byte counts.
       * Only the largest allocation epoch before this descriptor is freed is
       * needed; RESET starts a new epoch rather than accumulating objects. */
      size_t current_objects = 0, max_objects = 0;
      for (size_t j = k + 1; j < in->count; ++j) {
        const struct wg_event *future = &events[j];
        if (future->op == WG_END)
          break;
        if (future->ctx != c)
          continue;
        if (future->op == WG_FREE)
          break;
        if (future->op == WG_RESET)
          current_objects = 0;
        if (future->op == WG_ALLOC && ++current_objects > max_objects)
          max_objects = current_objects;
      }
      size_t extra =
          max_objects * (out->object_header_size - in->object_header_size);
      if (extra > WG_PAYLOAD_BYTES - e->size)
        wg_fail(314);
      size_t capacity = e->size + extra;
      struct ggml_init_params params = {
          capacity, e->arg ? NULL : wg_borrow_buffer(b, capacity), true};
      contexts[c] = ggml_init(params);
      if (!contexts[c])
        wg_fail(306);
      ctx_buffers[c] = b;
      owned[c] = e->arg;
      bound[b] = 1;
      ++out->inits;
      ++out->live_contexts;
    } else {
      if (!contexts[c] || ctx_buffers[c] != b)
        wg_fail(307);
      if (e->op == WG_ALLOC) {
        if (e->arg > 255 || !free_head)
          wg_fail(308);
        unsigned i = free_head;
        free_head = objects[i].next;
        struct object *o = &objects[i];
        o->p = wg_object_alloc(contexts[c], e->type, e->size);
        if (!o->p)
          wg_fail(309);
        o->size = e->size;
        o->value = e->arg;
        o->next = heads[b];
        heads[b] = i;
        memset(o->p, o->value, o->size);
        size_t used = ggml_used_mem(contexts[c]);
        out->layout_checksum =
            out->layout_checksum * 33 ^
            (used +
             7 * ((uintptr_t)o->p -
                  (uintptr_t)ggml_get_mem_buffer(contexts[c])) +
             e->type);
        if (used > out->peak_used)
          out->peak_used = used;
        ++out->objects;
      } else if (e->op == WG_RESET || e->op == WG_FREE) {
        if (e->size || e->type || e->arg)
          wg_fail(310);
        if (e->op == WG_RESET) {
          retire(b);
          ggml_reset(contexts[c]);
          ++out->resets;
        } else {
          if (owned[c]) {
            retire(b);
            ++out->owned_frees;
          } else {
            check_buffer(b);
            ++out->borrowed_frees;
          }
          ggml_free(contexts[c]);
          contexts[c] = NULL;
          bound[b] = 0;
          --out->live_contexts;
          /* This check deliberately runs AFTER descriptor destruction. */
          if (!owned[c])
            check_buffer(b);
        }
      } else
        wg_fail(311);
    }
    out->checksum = out->checksum * 33 ^ (e->op + 7 * e->ctx + 13 * e->buffer +
                                          17 * e->size + e->type + e->arg);
    ++out->completed;
  }
  wg_fail(312);
}
