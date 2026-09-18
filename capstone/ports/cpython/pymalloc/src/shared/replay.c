#include "port.h"
#include <string.h>
struct object {
  unsigned char *p;
  size_t size;
  unsigned char value;
};
static void check(struct object *o) {
  for (size_t j = 0; j < o->size; ++j)
    if (o->p[j] != o->value)
      pym_fail(301);
}
void pym_replay(const struct pym_header *input, struct pym_header *out,
                void *scratch) {
  unsigned mode = out->mode;
  memset(out, 0, sizeof *out);
  out->magic = PYM_MAGIC;
  out->mode = mode;
  if (mode > 1 || input->magic != PYM_MAGIC || !input->count ||
      input->count >
          (PYM_FILE_BYTES - sizeof *input) / sizeof(struct pym_event))
    pym_fail(302);
  out->count = input->count;
  struct object *objects = scratch;
  memset(objects, 0, sizeof(*objects) * PYM_MAX_OBJECTS);
  const struct pym_event *events = (const void *)(input + 1);
  size_t live = 0;
  for (size_t i = 0; i < input->count; ++i) {
    const struct pym_event *e = &events[i];
    if (e->op == PYM_END) {
      if (i + 1 != input->count || e->id != live || e->size || e->value)
        pym_fail(303);
      for (size_t j = 0; j < PYM_MAX_OBJECTS; ++j)
        if (objects[j].p)
          check(&objects[j]);
      out->completed++;
      pym_backing_stats(out);
      return;
    }
    if (e->id >= PYM_MAX_OBJECTS || e->size > 1048576 || e->value > 255)
      pym_fail(304);
    struct object *o = &objects[e->id];
    if (e->op == PYM_ALLOC || e->op == PYM_CALLOC) {
      if (o->p)
        pym_fail(305);
      o->p = e->op == PYM_ALLOC ? pym_malloc(e->size) : pym_calloc(1, e->size);
      if (!o->p)
        pym_fail(306);
      o->size = e->size;
      if (e->op == PYM_CALLOC) {
        o->value = 0;
        check(o);
      }
      live++;
      out->allocations++;
    } else if (e->op == PYM_REALLOC) {
      if (!o->p)
        pym_fail(307);
      check(o);
      unsigned char *p = pym_realloc(o->p, e->size);
      if (!p)
        pym_fail(308);
      o->p = p;
      if (o->size > e->size)
        o->size = e->size;
      check(o);
      o->size = e->size;
      out->reallocations++;
    } else if (e->op == PYM_FREE) {
      if (!o->p || e->size || e->value)
        pym_fail(309);
      check(o);
      pym_free(o->p);
      o->p = NULL;
      live--;
      out->frees++;
    } else
      pym_fail(310);
    if (o->p) {
      o->value = e->value;
      memset(o->p, o->value, o->size);
    }
    out->checksum =
        (out->checksum * 33) ^ (e->op + 7 * e->id + 13 * e->size + e->value);
    out->completed++;
  }
  pym_fail(311); /* an explicit checked END is required */
}
