#include "profile.h"
#include "domain-runtime.h"

/* Integer-only observer state. Context deletion follows PostgreSQL's
 * descendant walk; reset deletes children but retains the selected context.
 * Object lengths come from the replay's existing payload-check table. */
struct context {
  unsigned int parent, child, next, prev;
  unsigned long bytes, objects, active;
};
struct sample {
  unsigned long event, op, live_bytes, live_objects, live_contexts;
  struct pg_memory_backing backing;
};
static struct context *contexts;
static unsigned long context_count, live_bytes, live_objects, live_contexts;
static struct sample last, payload_peak, backing_peak, metadata_peak,
    tracked_peak;
static unsigned long observer_bytes;
static unsigned long previous_event = ~0UL;

static void field(unsigned long value) {
  pg_domain_text(",");
  pg_domain_uint(value);
}
static void emit(const char *kind, const struct sample *s) {
  pg_domain_text("PGMEM,");
  pg_domain_text(kind);
  field(s->event);
  field(s->op);
  field(s->live_bytes);
  field(s->live_objects);
  field(s->live_contexts);
  const struct pg_memory_backing *b = &s->backing;
  field(b->backing_bytes);
  field(b->assigned_bytes);
  field(b->block_bytes);
  field(b->stranded_bytes);
  field(b->metadata_live);
  field(b->metadata_reserved);
  field(b->pools);
  field(b->blocks);
  field(b->entries);
  field(b->nodes_created);
  field(b->revokes);
  field(b->init_bytes);
  field(b->arena_capacity);
  field(observer_bytes);
  pg_domain_text("\n");
}
static void clear_payload(unsigned int id) {
  struct context *c = &contexts[id];
  if (c->bytes > live_bytes || c->objects > live_objects)
    replay_die("profile live-payload accounting underflow");
  live_bytes -= c->bytes;
  live_objects -= c->objects;
  c->bytes = c->objects = 0;
}
static void remove_context(unsigned int id) {
  struct context *c = &contexts[id];
  while (c->child)
    remove_context(c->child);
  clear_payload(id);
  if (c->prev)
    contexts[c->prev].next = c->next;
  else if (c->parent)
    contexts[c->parent].child = c->next;
  if (c->next)
    contexts[c->next].prev = c->prev;
  if (!c->active)
    replay_die("profile removed an inactive context");
  c->active = 0;
  live_contexts--;
}
void pg_memory_begin(const struct a11_rec *footer) {
  context_count = footer->s2 + 2;
  observer_bytes = context_count * sizeof(*contexts);
  contexts = replay_alloc(observer_bytes);
  if (!contexts)
    replay_die("profile context table exceeds replay scratch");
  pg_domain_text(
      "PGMEM_HEADER,kind,event,op,live_payload_bytes,live_objects,live_"
      "contexts,backing_bytes,assigned_backing_bytes,block_bytes,stranded_"
      "block_bytes,metadata_records_live_bytes,metadata_records_reserved_bytes,"
      "pools,blocks,chunk_entries,nodes_created_cumulative,revokes_cumulative,"
      "init_bytes_cumulative,arena_capacity_bytes,observer_bytes\n");
}
void pg_memory_event(unsigned long i, const struct a11_rec *e,
                     const unsigned int *lengths) {
  if (e->ctx >= context_count ||
      (e->aux >= context_count && e->op >= A11_CREATE_ASET &&
       e->op <= A11_CREATE_BUMP))
    replay_die("profile context identity exceeds its table");
  struct context *c = &contexts[e->ctx];
  if (e->op >= A11_CREATE_ASET && e->op <= A11_CREATE_BUMP) {
    if (c->active)
      replay_die("profile context identity reused");
    c->active = 1;
    c->parent = e->aux;
    if (c->parent) {
      struct context *parent = &contexts[c->parent];
      if (!parent->active)
        replay_die("profile context parent is inactive");
      c->next = parent->child;
      if (c->next)
        contexts[c->next].prev = e->ctx;
      parent->child = e->ctx;
    }
    live_contexts++;
  } else if (e->op == A11_ALLOC) {
    c->bytes += e->s1;
    c->objects++;
    live_bytes += e->s1;
    live_objects++;
  } else if (e->op == A11_FREE || e->op == A11_REALLOC) {
    unsigned long bytes = lengths[e->ptr];
    if (!c->active || !c->objects || bytes > c->bytes)
      replay_die("profile free/realloc accounting mismatch");
    c->bytes -= bytes;
    c->objects--;
    live_bytes -= bytes;
    live_objects--;
    if (e->op == A11_REALLOC && e->aux) {
      c->bytes += e->s1;
      c->objects++;
      live_bytes += e->s1;
      live_objects++;
    }
  } else if (e->op == A11_RESET) {
    while (c->child)
      remove_context(c->child);
    clear_payload(e->ctx);
  } else if (e->op == A11_DELETE) {
    remove_context(e->ctx);
  }
  last.event = i + 1;
  last.op = e->op;
  last.live_bytes = live_bytes;
  last.live_objects = live_objects;
  last.live_contexts = live_contexts;
  pg_memory_backing(&last.backing);
  if (live_bytes > payload_peak.live_bytes)
    payload_peak = last;
  if (last.backing.backing_bytes > backing_peak.backing.backing_bytes)
    backing_peak = last;
  if (i == 0 ||
      last.backing.metadata_live > metadata_peak.backing.metadata_live)
    metadata_peak = last;
  if (last.backing.backing_bytes + last.backing.metadata_live >
      tracked_peak.backing.backing_bytes + tracked_peak.backing.metadata_live)
    tracked_peak = last;
  if (i < 64 || (i + 1) % 1024 == 0 || e->op == A11_RESET ||
      e->op == A11_DELETE) {
    emit("event", &last);
    previous_event = last.event;
  }
}
void pg_memory_end(unsigned long events) {
  if (previous_event != events)
    emit("event", &last);
  emit("payload_peak", &payload_peak);
  emit("backing_peak", &backing_peak);
  emit("metadata_peak", &metadata_peak);
  emit("tracked_peak", &tracked_peak);
  pg_domain_text("__CAPSTONE_PG_MEMORY_DONE__\n");
}
