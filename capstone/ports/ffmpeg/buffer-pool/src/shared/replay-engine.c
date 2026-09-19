/* Execute pool APIs and the allocator effects nested inside recorded callbacks.
 * Expected allocation choices and counters are absent from the command stream.
 * Refcount traffic before the last unref, codec work, and object bytes are not
 * replayed. Semantic tests cover shared references separately.
 */
#include "replay-engine.h"
#include "libavutil/buffer.h"
#include "libavutil/refstruct.h"
#include <string.h>

static struct ff2_header *report;
static const struct ff2_header *trace;
static uint64_t cursor;
static struct replay_pool {
  void *p;
  unsigned kind, closed;
} pools[FF2_POOLS];
static struct replay_lease {
  void *p;
  uint64_t id, pool;
} leases[2048];
static void dispatch(void);
#ifdef FFPOOL_DOMAIN
#include "../capstone-domain/node-snapshots.h"
#endif

void ff2_lock(void) {}
void ff2_unlock(int *p) { (void)p; }
static const struct ff2_event *peek(void) {
  if (cursor >= trace->count)
    ff2_fail(201);
  return (const struct ff2_event *)(trace + 1) + cursor;
}
void ff2_sink(const struct ff2_event *actual) {
#ifdef FF2_SECURITY
  (void)actual;
  report->reserved[0]++;
#else
  const struct ff2_event *command = peek();
  if (memcmp(actual, command, 8 * sizeof(uint64_t)))
    ff2_fail(202);
  const uint64_t *outcomes = &command->backing;
  for (unsigned i = 0; i < 8; i++)
    if (outcomes[i])
      ff2_fail(203);
  ((struct ff2_event *)(report + 1))[cursor++] = *actual;
  report->count = cursor;
#ifdef FFPOOL_DOMAIN
  if (!(cursor % 32768))
    ff2_node_snapshot(cursor);
#endif
#endif
}
static void callback_effects(void) {
  while (peek()->op != (FF2_CALLBACK | FF2_FINISH))
    dispatch();
}
static int init_cb(AVRefStructOpaque opaque, void *obj) {
  (void)opaque;
  (void)obj;
  callback_effects();
  return 0;
}
static void object_cb(AVRefStructOpaque opaque, void *obj) {
  (void)opaque;
  (void)obj;
  callback_effects();
}
static void pool_cb(AVRefStructOpaque opaque) {
  (void)opaque;
  callback_effects();
}

static void dispatch(void) {
  struct ff2_event cmd = *peek();
  if (!cmd.pool || cmd.pool >= FF2_POOLS || cmd.kind < 1 || cmd.kind > 2)
    ff2_fail(204);
  struct replay_pool *p = &pools[cmd.pool];
  switch (cmd.op) {
  case FF2_CREATE:
    if (p->p || p->kind || !cmd.size || cmd.size > FF2_PAYLOAD_BYTES)
      ff2_fail(205);
    p->kind = cmd.kind;
    if (cmd.kind == FF2_BUFFER) {
      if (cmd.flags > 1)
        ff2_fail(206);
      p->p = av_buffer_pool_init(cmd.size, cmd.flags ? av_buffer_allocz : NULL);
    } else {
      uint64_t allowed = FF2_HAS_INIT | FF2_HAS_RESET | FF2_HAS_FREE_ENTRY |
                         FF2_HAS_FREE_POOL | UINT64_C(0xffffffff);
      if (cmd.flags & ~allowed)
        ff2_fail(207);
      p->p = av_refstruct_pool_alloc_ext(
          cmd.size, (unsigned)cmd.flags, p,
          cmd.flags & FF2_HAS_INIT ? init_cb : NULL,
          cmd.flags & FF2_HAS_RESET ? object_cb : NULL,
          cmd.flags & FF2_HAS_FREE_ENTRY ? object_cb : NULL,
          cmd.flags & FF2_HAS_FREE_POOL ? pool_cb : NULL);
    }
    if (!p->p)
      ff2_fail(208);
    break;
  case FF2_GET: {
    if (!p->p || p->closed || p->kind != cmd.kind)
      ff2_fail(209);
    /* Reserve before entering callbacks, which may allocate more leases. */
    unsigned i;
    for (i = 0; i < 2048 && leases[i].id; i++) {
    }
    if (i == 2048)
      ff2_fail(210);
    leases[i].id = cmd.object;
    leases[i].pool = cmd.pool;
    leases[i].p = cmd.kind == FF2_BUFFER ? (void *)av_buffer_pool_get(p->p)
                                         : av_refstruct_pool_get(p->p);
    if (!leases[i].p)
      ff2_fail(211);
    /* Valid payload probes exercise access without pretending to decode. */
    unsigned char *data = cmd.kind == FF2_BUFFER
                              ? ((AVBufferRef *)leases[i].p)->data
                              : leases[i].p;
    data[0] = (unsigned char)cmd.object;
    data[cmd.size - 1] = (unsigned char)(cmd.object >> 8);
    break;
  }
  case FF2_RETURN: {
    unsigned i;
    for (i = 0; i < 2048; i++)
      if (leases[i].id == cmd.object)
        break;
    if (i == 2048 || !leases[i].p || leases[i].pool != cmd.pool ||
        p->kind != cmd.kind)
      ff2_fail(212);
    void *obj = leases[i].p;
    unsigned char *data =
        cmd.kind == FF2_BUFFER ? ((AVBufferRef *)obj)->data : obj;
    if (cmd.size > 1 &&
        (data[0] != (unsigned char)cmd.object ||
         data[cmd.size - 1] != (unsigned char)(cmd.object >> 8)))
      ff2_fail(213);
    leases[i].id = 0;
    leases[i].p = NULL;
    if (cmd.kind == FF2_BUFFER)
      av_buffer_unref((AVBufferRef **)&obj);
    else
      av_refstruct_unref(&obj);
    break;
  }
  case FF2_CLOSE:
    if (!p->p || p->closed || p->kind != cmd.kind)
      ff2_fail(214);
    p->closed = 1;
    if (cmd.kind == FF2_BUFFER)
      av_buffer_pool_uninit((AVBufferPool **)&p->p);
    else
      av_refstruct_unref(&p->p);
    break;
  default:
    ff2_fail(215);
  }
}

uint64_t ff2_replay_cursor(void) { return cursor; }

uint64_t ff2_replay_operation(void) {
  if (!trace || cursor >= trace->count)
    return 0;
  return ((const struct ff2_event *)(trace + 1))[cursor].op;
}

void ff2_replay_run(const struct ff2_header *input, struct ff2_header *output,
                    void *metadata) {
  trace = input;
  report = output;
  cursor = 0;
  memset(pools, 0, sizeof pools);
  memset(leases, 0, sizeof leases);
  uint64_t mode = report->mode;
  *report = (struct ff2_header){.magic = FF2_MAGIC, .status = 1, .mode = mode};
  if (trace->magic != FF2_MAGIC || trace->status || !trace->count ||
      trace->count >
          (FF2_FILE_BYTES - sizeof *trace) / sizeof(struct ff2_event))
    ff2_fail(216);
  ff2_memory_init(metadata, FF2_META_BYTES);
  ff2_set_mode(mode);
  ff2_reset();
#ifdef FF2_SECURITY
  void ff2_security_run(unsigned, unsigned long, unsigned);
  ff2_security_run(trace->reserved[0], trace->reserved[1], mode);
  ff2_finish();
  report->count = trace->count;
#else
  while (peek()->op != FF2_DONE)
    dispatch();
  ff2_finish();
  if (cursor != trace->count)
    ff2_fail(217);
#endif
  ff2_memory_report(report);
#ifdef FFPOOL_DOMAIN
  ff2_node_snapshot(cursor);
#endif
  report->status = 0;
}
