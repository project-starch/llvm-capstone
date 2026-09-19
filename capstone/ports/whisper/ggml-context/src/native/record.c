/* Native instrumentation only; never linked into a capability domain. */
#include "record.h"
#include "port.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static struct {
  void *ptr;
  unsigned buffer;
} contexts[WG_CONTEXTS];
static void *buffers[WG_BUFFERS];
static unsigned buffer_count, initialized, live;
static FILE *stream;
static struct wg_header header = {.magic = WG_MAGIC, .object_header_size = 32};
static void emit(uint64_t op, unsigned c, unsigned b, size_t n, unsigned t,
                 unsigned arg) {
  struct wg_event e = {op, c, b, n, t, arg};
  if (header.count == (WG_TRACE_BYTES - sizeof header) / sizeof e ||
      fwrite(&e, sizeof e, 1, stream) != 1)
    abort();
  ++header.count;
}
static void finish(void) {
  pthread_mutex_lock(&lock);
  if (stream) {
    emit(WG_END, live, 0, 0, 0, 0);
    if (fseek(stream, 0, SEEK_SET) ||
        fwrite(&header, sizeof header, 1, stream) != 1 || fclose(stream))
      abort();
    stream = NULL;
  }
  pthread_mutex_unlock(&lock);
}
static void start(void) {
  if (initialized)
    return;
  initialized = 1;
  const char *path = getenv("WG_RECORD_PATH");
  if (!path)
    return;
  uint16_t endian = 1;
  if (*(unsigned char *)&endian != 1)
    abort();
  stream = fopen(path, "wbx");
  if (!stream || fwrite(&header, sizeof header, 1, stream) != 1 ||
      atexit(finish))
    abort();
}
static unsigned context(void *ptr) {
  for (unsigned c = 0; c < WG_CONTEXTS; ++c)
    if (contexts[c].ptr == ptr)
      return c;
  abort();
}
void wg_record_init(void *ctx, void *buffer, size_t size, int owned,
                    size_t header_size) {
  if (header_size != 32)
    abort();
  pthread_mutex_lock(&lock);
  start();
  if (stream) {
    unsigned c = 0, b = 0;
    while (c < WG_CONTEXTS && contexts[c].ptr)
      ++c;
    while (b < buffer_count && buffers[b] != buffer)
      ++b;
    if (c == WG_CONTEXTS || b == WG_BUFFERS)
      abort();
    if (b == buffer_count)
      buffers[buffer_count++] = buffer;
    for (unsigned i = 0; i < WG_CONTEXTS; ++i)
      if (contexts[i].ptr && contexts[i].buffer == b)
        abort();
    contexts[c].ptr = ctx;
    contexts[c].buffer = b;
    ++live;
    emit(WG_INIT, c, b, size, 0, owned);
  }
  pthread_mutex_unlock(&lock);
}
void wg_record_object(void *ctx, unsigned type, size_t size) {
  pthread_mutex_lock(&lock);
  if (stream) {
    unsigned c = context(ctx);
    emit(WG_ALLOC, c, contexts[c].buffer, size, type,
         (unsigned)header.count % 251 + 1);
  }
  pthread_mutex_unlock(&lock);
}
void wg_record_reset(void *ctx) {
  pthread_mutex_lock(&lock);
  if (stream) {
    unsigned c = context(ctx);
    emit(WG_RESET, c, contexts[c].buffer, 0, 0, 0);
  }
  pthread_mutex_unlock(&lock);
}
void wg_record_free(void *ctx) {
  pthread_mutex_lock(&lock);
  if (stream) {
    unsigned c = context(ctx);
    emit(WG_FREE, c, contexts[c].buffer, 0, 0, 0);
    contexts[c].ptr = NULL;
    --live;
  }
  pthread_mutex_unlock(&lock);
}
