/* FFmpeg's memory functions for a Capstone process: the process's own malloc, which is virtual
 * mallocng on the virtual profile. The pools' payloads and metadata come from it, as they do in
 * FFmpeg linked against libc; every allocation is one heap object, bounded to its request and
 * retired by free. Aligned to 64 bytes, as FFmpeg's av_malloc aligns for its SIMD code. */
#define _POSIX_C_SOURCE 200112L /* posix_memalign under -std=c11 */
#include "libavutil/log.h"
#include "libavutil/mem.h"
#include <stdlib.h>
#include <string.h>

void *av_malloc(size_t size) {
  void *p;
  return posix_memalign(&p, 64, size ? size : 1) ? NULL : p;
}
void av_free(void *p) { free(p); }
void *av_mallocz(size_t size) {
  void *p = av_malloc(size);
  if (p)
    memset(p, 0, size);
  return p;
}
void av_freep(void *slot) {
  void *p;
  memcpy(&p, slot, sizeof(p));
  memcpy(slot, &(void *){NULL}, sizeof(p));
  free(p);
}
void *av_realloc(void *p, size_t size) { return realloc(p, size ? size : 1); }
void av_log(void *avcl, int level, const char *fmt, ...) {
  (void)avcl;
  (void)level;
  (void)fmt;
}
