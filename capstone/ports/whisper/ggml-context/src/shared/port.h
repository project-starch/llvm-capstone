#ifndef WHISPER_CONTEXT_PORT_H
#define WHISPER_CONTEXT_PORT_H
#include <stddef.h>
#include <stdint.h>
#define WG_MAGIC UINT64_C(0x315854434c4d4747)
#define WG_PAYLOAD_BYTES (384UL << 20)
#define WG_META_BYTES (8UL << 20)
#define WG_TRACE_BYTES (16UL << 20)
#define WG_BUFFERS 128
#define WG_CONTEXTS 128
#define WG_OBJECTS 32768
enum { WG_INIT = 1, WG_ALLOC, WG_RESET, WG_FREE, WG_END };
struct wg_header {
  uint64_t magic, count, mode, status, completed, inits, objects, resets;
  uint64_t owned_frees, borrowed_frees, rebinds, peak_used, checksum,
      live_contexts;
  uint64_t object_header_size, layout_checksum;
};
struct wg_event {
  uint64_t op, ctx, buffer, size, type, arg;
};
struct ggml_context;
_Noreturn void wg_fail(unsigned code);
void wg_init_backing(void *metadata, void *payload, unsigned mode);
void *wg_meta_alloc(size_t n);
void wg_meta_free(void *p);
void *wg_aligned_alloc(size_t n);
void wg_aligned_free(void *p, size_t n);
void wg_select_buffer(unsigned id);
void *wg_borrow_buffer(unsigned id, size_t n);
void *wg_bind(struct ggml_context *ctx, void *p, size_t n, int owned);
void *wg_reset_buffer(struct ggml_context *ctx, void *p);
void wg_unbind(struct ggml_context *ctx, void *p, int owned);
void wg_backing_stats(struct wg_header *out);
void *wg_object_alloc(struct ggml_context *ctx, unsigned type, size_t n);
size_t wg_object_header_size(void);
void wg_replay(const struct wg_header *in, struct wg_header *out);
#endif
