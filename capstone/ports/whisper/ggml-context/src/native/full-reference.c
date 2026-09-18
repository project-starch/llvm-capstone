/* Cross-check extraction against the ordinary complete upstream ggml library.
 * Object kind is metadata only in ggml_new_object; new_buffer exposes the same
 * allocation algorithm publicly, with no tensor/graph construction added. */
#include "ggml.h"
#include "port.h"
size_t wg_object_header_size(void) {
  return ggml_tensor_overhead() - sizeof(struct ggml_tensor);
}
void *wg_object_alloc(struct ggml_context *ctx, unsigned type, size_t n) {
  (void)type;
  return ggml_new_buffer(ctx, n);
}
