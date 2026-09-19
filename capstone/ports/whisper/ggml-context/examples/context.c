/* Owned and borrowed ggml contexts, without inference or replay machinery. */
#include "ggml.h"
#include "port.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Noreturn void wg_fail(unsigned code) {
  fprintf(stderr, "GGML example failed: %u\n", code);
  exit(1);
}
int main(void) {
  void *payload = aligned_alloc(16, WG_PAYLOAD_BYTES);
  if (!payload)
    return 1;
  wg_init_backing(NULL, payload, 0);
  wg_select_buffer(1);
  struct ggml_context *owned = ggml_init((struct ggml_init_params){4096, NULL, false});
  if (!owned)
    return 1;
  unsigned char *p = wg_object_alloc(owned, 2, 64);
  memset(p, 17, 64);
  if (p[63] != 17)
    return 1;
  ggml_reset(owned);
  p = wg_object_alloc(owned, 2, 64);
  memset(p, 41, 64);
  if (p[0] != 41)
    return 1;
  ggml_free(owned);
  void *buffer = wg_borrow_buffer(2, 4096);
  wg_select_buffer(2);
  struct ggml_context *borrowed = ggml_init((struct ggml_init_params){4096, buffer, false});
  if (!borrowed)
    return 1;
  p = wg_object_alloc(borrowed, 2, 64);
  p[0] = 61;
  ggml_free(borrowed);
  /* Ending a borrowed context does not free its backing storage. */
  ((unsigned char *)buffer)[4095] = 93;
  if (((unsigned char *)buffer)[4095] != 93)
    return 1;
  free(payload);
  printf("ALLOCATOR_EXAMPLE whisper PASS pointer_bytes=%zu\n", sizeof(void *));
  return 0;
}
