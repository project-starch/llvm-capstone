/* The dissection-loop pool lifecycle, without dissectors or replay machinery. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "port.h"
#include "scopes.h"

_Noreturn void wm_fail(unsigned code) {
  fprintf(stderr, "wmem example failed: %u\n", code);
  exit(1);
}
int main(void) {
  void *payload = aligned_alloc(16, WM_PAYLOAD_BYTES);
  if (!payload)
    return 1;
  wm_init_backing(NULL, payload, 0);
  wm_scopes_init();
  wm_enter_file_scope();
  wmem_allocator_t *packet = wm_packet_pool_acquire();
  unsigned char *p = wmem_alloc(packet, 64);
  memset(p, 17, 64);
  unsigned char *f = wmem_alloc(wm_file_scope(), 200);
  memset(f, 29, 200);
  if (p[63] != 17 || f[199] != 29)
    return 1;
  /* The packet pool is reset per dissection; its block and storage remain. */
  wm_packet_pool_reset(packet);
  unsigned char *q = wmem_alloc(packet, 64);
  if (q != p)
    return 1;
  memset(q, 41, 64);
  f = wmem_realloc(wm_file_scope(), f, 400);
  if (f[199] != 29)
    return 1;
  wmem_free(wm_file_scope(), f);
  wm_packet_pool_release(packet);
  wm_leave_file_scope();
  wm_scopes_cleanup();
  free(payload);
  printf("ALLOCATOR_EXAMPLE wireshark PASS pointer_bytes=%zu\n", sizeof(void *));
  return 0;
}
