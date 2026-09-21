/* Hosted entry: the same protocol as the domain, read from and written to
 * files, with the payload and metadata regions taken from the host heap. Only
 * mode 0 runs here; node-pointers.c refuses mode 1, because a native build has
 * no authority to revoke. */
#include "port.h"
#include "apr_shim.h"
#include "apr_pools.h"
#include <stdio.h>
#include <stdlib.h>
_Noreturn void aprp_fail(unsigned code) {
  fprintf(stderr, "APRP failed=%u\n", code);
  exit(1);
}
int main(int argc, char **argv) {
  if (argc != 3 && argc != 4)
    return 2;
  FILE *f = fopen(argv[1], "rb");
  struct aprp_header *input = malloc(APRP_FILE_BYTES), out = {0};
  if (!f || !input)
    return 2;
  size_t bytes = fread(input, 1, APRP_FILE_BYTES, f);
  if (ferror(f) || fgetc(f) != EOF || bytes < sizeof *input ||
      input->count > (APRP_FILE_BYTES - sizeof *input) / sizeof(struct aprp_event) ||
      bytes != sizeof *input + input->count * sizeof(struct aprp_event))
    return 3;
  fclose(f);
  if (argc == 4) {
    char *end;
    unsigned long mode = strtoul(argv[3], &end, 10);
    if (!argv[3][0] || *end || mode > 1)
      return 2;
    out.mode = mode;
  }
  void *metadata = aligned_alloc(4096, APRP_META_BYTES);
#ifdef APRP_NODES_FROM_MALLOC
  void *payload = NULL; /* the platform's malloc is the region */
#else
  void *payload = aligned_alloc(4096, APRP_PAYLOAD_BYTES);
  if (!payload)
    return 4;
#endif
  if (!metadata)
    return 4;
  aprp_meta_init(metadata);
  aprp_payload_init(payload);
  aprp_set_mode(out.mode);
  if (apr_pool_initialize() != APR_SUCCESS)
    return 4;
  aprp_replay(input, &out);
  apr_pool_terminate();
  f = fopen(argv[2], "wb");
  if (!f || fwrite(&out, sizeof out, 1, f) != 1 || fclose(f))
    return 5;
  printf("APRP completed=%llu nodes=%llu reuses=%llu releases=%llu "
         "discards=%llu\n",
         (unsigned long long)out.completed, (unsigned long long)out.nodes,
         (unsigned long long)out.node_reuses,
         (unsigned long long)out.node_releases,
         (unsigned long long)out.node_discards);
  free(metadata);
  free(payload);
  free(input);
  return 0;
}
