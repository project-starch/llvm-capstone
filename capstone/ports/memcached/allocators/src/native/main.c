/* Hosted entry: the same protocol as the domain, read from and written to
 * files, with the payload and metadata regions taken from the host heap. Only
 * mode 0 runs here; the native authority layer has nothing to revoke, and
 * leases.c refuses mode 1 before anything is allocated. */
#include "mc_slabs_shim.h"
#include "port.h"
#include "slabs.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef MCP_POISONCAP
#include "poisoncap.h"
#endif
_Noreturn void mcp_fail(unsigned code) {
  fprintf(stderr, "MCP failed=%u\n", code);
  exit(1);
}
int main(int argc, char **argv) {
  if (argc != 3 && argc != 4)
    return 2;
  FILE *f = fopen(argv[1], "rb");
  struct mcp_header *input = malloc(MCP_FILE_BYTES), out = {0};
  if (!f || !input)
    return 2;
  size_t bytes = fread(input, 1, MCP_FILE_BYTES, f);
  if (ferror(f) || fgetc(f) != EOF || bytes < sizeof *input ||
      input->count > (MCP_FILE_BYTES - sizeof *input) / sizeof(struct mcp_event) ||
      bytes != sizeof *input + input->count * sizeof(struct mcp_event))
    return 3;
  fclose(f);
  if (argc == 4) {
    char *end;
    unsigned long mode = strtoul(argv[3], &end, 10);
    if (!argv[3][0] || *end || mode > 1)
      return 2;
    out.mode = mode;
  }
  void *metadata = aligned_alloc(4096, MCP_META_BYTES);
#ifdef MCP_ADAPTER_BACKING
  void *payload = NULL; /* the CheriBSD adapter owns its own backing */
#else
  void *payload = aligned_alloc(4096, MCP_PAYLOAD_BYTES);
#endif
#ifndef MCP_ADAPTER_BACKING
  if (!payload)
    return 4;
#endif
  if (!metadata)
    return 4;
  mcp_meta_init(metadata);
  mcp_payload_init(payload);
  mcp_set_mode(out.mode);
  slabs_init(settings.maxbytes, settings.factor, false, NULL, NULL, false);
  mcp_replay(input, &out);
  f = fopen(argv[2], "wb");
  if (!f || fwrite(&out, sizeof out, 1, f) != 1 || fclose(f))
    return 5;
  printf("MCP completed=%llu pages=%llu chunk_reuses=%llu chunk_releases=%llu "
         "object_reuses=%llu object_releases=%llu\n",
         (unsigned long long)out.completed, (unsigned long long)out.pages,
         (unsigned long long)out.chunk_reuses,
         (unsigned long long)out.chunk_releases,
         (unsigned long long)out.object_reuses,
         (unsigned long long)out.object_releases);
#ifdef MCP_POISONCAP
  /* What the protection cost, beside what the allocators did. In the spatial
   * arm every counter here must be zero. */
  mcp_poisoncap_report();
#endif
  free(metadata);
  free(payload);
  free(input);
  return 0;
}
