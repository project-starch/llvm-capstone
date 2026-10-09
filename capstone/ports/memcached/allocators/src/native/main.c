/* Hosted entry: the same protocol as the domain, read from and written to
 * files, with the metadata region taken from the host heap.
 *
 * Which modes run depends on the authority layer underneath. With the native
 * one, only mode 0: it has nothing to revoke, and leases.c refuses mode 1
 * before anything is allocated. With MCP_BORROW_LINEAR the Sublet authority is
 * underneath, the payload is LENT by the system allocator as one linear
 * capability, and both modes run -- which is what makes this entry the vehicle
 * for the nested arm in a Capstone process as well. */
#include "mc_slabs_shim.h"
#include "port.h"
#include "slabs.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef MCP_POISONCAP
#include "poisoncap.h"
#endif
#ifdef MCP_BORROW_LINEAR
/* The heap lends one linear block and keeps the senior handle, so the
 * authority layer can revoke inside it and the heap can still reclaim all of
 * it. Declared, not included: this is the allocator's internal interface. */
#include "../../../../common/include/borrow-aligned-block.h"
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
#ifdef MCP_BORROW_LINEAR
  /* mcp_authority_init requires the region linear, exactly MCP_PAYLOAD_BYTES
   * long and MCP_GRAIN-aligned. The heap gives all three by construction -- it
   * rounds a request up to a power of two and acquires each arena aligned to
   * its own size -- and the authority layer checks them again and calls
   * mcp_fail(501) if they do not hold, so a region that is merely nearly right
   * stops the arm instead of quietly changing what it measures. */
  capstone_cap_slot lent, head, tail;
  void *payload = NULL;
  if (!capstone_borrow_aligned_block(MCP_PAYLOAD_BYTES, MCP_GRAIN, &lent,
                                     &head, &tail))
    return 4;
#else
  void *payload = aligned_alloc(4096, MCP_PAYLOAD_BYTES);
#endif
#endif
#if !defined(MCP_ADAPTER_BACKING) && !defined(MCP_BORROW_LINEAR)
  if (!payload)
    return 4;
#endif
  if (!metadata)
    return 4;
  mcp_meta_init(metadata);
#ifdef MCP_BORROW_LINEAR
  /* capstone_cap_load moves the region out of the slot into a register, which
   * is how the authority layer takes it: in a domain it arrives that way. */
  mcp_payload_init(capstone_cap_load(&lent));
#else
  mcp_payload_init(payload);
#endif
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
#ifdef MCP_BORROW_LINEAR
  /* The payload is the heap's; its senior handle is what reclaims it, and the
   * process is ending anyway. */
#else
  free(payload);
#endif
  free(input);
  return 0;
}
