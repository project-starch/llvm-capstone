/* Hosted entry: the same protocol as the domain, read from and written to
 * files, with the metadata region taken from the host heap.
 *
 * Which modes run depends on the lease layer underneath. With
 * node-pointers.c, only mode 0: a native build has no authority to revoke and
 * it refuses mode 1. With APRP_BORROW_LINEAR the Sublet layer is underneath,
 * the payload is LENT by the system allocator as one linear capability, and
 * both modes run -- which makes this entry the vehicle for the nested arm in a
 * Capstone process too. */
#include "port.h"
#include "apr_shim.h"
#include "apr_pools.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef APRP_BORROW_LINEAR
/* The heap lends one linear block and keeps the senior handle, so the lease
 * layer can revoke inside it. Declared, not included: the allocator's own
 * interface. */
#include "../../../../common/include/borrow-aligned-block.h"
#endif
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
#if defined(APRP_NODES_FROM_MALLOC) || defined(APRP_POISONCAP)
  void *payload = NULL; /* the platform's malloc, or mapped regions, is the region */
#else
#ifdef APRP_BORROW_LINEAR
  /* aprp_payload_init requires the region linear, exactly APRP_PAYLOAD_BYTES
   * long and page-aligned, and calls aprp_fail(501) if it is not -- so a
   * region that is merely nearly right stops the arm rather than changing what
   * it measures. The heap gives all three by construction: a request is
   * rounded up to a power of two and each arena is acquired aligned to its own
   * size. */
  capstone_cap_slot lent, head, tail;
  void *payload = NULL;
  /* A page, because aprp_payload_init requires the region page-aligned. */
  if (!capstone_borrow_aligned_block(APRP_PAYLOAD_BYTES, 4096, &lent,
                                     &head, &tail))
    return 4;
#else
  void *payload = aligned_alloc(4096, APRP_PAYLOAD_BYTES);
  if (!payload)
    return 4;
#endif
#endif
  if (!metadata)
    return 4;
  aprp_meta_init(metadata);
#ifdef APRP_BORROW_LINEAR
  /* capstone_cap_load moves the region out of the slot into a register, which
   * is how the lease layer takes it: in a domain it arrives that way. */
  aprp_payload_init(capstone_cap_load(&lent));
#else
  aprp_payload_init(payload);
#endif
  aprp_set_mode(out.mode);
  if (apr_pool_initialize() != APR_SUCCESS)
    return 4;
  aprp_replay(input, &out);
  apr_pool_terminate();
#ifdef APRP_POISONCAP
  {
    void aprp_poison_report(void);
    aprp_poison_report();
  }
#endif
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
#ifdef APRP_BORROW_LINEAR
  /* The payload is the heap's; its senior handle reclaims it. */
#else
  free(payload);
#endif
  free(input);
  return 0;
}
