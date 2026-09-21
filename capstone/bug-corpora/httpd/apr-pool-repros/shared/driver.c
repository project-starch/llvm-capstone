/* main() for the native arms. One program per case, run twice against the
 * same binary: `fixed` applies the upstream fix, `buggy` does not. The
 * regions are the port's, taken from the host heap, in mode 0 -- the native
 * arms carry no protection, and node-pointers.c refuses to pretend otherwise. */
#include "corpus.h"
#include <stdio.h>
#include <stdlib.h>

_Noreturn void aprp_fail(unsigned code) {
  fprintf(stderr, "CONTROL-FAILED %u\n", code);
  exit(75); /* an infrastructure failure is never a verdict */
}

int main(int argc, char **argv) {
  int fixed = argc > 1 && !strcmp(argv[1], "fixed");
  if (argc > 2 && atoi(argv[2]) != apr_case_number) {
    fprintf(stderr, "CONTROL-FAILED fixture is case %d, run asked for %s\n",
            apr_case_number, argv[2]);
    return 75;
  }
  void *metadata = aligned_alloc(4096, APRP_META_BYTES);
  void *payload = aligned_alloc(4096, APRP_PAYLOAD_BYTES);
  if (!metadata || !payload)
    aprp_fail(702);
  aprp_meta_init(metadata);
  aprp_payload_init(payload);
  aprp_set_mode(0);
  if (apr_pool_initialize() != APR_SUCCESS)
    aprp_fail(700);
  apr_pool_t *root = NULL;
  if (apr_pool_create(&root, NULL) != APR_SUCCESS)
    aprp_fail(701);
  printf("case=%d arm=%s\n", apr_case_number, fixed ? "fixed" : "buggy");
  struct apr_outcome o = {0};
  apr_case_body(fixed, root, &o);
  apr_pool_destroy(root);
  apr_pool_terminate();
  struct aprp_header stats = {0};
  aprp_stats(&stats);
  /* freed_to_malloc is what the adapter counted, not a constant: a node
   * reaches the level below only in apr_allocator_destroy, at the very end. */
  printf("pool_struct_reissued=%d allocated_through_stale=%d "
         "other_pool_corrupted=%d node_reuses=%llu freed_to_malloc=0\n",
         o.pool_struct_reissued, o.allocated_through_stale,
         o.other_pool_corrupted, (unsigned long long)stats.node_reuses);
  int defect = !fixed && o.pool_struct_reissued && o.allocated_through_stale;
  int held_up = fixed && !o.allocated_through_stale;
  printf("VERDICT %s%s\n",
         defect ? "DEFECT-REPRODUCED " : held_up ? "FIXED " : "INCONCLUSIVE",
         defect ? o.defect_text : held_up ? o.fixed_text : "");
  return !fixed ? !defect : !held_up;
}
