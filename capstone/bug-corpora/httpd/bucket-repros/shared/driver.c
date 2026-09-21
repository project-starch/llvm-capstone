/* main() for the native arms. One program per case, run twice against the
 * same binary: `fixed` applies the upstream fix, `buggy` does not. The
 * regions are the port's, taken from the host heap, in mode 0 -- the native
 * arms carry no protection, and the pointer adapters refuse to pretend
 * otherwise. Nothing here reaches malloc: a node goes to the level below only
 * in apr_allocator_destroy, and the adapter counts what it did instead. */
#include "corpus.h"
#include <stdio.h>
#include <stdlib.h>

_Noreturn void aprp_fail(unsigned code) {
  fprintf(stderr, "CONTROL-FAILED %u\n", code);
  exit(75); /* an infrastructure failure is never a verdict */
}

int main(int argc, char **argv) {
  int fixed = argc > 1 && !strcmp(argv[1], "fixed");
  if (argc > 2 && atoi(argv[2]) != aprb_case_number) {
    fprintf(stderr, "CONTROL-FAILED fixture is case %d, run asked for %s\n",
            aprb_case_number, argv[2]);
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
    aprp_fail(703);
  apr_pool_t *root = NULL;
  if (apr_pool_create(&root, NULL) != APR_SUCCESS)
    aprp_fail(704);
  printf("case=%d arm=%s\n", aprb_case_number, fixed ? "fixed" : "buggy");
  struct aprb_outcome o = {.now = -1};
  aprb_case_body(fixed, root, &o);
  apr_pool_destroy(root);
  apr_pool_terminate();
  struct aprp_header stats = {0};
  aprp_stats(&stats);
  unsigned long pieces, reissues, files;
  aprb_stats(&pieces, &reissues, &files);
  printf("through_dead_allocator=%d still_held=%d reissued_same_address=%d "
         "read_after_destroy=%d now=%d node_reuses=%llu node_discards=%llu "
         "pieces=%lu files=%lu reissues=%lu\n",
         o.through_dead_allocator, o.still_held, o.reissued_same_address,
         o.read_after_destroy, o.now, (unsigned long long)stats.node_reuses,
         (unsigned long long)stats.node_discards, pieces, files, reissues);
  int defect = !fixed && o.defect, held_up = fixed && o.held_up;
  printf("VERDICT %s%s\n",
         defect ? "DEFECT-REPRODUCED " : held_up ? "FIXED " : "INCONCLUSIVE",
         defect ? o.defect_text : held_up ? o.fixed_text : "");
  return !fixed ? !defect : !held_up;
}
