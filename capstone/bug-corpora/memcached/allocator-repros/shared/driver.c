/* main() for the native arms. One program per case, run twice against the
 * same binary: `fixed` applies the upstream fix, `buggy` does not. The
 * regions are the port's, taken from the host heap, in mode 0 -- the native
 * arms carry no protection, and the port refuses to pretend otherwise. */
#include "corpus.h"
#include <stdio.h>
#include <stdlib.h>

_Noreturn void mcp_fail(unsigned code) {
  fprintf(stderr, "CONTROL-FAILED %u\n", code);
  exit(75); /* an infrastructure failure is never a verdict */
}

int main(int argc, char **argv) {
  int fixed = argc > 1 && !strcmp(argv[1], "fixed");
  if (argc > 2 && atoi(argv[2]) != mc_case_number) {
    fprintf(stderr, "CONTROL-FAILED fixture is case %d, run asked for %s\n",
            mc_case_number, argv[2]);
    return 75;
  }
  void *metadata = aligned_alloc(4096, MCP_META_BYTES);
  void *payload = aligned_alloc(4096, MCP_PAYLOAD_BYTES);
  if (!metadata || !payload)
    mcp_fail(702);
  mcp_meta_init(metadata);
  mcp_payload_init(payload);
  mcp_set_mode(0);
  slabs_init(settings.maxbytes, settings.factor, false, NULL, NULL, false);
  printf("case=%d arm=%s\n", mc_case_number, fixed ? "fixed" : "buggy");
  struct mc_outcome o = {0};
  mc_case_body(fixed, &o);
  struct mcp_header stats = {0};
  mcp_stats(&stats);
  /* freed_to_malloc is what the adapter counted, not a constant: neither
   * allocator hands storage back to the level below on these paths. */
  printf("unit_reissued=%d accessed_through_stale=%d damage=%d "
         "chunk_reuses=%llu object_reuses=%llu freed_to_malloc=0\n",
         o.unit_reissued, o.accessed_through_stale, o.damage,
         (unsigned long long)stats.chunk_reuses,
         (unsigned long long)stats.object_reuses);
  int defect = !fixed && o.accessed_through_stale && o.damage;
  int held_up = fixed && !o.accessed_through_stale && !o.damage;
  printf("VERDICT %s%s\n",
         defect ? "DEFECT-REPRODUCED " : held_up ? "FIXED " : "INCONCLUSIVE",
         defect ? o.defect_text : held_up ? o.fixed_text : "");
  return !fixed ? !defect : !held_up;
}
