/* main() for the VIRTUAL Capstone process build with the Sublet protection (MCP_SUBLET): the corpus's
 * `virtual-nested-pools` arm. slabs.c and cache.c carry the application's patch 0006 and no hooks:
 * their pages and objects come from virtual mallocng, as upstream's do from malloc, and every chunk or
 * object they hand out is a child lifetime (CDERIVE) that slabs_free or cache_free revokes (CREVOKE).
 * There is no ledger, no payload region and no mode.
 *
 * The result lines are driver.c's, so one observer reads every arm; the ledger's reuse counters, which
 * this build does not have, are left out. Run as `fixed|buggy N`. */
#include "corpus.h"
#include <stdio.h>
#include <stdlib.h>

_Noreturn void mcp_fail(unsigned code) {
  fprintf(stderr, "CONTROL-FAILED %u\n", code);
  exit(75); /* an infrastructure failure is never a verdict */
}

int main(int argc, char **argv) {
  if (argc != 3 || (strcmp(argv[1], "fixed") && strcmp(argv[1], "buggy"))) {
    fprintf(stderr, "CONTROL-FAILED usage: fixed|buggy N\n");
    return 75;
  }
  int fixed = !strcmp(argv[1], "fixed");
  if (atoi(argv[2]) != mc_case_number) {
    fprintf(stderr, "CONTROL-FAILED fixture is case %d, run asked for %s\n",
            mc_case_number, argv[2]);
    return 75;
  }
  slabs_init(settings.maxbytes, settings.factor, false, NULL, NULL, false);
  printf("case=%d arm=%s\n", mc_case_number, fixed ? "fixed" : "buggy");
  struct mc_outcome o = {0};
  mc_case_body(fixed, &o);
  printf("unit_reissued=%d accessed_through_stale=%d damage=%d\n",
         o.unit_reissued, o.accessed_through_stale, o.damage);
  int defect = !fixed && o.accessed_through_stale && o.damage;
  int held_up = fixed && !o.accessed_through_stale && !o.damage;
  printf("VERDICT %s%s\n",
         defect ? "DEFECT-REPRODUCED " : held_up ? "FIXED " : "INCONCLUSIVE",
         defect ? o.defect_text : held_up ? o.fixed_text : "");
  return !fixed ? !defect : !held_up;
}
