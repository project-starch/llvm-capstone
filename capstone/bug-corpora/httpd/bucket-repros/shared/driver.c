/* main(), the root pool, and the free() counter. One program per case. */
#include "corpus.h"

unsigned long freed_to_malloc;

_Noreturn void aprb_fail(unsigned code) {
  fprintf(stderr, "CONTROL-FAILED %u\n", code);
  exit(75);
}

int main(int argc, char **argv) {
  int fixed = argc > 1 && !strcmp(argv[1], "fixed");
  if (argc > 2 && atoi(argv[2]) != aprb_case_number) {
    fprintf(stderr, "CONTROL-FAILED fixture is case %d, run asked for %s\n",
            aprb_case_number, argv[2]);
    return 75;
  }
  if (apr_pool_initialize() != APR_SUCCESS)
    aprb_fail(700);
  apr_pool_t *root = NULL;
  if (apr_pool_create(&root, NULL) != APR_SUCCESS)
    aprb_fail(701);
  printf("case=%d arm=%s\n", aprb_case_number, fixed ? "fixed" : "buggy");
  int rc = aprb_case_run(fixed, root);
  apr_pool_destroy(root);
  apr_pool_terminate();
  return rc;
}
