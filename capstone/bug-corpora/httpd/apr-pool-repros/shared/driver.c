/* main() and the root pool. One program per case. */
#include "corpus.h"

_Noreturn void apr_corpus_fail(unsigned code) {
  fprintf(stderr, "CONTROL-FAILED %u\n", code);
  exit(75);
}

int main(int argc, char **argv) {
  int fixed = argc > 1 && !strcmp(argv[1], "fixed");
  if (argc > 2 && atoi(argv[2]) != apr_case_number) {
    fprintf(stderr, "CONTROL-FAILED fixture is case %d, run asked for %s\n",
            apr_case_number, argv[2]);
    return 75;
  }
  if (apr_pool_initialize() != APR_SUCCESS)
    apr_corpus_fail(700);
  apr_pool_t *root = NULL;
  if (apr_pool_create(&root, NULL) != APR_SUCCESS)
    apr_corpus_fail(701);
  printf("case=%d arm=%s\n", apr_case_number, fixed ? "fixed" : "buggy");
  int rc = apr_case_run(fixed, root);
  apr_pool_destroy(root);
  apr_pool_terminate();
  return rc;
}
