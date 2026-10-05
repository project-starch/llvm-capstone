/* main() for the plane corpus. One program per case, run twice against the same
 * binary: the control arm (fixed) first. The frame and its single AVBuffer come
 * from real libavutil -- a case reduces its consumer, never the allocator.
 */
#include "corpus.h"

_Noreturn void ffp_fail(unsigned code) {
  fprintf(stderr, "CONTROL-FAILED %u\n", code);
  exit(75); /* an infrastructure failure is never a verdict */
}

int main(int argc, char **argv) {
  int fixed = argc > 1 && !strcmp(argv[1], "fixed");
  if (argc > 2 && atoi(argv[2]) != ffp_case_number) {
    fprintf(stderr, "CONTROL-FAILED fixture is case %d, run asked for %s\n",
            ffp_case_number, argv[2]);
    return 75;
  }
  struct ffp_outcome o = {0};
  printf("case=%d arm=%s\n", ffp_case_number, fixed ? "fixed" : "buggy");
  ffp_case_run(fixed, &o);
  printf("crossed=%d contained=%d damage=%d plane_slack=%ld\n",
         o.crossed, o.contained, o.damage, o.plane_slack);
  if (!fixed && o.crossed && o.contained && o.damage)
    printf("VERDICT DEFECT-REPRODUCED %s\n", o.defect_text);
  else if (fixed && !o.crossed && !o.damage)
    printf("VERDICT FIXED %s\n", o.fixed_text);
  else
    printf("VERDICT INCONCLUSIVE\n");
  return fixed ? (o.crossed || o.damage) : !(o.crossed && o.contained && o.damage);
}
