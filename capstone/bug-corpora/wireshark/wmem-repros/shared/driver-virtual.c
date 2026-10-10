/* The corpus driver for the VIRTUAL Capstone process build of the wmem port (WM_SUBLET): the Sublet
 * region and chunk layers in a Linux process under capstone-vexec, the payload LENT by the virtual
 * heap as one linear capability (__capstone_sublet_malloc_linear) that the region layer carves and
 * revokes inside. This is the corpus's `virtual-nested-pools` arm.
 *
 * driver.c is compiled here unchanged, so every other build of it, and every image a committed
 * record cites, stays byte-identical. Only its hosted main() is replaced:
 *   - both modes exist: 0 narrows every object and revokes nothing, 1 also revokes at the
 *     allocator's own release points (a packet-pool reset, a chunk free);
 *   - the fix differential (`MODE N buggy|fixed`) runs in either mode, so the fixed sequence in
 *     mode 1 shows the protected build runs the corrected case to completion;
 *   - the payload comes from the virtual heap, not aligned_alloc.
 */
#define main wm_driver_hosted_main
#include "driver.c"
#undef main
#include <capstone/capability.h>

unsigned long __capstone_sublet_malloc_linear(size_t, capstone_cap_slot *);

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc < 3 || argc > 4 || (strcmp(argv[1], "0") && strcmp(argv[1], "1")))
    return 75;
  if ((unsigned)atoi(argv[2]) != wm_case_number) {
    fprintf(stderr, "CONTROL-FAILED fixture is case %u, run asked for %s\n",
            wm_case_number, argv[2]);
    return 75;
  }
  int differential = argc == 4;
  if (differential) {
    if (strcmp(argv[3], "buggy") && strcmp(argv[3], "fixed"))
      return 75;
    wm_fixed = !strcmp(argv[3], "fixed");
    wm_observe = 1;
  }
  unsigned mode = (unsigned)(argv[1][0] - '0');
  capstone_cap_slot lent;
  if (!__capstone_sublet_malloc_linear(WM_PAYLOAD_BYTES, &lent)) {
    fprintf(stderr, "CONTROL-FAILED the virtual heap lent no %lu-byte payload\n",
            (unsigned long)WM_PAYLOAD_BYTES);
    return 75;
  }
  wm_init_backing(NULL, capstone_cap_load(&lent), mode);
  start();
  wm_case_run();
  printf("WM_DEFECT case=%u mode=%u completed\n", wm_case_number, mode);
  if (differential) {
    if (!wm_fixed && wm_defect)
      printf("VERDICT DEFECT-REPRODUCED the access reached storage outside its object\n");
    else if (wm_fixed && !wm_defect)
      printf("VERDICT FIXED the fix's sequence does not reach another object's storage\n");
    else
      printf("VERDICT INCONCLUSIVE\n");
    return wm_fixed ? !!wm_defect : !wm_defect;
  }
  return 0;
}
