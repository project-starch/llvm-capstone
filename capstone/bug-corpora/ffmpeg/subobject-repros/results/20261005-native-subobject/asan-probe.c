/* Is ASan blind to a member-to-member crossing inside ONE allocation?
 * Two arms over the same shape, so the answer is two-sided:
 *   inside   write index 600 of a uint16_t[600] that is followed by another
 *            member -- the sub-object crossing the corpus case reproduces
 *   past     write past the END of the whole allocation -- the POSITIVE
 *            CONTROL, which ASan must report, or its silence above says
 *            nothing about the subject and something about the instrument */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#define SEGMENTS 600
struct pic_timing {
  uint16_t num_nalus_in_du_minus1[SEGMENTS];
  uint32_t du_cpb_removal_delay_increment_minus1[SEGMENTS];
};
int main(int argc, char **argv) {
  int past = argc > 1 && !strcmp(argv[1], "past");
  struct pic_timing *pt = malloc(sizeof *pt);
  if (!pt) return 75;
  memset(pt, 0, sizeof *pt);
  if (past) {
    /* one uint16_t past the ALLOCATION */
    volatile uint16_t *p = (volatile uint16_t *)((char *)pt + sizeof *pt);
    *p = 0x4141;
    printf("  (read back 0x%04x)\n", *p);
    printf("ARM past: wrote one uint16_t past the allocation\n");
  } else {
    /* index 600 of a [600] array, i.e. the next MEMBER, inside the allocation */
    *(volatile uint16_t *)&pt->num_nalus_in_du_minus1[SEGMENTS] = 0x4141;
    printf("ARM inside: wrote index %d of num_nalus_in_du_minus1[%d]; "
           "du_cpb_removal_delay_increment_minus1[0] is now 0x%08x\n",
           SEGMENTS, SEGMENTS, pt->du_cpb_removal_delay_increment_minus1[0]);
  }
  free(pt);
  return 0;
}
