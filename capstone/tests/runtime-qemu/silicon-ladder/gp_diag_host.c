/* Native oracle for the gp-captable diagnostic rung: the checksum the domain must
 * return, plus each probe's expected raw value (so a board capture can be read
 * against this without recomputing anything by hand). */
#include <stdio.h>
#include "gp_diag_kernel.h"
int main(void) {
  printf("%u\n", gpd_compute());
  return 0;
}
