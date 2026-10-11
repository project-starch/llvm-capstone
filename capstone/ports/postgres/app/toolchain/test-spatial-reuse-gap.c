#include "spatial-reuse-gap.h"
#include <assert.h>

int main(void) {
  pg_spatial_gap_issue(1, 0x1000, 16);
  pg_spatial_gap_issue(2, 0x2000, 32);
  pg_spatial_gap_resize(0x1000, 24);
  pg_spatial_gap_reset(1);
  pg_spatial_gap_issue(2, 0x1000, 16);
  pg_spatial_gap_release(0x2000);
  pg_spatial_gap_reset(2);
  pg_spatial_gap_issue(1, 0x2000, 32);
  pg_reuse_gap_report();
  return 0;
}
