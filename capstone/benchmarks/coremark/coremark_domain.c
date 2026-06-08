#include "coremark_hostcall.h"

/* Shared regions received via REGION_SHARE invocations (first=metadata, second=payload). */
volatile struct hostcall_v0 *hc_metadata = 0;
volatile char               *hc_payload  = 0;
unsigned                     g_region_count = 0;

extern int coremark_main(void);

#define CAPSTONE_DPI_REGION_SHARE 1U

void domain_main(unsigned *res, unsigned func) {
  if (func == CAPSTONE_DPI_REGION_SHARE) {
    if (g_region_count == 0)
      hc_metadata = (volatile struct hostcall_v0 *)res;
    else if (g_region_count == 1)
      hc_payload = (volatile char *)res;
    ++g_region_count;
    return;
  }

  if (hc_metadata)
    hc_metadata->length = 0;

  coremark_main();
  *res = HC_V0_RET_DONE;
}
