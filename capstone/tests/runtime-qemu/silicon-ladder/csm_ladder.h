/* Wrap a csmith program as a silicon-ladder rung.
 *
 * The csmith campaign (capstone/tests/fuzz/) runs its programs in the QEMU-default ABI. A
 * rung built through this header runs the SAME program in the silicon configuration
 * (gp-captable, shrink off, no jump tables) so that a seed which matched its native
 * checksum on QEMU can be re-checked on the board with the ladder controller.
 *
 * Define CSM_PROGRAM to the csmith source before including. Build with the csmith runtime
 * flags the campaign uses (DOMAIN_EXTRA_CFLAGS / HOST_EXTRA_CFLAGS):
 *   -I capstone/tests/fuzz/csmith-rt -I <csmith>/include
 *   -include capstone/tests/fuzz/csmith-rt/capstone_platform.h -w -D_GNU_SOURCE
 * The checksum is what platform_main_end received; if main returned by another route the
 * rung reports 0xDEADC0DE, which no csmith checksum in the campaign equals. */
#include <stdint.h>
volatile uint32_t capstone_fuzz_checksum = 0;
volatile uint32_t capstone_fuzz_stage = 0;
#ifndef CSM_HOST
/* The platform header declares printf; with NOT_PRINT_CHECKSUM nothing calls it, but the
   domain is freestanding so give the reference a definition. */
int printf(const char *fmt, ...) { (void)fmt; return 0; }
#endif
#define main csmith_main
#include CSM_PROGRAM
#undef main
static unsigned csm_compute(void) {
  csmith_main();
  uint32_t r = capstone_fuzz_checksum;
  if (capstone_fuzz_stage != 2)
    r = 0xDEADC0DEu;
  return (unsigned)r;
}
