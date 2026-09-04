/* Domain entry for a csmith (or yarpgen) program.
 *
 * The domain ABI is start.S's: domain_main(unsigned *res, unsigned func), result
 * written through res, 32 bits.  A csmith program's entire observable behaviour is
 * the CRC it hands to platform_main_end, so that is what comes back.
 *
 * FUZZ_XOR: the campaign's positive control.  A domain built with -DFUZZ_XOR=1
 * returns the checksum with one bit flipped, so the comparison against the native
 * reference MUST report a mismatch for it; a harness that reports MATCH for that
 * build is not comparing anything.
 */
#include <stdint.h>

volatile uint32_t capstone_fuzz_checksum = 0;
volatile uint32_t capstone_fuzz_stage = 0;

int printf(const char *fmt, ...) {
  (void)fmt;
  return 0;
}

extern int main(void);

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  main();
  uint32_t r = capstone_fuzz_checksum;
  if (capstone_fuzz_stage != 2)
    r = 0xDEADC0DEu;   /* platform_main_end never ran: main returned another way */
#ifdef FUZZ_XOR
  r ^= 1u;
#endif
  *res = r;
}
