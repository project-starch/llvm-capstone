/* The campaign's FAULT positive control: a domain that deliberately dereferences
 * a null capability.  On this ISA a load through an untagged base raises a
 * capability fault; the emulator aborts the guest, and the batch runner must
 * record FAULT for this item, reboot, and still run the item after it.  A runner
 * that reports anything else for this domain cannot be trusted with a real one. */
#include <stdint.h>

volatile uint32_t *capstone_fuzz_null_probe = 0;

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  *res = *capstone_fuzz_null_probe;   /* faults here */
  *res = 0x0BADCAFEu;                  /* never reached */
}
