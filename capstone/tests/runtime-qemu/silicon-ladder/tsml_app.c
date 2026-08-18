/* MATCHED PARTNER to tagsweep. Identical source and identical TOTAL check count; the ONLY
 * difference is the buffer footprint. 512 slots x 16 B = 8 KiB fits inside the 32 KiB D-cache,
 * so after the first rep the reloads HIT instead of refilling from DRAM, whereas tagsweep's
 * 64 KiB cannot stay resident and every reload is a miss refill.
 * If tag loss appears only in the big arm, it is localized to the refill path -- which is what
 * the measured wedge implicates (src=1, MISS REFILL). A one-variable pair localizes better than
 * a long ladder, because the difference between the arms IS the variable. */
#define TAGSWEEP_SEED 0u
#define TAGSWEEP_N 512u
#include "tagsweep_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = tagsweep_compute(); }
