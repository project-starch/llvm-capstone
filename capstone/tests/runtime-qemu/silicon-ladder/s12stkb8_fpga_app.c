/* Board variant of rung 's12stkb8': the S-12 shape with slots taken from the STACK rather than
   from a cap-table global, matching SQLite's frame-offset slots. s12shape_kernel.h has
   the verdict table; a returned word beginning 0xF is the in-domain trap report.
   Slot provenance is the largest remaining structural difference from SQLite's window
   after shape, registers, domain context and store pressure were all eliminated. */
#define S12SHAPE_STACK_SLOT 1
#define S12SHAPE_BURST 8
#include "s12shape_kernel.h"
void domain_main(unsigned long *res, unsigned func){ (void)func; s12shape_run((volatile unsigned long *)res); }
