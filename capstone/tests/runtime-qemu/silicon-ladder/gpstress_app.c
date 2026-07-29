/* Domain entry for the descriptor-stress rung: exercises every glue path SQLite needs
   (zero-fill, bulk copy, byte tail, >2040 B global, private .L symbol) in the smallest
   domain that can. See gpstress_kernel.h. Oracle 43662404. */
#include "gpstress_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = gpstress_compute(); }
