/* Domain entry: run the CoreMark 1.01 matrix benchmark and return its crc16
   through the shared region cap `res`. Built with -capstone-gp-captable +
   shrink-off (the silicon config). */
#include "coremark_matrix_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = coremark_matrix_compute(); }
