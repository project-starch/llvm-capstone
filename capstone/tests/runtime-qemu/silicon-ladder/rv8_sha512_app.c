/* Domain entry: run RV8 sha512 and return the checksum through `res`. */
#include "rv8_sha512_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = sha512_compute(); }
