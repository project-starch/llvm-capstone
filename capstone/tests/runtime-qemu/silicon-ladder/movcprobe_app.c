#include "movcprobe_kernel.h"
void domain_main(unsigned *res, unsigned func){ (void)func; *res = movcprobe_compute(); }
