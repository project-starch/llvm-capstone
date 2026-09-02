/* QEMU-side variant of rung 's12sb32'. See s12shape_kernel.h. Writes *res only. */
#define S12SHAPE_RES0_ONLY 1
#define S12SHAPE_BURST 32
#include "s12shape_kernel.h"
void domain_main(unsigned long *res, unsigned func){ (void)func; s12shape_run((volatile unsigned long *)res); }
