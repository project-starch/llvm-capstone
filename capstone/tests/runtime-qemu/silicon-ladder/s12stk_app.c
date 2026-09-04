/* QEMU-side variant of 's12stk'; writes *res only, per the ladder convention. */
#define S12SHAPE_RES0_ONLY 1
#define S12SHAPE_STACK_SLOT 1
#include "s12shape_kernel.h"
void domain_main(unsigned long *res, unsigned func){ (void)func; s12shape_run((volatile unsigned long *)res); }
