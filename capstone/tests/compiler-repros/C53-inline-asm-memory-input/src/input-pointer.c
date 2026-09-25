void f(unsigned int *p) { __asm__ __volatile__("lw zero, %0" : : "m"(*p)); }
