void f(unsigned int *p) { __asm__ __volatile__("sw zero, %0" : "=m"(*p)); }
