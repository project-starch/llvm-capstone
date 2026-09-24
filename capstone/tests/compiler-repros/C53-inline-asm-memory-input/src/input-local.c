void f(void) { unsigned short cw = 1; __asm__ __volatile__("lhu zero, %0" : : "m"(cw)); }
