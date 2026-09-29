unsigned short f(void) { unsigned short cw; __asm__ __volatile__("lhu zero, %0" : "=m"(cw)); return cw; }
