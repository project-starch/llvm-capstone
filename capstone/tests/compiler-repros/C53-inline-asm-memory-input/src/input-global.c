unsigned short cw;
void f(void) { __asm__ __volatile__("lhu zero, %0" : : "m"(cw)); }
