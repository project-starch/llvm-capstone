__attribute__((noreturn)) void r0_c_entry(volatile unsigned long *data)
{
    data[0] = 0x123;
    data[512] = 0x123;
    __asm__ volatile("li a0, 42\necall" : : : "a0", "memory");
    __builtin_unreachable();
}
