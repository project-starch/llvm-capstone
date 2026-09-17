static void node_snapshot(void)
{
    unsigned long mark = 0xff20000000000000UL | cursor;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x43, x0, %0, x0\n"
                     ".insn r 0x5b, 0x1, 0x46, x0, x0, x0\n" : : "r"(mark) : "memory");
}
