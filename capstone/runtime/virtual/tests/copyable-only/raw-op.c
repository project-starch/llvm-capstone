/* Protected applications must never execute privileged raw tag transport. */
int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    switch (argv[1][0]) {
    case 'L': __asm__ volatile(".globl raw_site_L\nraw_site_L:\n.insn i 0x0b, 4, t0, zero, 0" ::: "t0", "memory"); break;
    case 'S': __asm__ volatile(".globl raw_site_S\nraw_site_S:\n.insn s 0x0b, 5, zero, 0(zero)" ::: "memory"); break;
    case 'Q': __asm__ volatile(".globl raw_site_Q\nraw_site_Q:\n.insn i 0x0b, 6, t0, zero, 0" ::: "t0", "memory"); break;
    case 'A': __asm__ volatile(".globl raw_site_A\nraw_site_A:\n.insn i 0x0b, 6, t0, zero, 1" ::: "t0", "memory"); break;
    default: return 2;
    }
    return 99; /* A successful raw operation is a failed test. */
}
