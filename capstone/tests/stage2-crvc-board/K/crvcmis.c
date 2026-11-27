/* crvcmis: stage 2's firmware positive control. One capability load 8 bytes past a 16-byte boundary, inside its
 * object's bounds, in a CRVC-flagged image: in the mode it is c.ldc, which traps misaligned (cause 4) to M. With the
 * redirect it reaches S and the process dies; emulated as C.FLD it would load into an FP register and go on. */
#include <stdio.h>
struct S { void *a[4]; } __attribute__((aligned(16)));
static struct S s;
int main(int argc, char **argv) {
    char *base = (char *)&s;
    void *volatile *p = (void *volatile *)(base + 8 * (argc > 0));
    printf("crvcmis: loading a capability at offset %d of a 16-byte-aligned object\n", (int)((char *)p - base));
    /* c.ldc a1, 0(a0): Q0 funct3 001, rs1' = a0, rd' = a1. In the mode a capability load; standard decode would
     * read it as c.fld fa1, 0(a0). The compiler may pick a 32-bit ldc, so the form is fixed here. */
    void *v;
    __asm__ volatile("movc a0, %1\n\t.insn 2, 0x210c\n\tmovc %0, a1" : "=r"(v) : "r"(p) : "a0", "a1", "memory");
    printf("crvcmis: no trap, the load returned (%s)\n", v ? "non-null" : "null");
    return 0;
}
