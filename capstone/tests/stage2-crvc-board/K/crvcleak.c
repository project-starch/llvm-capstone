/* crvcleak: an ORDINARY (unprotected) program for K's leak mutant. It sleeps so that switch_mm runs for it (where
 * SUBLET_MUTANT_CRVC_LEAK writes srevroot = 1, the mode bit with no table, for a task named crvcleak*), then runs one
 * c.fsd and one c.fld (Q0 101/001). Standard decode stores and loads the double; decoded in the CRVC mode they would be
 * c.stc/c.ldc through an integer and fault. */
#include <stdio.h>
#include <time.h>
static double src = 1.5, dst;
int main(void) {
    struct timespec ts = {0, 50000000};
    for (int i = 0; i < 4; i++) nanosleep(&ts, 0);
    register double *s __asm__("a0") = &src;
    register double *d __asm__("a1") = &dst;
    __asm__ volatile(".option push\n\t.option rvc\n\tc.fld fa0, 0(%0)\n\tc.fsd fa0, 0(%1)\n\t.option pop" : : "r"(s), "r"(d) : "fa0", "memory");
    printf("crvcleak: c.fld/c.fsd moved %g: %s\n", dst, dst == 1.5 ? "standard RVC" : "WRONG");
    return dst == 1.5 ? 0 : 3;
}
