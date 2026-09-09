/* Is AddressSanitizer blind to a use-after-free inside a NESTED allocator?
 *
 * Same shape as MicroPython: one region obtained once, then sub-allocated in
 * software. asan_control.c already showed ASan catching an ordinary malloc UAF
 * in this exact toolchain, so silence here is about the allocator, not the tool.
 */
#include <stdlib.h>
#include <stdio.h>

static unsigned char *region;      /* the "GC heap": ONE allocation */
static unsigned long bump;

static unsigned char *sub_alloc(unsigned long n) {
    unsigned char *p = region + bump;
    bump += n;
    return p;
}
static void sub_free(unsigned char *p) { (void)p; }   /* bookkeeping only */

int main(void) {
    region = malloc(4096);          /* ASan knows about THIS block, and only this */

    unsigned char *a = sub_alloc(64);
    a[0] = 0x11;
    sub_free(a);                    /* a is dead */
    bump = 0;                       /* sweep: storage returns to the allocator */

    unsigned char *b = sub_alloc(64);   /* b gets a's storage */
    b[0] = 0x22;
    a[0] = 0xAA;                    /* USE AFTER FREE, write through the dead pointer */

    printf("nested UAF completed, b[0] = 0x%02x (0xaa means it corrupted b)\n", b[0]);
    free(region);
    return 0;
}
