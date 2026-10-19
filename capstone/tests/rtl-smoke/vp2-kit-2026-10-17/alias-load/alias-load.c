/* alias-load: is a vp2 load at an out-of-window cursor CHECKED against the bounds decoded at that cursor, or not
 * checked at all? A 20,160-byte child (E=2, span 64 KiB) of a 64 KiB-aligned static arena; one load per process at
 *   capstone-vexec alias-load.dom OFFSET      (OFFSET from the child's base, decimal or 0x)
 * Prints AL_PRE with the cursor and the bounds LCC decodes for the offset pointer, then AL_LOADED with the byte, then
 * AL_DONE. A fault in between ends the process; the launcher's fault line carries the cause. Every target lies
 * inside the arena, so a load that is not refused reads mapped memory. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <capstone/capability.h>

static unsigned char arena[512 * 1024] __attribute__((aligned(65536)));

int main(int argc, char **argv)
{
    unsigned long off = argc > 1 ? strtoul(argv[1], 0, 0) : 100;
    for (unsigned long i = 0; i < sizeof arena; i += 4096) arena[i] = (unsigned char)(i >> 12);
    arena[0x20000 + 0x10100] = 0xa5;   /* the alias arm's target: a load that lands reads 165 */
    arena[0x20000 + 0x16000] = 0x5a;   /* the outside arm's target: an unchecked load would read 90 */
    unsigned char *parent = capstone_cap_derive(arena, 0, sizeof arena);
    unsigned char *child = capstone_cap_derive(parent, 0x20000, 20160);
    volatile unsigned long voff = off;
    unsigned char *p = child + voff;
    printf("AL_PRE off=%#lx child=%#lx:%#lx cursor=%#lx decoded=%#lx:%#lx\n", off,
           (unsigned long)__builtin_capstone_cap_get_base(child), (unsigned long)__builtin_capstone_cap_get_end(child),
           (unsigned long)(uintptr_t)p, (unsigned long)__builtin_capstone_cap_get_base(p),
           (unsigned long)__builtin_capstone_cap_get_end(p));
    fflush(stdout);
    unsigned char v = *(volatile unsigned char *)p;
    printf("AL_LOADED off=%#lx value=%u\n", off, (unsigned)v);
    printf("AL_DONE\n");
    return 0;
}
