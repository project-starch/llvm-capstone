/* alias-store: the store twin of alias-load. A vp2 store at an out-of-window cursor: checked against the bounds decoded
 * at that cursor (and so landing a span away when that decode contains the cursor), or refused, or unchecked?
 * A 20,160-byte child (E=2, span 64 KiB) of a 64 KiB-aligned static arena; one STORE per process at
 *   capstone-vexec alias-store.dom OFFSET
 * Prints AS_PRE (cursor, the bounds LCC decodes for the offset pointer, the target byte before), stores 0x3c through
 * the offset pointer, then AS_STORED and AS_READBACK: the target byte read back through the arena's own in-bounds
 * capability, which shows whether the store landed. Every target lies inside the arena. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <capstone/capability.h>

static unsigned char arena[512 * 1024] __attribute__((aligned(65536)));

int main(int argc, char **argv)
{
    unsigned long off = argc > 1 ? strtoul(argv[1], 0, 0) : 100;
    arena[0x20000 + 0x10100] = 0xa5;
    arena[0x20000 + 0x16000] = 0x5a;
    unsigned char *parent = capstone_cap_derive(arena, 0, sizeof arena);
    unsigned char *child = capstone_cap_derive(parent, 0x20000, 20160);
    volatile unsigned long voff = off;
    unsigned char *p = child + voff;
    unsigned char *target = &arena[0x20000 + off];      /* the same address through the arena's own capability */
    printf("AS_PRE off=%#lx child=%#lx:%#lx cursor=%#lx decoded=%#lx:%#lx before=%u\n", off,
           (unsigned long)__builtin_capstone_cap_get_base(child), (unsigned long)__builtin_capstone_cap_get_end(child),
           (unsigned long)(uintptr_t)p, (unsigned long)__builtin_capstone_cap_get_base(p),
           (unsigned long)__builtin_capstone_cap_get_end(p), (unsigned)*(volatile unsigned char *)target);
    fflush(stdout);
    *(volatile unsigned char *)p = 0x3c;
    printf("AS_STORED off=%#lx\n", off);
    printf("AS_READBACK off=%#lx value=%u\n", off, (unsigned)*(volatile unsigned char *)target);
    printf("AS_DONE\n");
    return 0;
}
