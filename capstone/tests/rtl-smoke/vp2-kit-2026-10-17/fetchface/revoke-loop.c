/* Fetch-face discriminator pair: one source, two images that differ only in the #ifdef around the inner loop of
 * node-invalidating lifetime ops (CDERIVE a child, CREVOKE it). Each round runs that loop, then spins long enough
 * to take quantum escapes, so every escape follows invalidating ops (WITH_REVOKES) or none (without).
 *   capstone-vexec revoke-loop[-ops].dom ROUNDS INNER SPIN     (defaults 200 64 2000000)
 * No malloc/free: the parent is derived from a static arena, so the runtime's own lifetime ops are never executed. */
#include <stdio.h>
#include <stdlib.h>
#include <capstone/capability.h>

static unsigned char arena[4096] __attribute__((aligned(4096)));

int main(int argc, char **argv)
{
    unsigned long rounds = argc > 1 ? strtoul(argv[1], 0, 0) : 200;
    unsigned long inner = argc > 2 ? strtoul(argv[2], 0, 0) : 64;
    unsigned long spin = argc > 3 ? strtoul(argv[3], 0, 0) : 2000000;
    volatile unsigned long sink = 0;
    unsigned long ops = 0;
    unsigned char *parent = capstone_cap_derive(arena, 0, sizeof arena);
    printf("FF_BEGIN rounds=%lu inner=%lu spin=%lu ops_compiled=%d\n", rounds, inner, spin,
#ifdef WITH_REVOKES
           1
#else
           0
#endif
    );
    for (unsigned long r = 0; r < rounds; ++r) {
#ifdef WITH_REVOKES
        for (unsigned long i = 0; i < inner; ++i) {
            unsigned char *child = capstone_cap_derive(parent, (i % 64) * 64, 64);
            child[0] = (unsigned char)i;
            capstone_cap_revoke_child(parent, child);
            ++ops;
        }
#endif
        for (unsigned long s = 0; s < spin; ++s)
            sink += s ^ r;
    }
    printf("FF_DONE rounds=%lu revoke_pairs=%lu sink=%lu\n", rounds, ops, (unsigned long)sink);
    return 0;
}
