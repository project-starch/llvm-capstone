/* Node capacity is independent of the malloc block-record limit. Build with
 * the virtual SDK. Keep 70,000 structural nodes live, then revoke and reuse. */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vm.h"

/* vp2 caps the flat lifetime table at 65536 records (a 16-bit node index), which 70000 exceeds
 * by construction. The count is a parameter so the same source runs at 70000 under QEMU, where
 * the namespace is wider, and at 60000 on silicon. CAPSTONE_NODE_COUNT overrides it; the default
 * is unchanged, so an existing invocation behaves exactly as before. The count used is PRINTED,
 * and run-nodes.py records it -- a run whose count is not in the record cannot be compared with
 * one whose count differed. */
static unsigned long node_count(void)
{
    const char *e = getenv("CAPSTONE_NODE_COUNT");
    unsigned long n = 0;
    if (e && *e) { char *end = 0; n = strtoul(e, &end, 10); if (end && *end) n = 0; }
    return n ? n : 70000UL;
}
#define NODES (node_count())
static unsigned char *volatile stale;
static void require(int ok, const char *message)
{
    if (!ok) { fprintf(stderr, "NODE_FAIL:%s\n", message); exit(1); }
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "growth";
    if (!strcmp(mode, "budget")) {
        unsigned char *p = malloc(64);
        require(p != NULL, "initial malloc"); p[0] = 42;
        require(__capstone_vm_nodes(NODES) == -ENOMEM, "quota errno");
        require(p[0] == 42, "authority after rejected growth");
        p[63] = 17; require(p[63] == 17, "writable after rejected growth");
        free(p);
        puts("NODE_BUDGET_OK");
        return 0;
    }
    sublet_cap region, ancestor, tail, handle;
    require(!cap_vm_acquire(&region, 4096, 4096, PROT_READ | PROT_WRITE,
                            6, CAP_VM_HEAP), "linear root");
    for (unsigned round = 0; round < 2; ++round) {
        sublet_handle(&region, &ancestor);
        if (!round) {
            sublet_split(&region, sublet_base(&region) + 2048, &tail);
            stale = sublet_take(&tail);
            stale[0] = 42;
        }
        for (unsigned long i = 0; i < NODES; ++i) {
            sublet_handle(&region, &handle);
            /* Losing the REV handle does not retire its structural node. */
            sublet_clear(&handle);
        }
        if (!round) require(stale[0] == 42, "old live capability after growth");
        printf("NODE_LIVE:%u:%lu\n", round, NODES); fflush(stdout);
        sublet_give_to(&ancestor, &region);
        /* One growth batch adds at most 4096 slots in this gate. Asking for
         * more forces the retired population through the complete sweep. */
        require(!__capstone_vm_nodes(8192), "sweep and refill");
    }
    if (!strcmp(mode, "stale")) {
        puts("NODE_STALE_ACCESS"); fflush(stdout);
        __asm__ volatile(".global cap_node_stale\ncap_node_stale:\nlbu zero, 0(%0)"
                         : : "r"(stale) : "memory");
        require(0, "stale capability resurrected");
    }
    puts("NODE_GROWTH_OK");
    return 0;
}
