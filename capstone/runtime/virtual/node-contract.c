/* Node capacity is independent of the malloc block-record limit. Build with
 * the virtual SDK. Keep 70,000 structural nodes live, then revoke and reuse.
 *
 * With root_grants the grant is a copyable child of the address-space reference
 * (type 1, not linear), and the linear discipline of the default path (MREV,
 * SPLIT, INIT) does not apply to it. The same population is then made with
 * CDERIVE and ended with CREVOKE (growth_derived): an ancestor object over the
 * whole page, one live object derived from it before the growth, and the
 * structural nodes as children whose references are dropped. The output lines
 * are the same, so run-nodes.py reads both forms. */
#include <capstone/capability.h>
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

/* Mode stale: the access through the object from before the revoke, which must fault at the
 * symbol cap_node_stale (run-nodes.py). One site for both forms. */
static __attribute__((noinline)) void stale_access(void)
{
    puts("NODE_STALE_ACCESS"); fflush(stdout);
    __asm__ volatile(".global cap_node_stale\ncap_node_stale:\nlbu zero, 0(%0)"
                     : : "r"(stale) : "memory");
    require(0, "stale capability resurrected");
}

/* The root_grants form of the growth and stale modes. */
static int growth_derived(sublet_cap *region, const char *mode)
{
    char *page = cap_vm_copyable(region);
    for (unsigned round = 0; round < 2; ++round) {
        char *ancestor = capstone_cap_derive(page, 0, 4096);
        if (!round) {
            stale = capstone_cap_derive(ancestor, 2048, 2048);
            stale[0] = 42;
        }
        for (unsigned long i = 0; i < NODES; ++i)
            /* Losing the reference does not end its lifetime. */
            (void)capstone_cap_derive(ancestor, 0, 64);
        if (!round) require(stale[0] == 42, "old live capability after growth");
        printf("NODE_LIVE:%u:%lu\n", round, NODES); fflush(stdout);
        capstone_cap_release(ancestor);
        require(!__capstone_vm_nodes(8192), "sweep and refill");
    }
    if (!strcmp(mode, "stale")) stale_access();
    puts("NODE_GROWTH_OK");
    return 0;
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
    if (sublet_type(&region) == SUBLET_TYPE_NONLIN) return growth_derived(&region, mode);
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
    if (!strcmp(mode, "stale")) stale_access();
    puts("NODE_GROWTH_OK");
    return 0;
}
