/* Root grants (module parameter root_grants=1): a mapping is a child of the one reference over
 * the address space, and the kernel ends it with CREVOKE on its copy of the mapping's reference.
 * run-root-grants.py runs this program once. Build with the virtual SDK.
 *
 *   grant_reads_and_writes       a mapped page reads zero and keeps what the program writes
 *   libc_accepts_copyable_grant  malloc of a small and of a large block and a second mmap work
 *   other_mapping_survives_unmap after the unmap of the first mapping the other ones still work
 *   stale access                 the last line reads through an object derived in the unmapped
 *                                page. It must fault with cause 25, so the program must not
 *                                reach ROOT_STALE_READ.
 *
 * The program maps the same address again before the stale access. The page is then mapped, so
 * an access that the processor lets through reads the new page and does not fault at all. That
 * is what the negative control shows: an adapter that skips the CREVOKE lets the read through.
 *
 * ROOT_GRANT_TYPE prints the type of a grant as the adapter issued it: 1 (non-linear) under
 * root_grants=1, 0 (linear) without.
 *
 * Mode `stale` (argv[1]): the program ends a mapping itself with CREVOKE and unmaps it
 * afterwards. The kernel's copy of that mapping's reference is then stale. The unmap must
 * succeed, since the lifetime has ended, and the kernel must not trap on the stale copy.
 *
 *   unmap_after_own_release                 directly after the release
 *   unmap_after_own_release_and_collection  after enough lifetimes ended for the adapter to
 *                                           collect identities (CAPSTONE_VM_STATS collections) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <capstone/capability.h>
#include "vm.h"

#define PAGE 4096UL
#define LARGE (1UL << 20)
#define ROUNDS 12000UL

static char *volatile stale;

static void fail(const char *what)
{
    fprintf(stderr, "ROOT_FAIL:%s\n", what);
    fflush(stderr);
    exit(1);
}

static void require(int ok, const char *what)
{
    if (!ok) fail(what);
}

static void pass(const char *check)
{
    printf("ROOT_CHECK:%s:PASS\n", check);
    fflush(stdout);
}

static unsigned long cursor(void *p)
{
    return __builtin_capstone_cap_get_cursor(p);
}

static void *map_page(void)
{
    char *page = mmap(NULL, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(page != MAP_FAILED, "mmap");
    page[0] = 1;
    return page;
}

static int mode_stale(void)
{
    char *keep = map_page();
    char *page = map_page();
    capstone_cap_release(page);
    require(!munmap(page, PAGE), "unmap of a mapping the program ended itself");
    require(keep[0] == 1, "the other mapping after that unmap");
    pass("unmap_after_own_release");

    page = map_page();
    capstone_cap_release(page);
    for (unsigned long i = 0; i < ROUNDS; ++i)
        capstone_cap_release(capstone_cap_derive(keep, 64, 64));
    require(!munmap(page, PAGE), "unmap of an ended mapping after a collection");
    keep[1] = 2;
    require(keep[0] == 1 && keep[1] == 2, "the other mapping after the collection");
    pass("unmap_after_own_release_and_collection");
    return 0;
}

int main(int argc, char **argv)
{
    if (argc > 1 && !strcmp(argv[1], "stale")) return mode_stale();

    sublet_cap slot;
    require(!cap_vm_acquire(&slot, PAGE, PAGE, PROT_READ | PROT_WRITE, 6, CAP_VM_HEAP), "grant");
    printf("ROOT_GRANT_TYPE:%lu\n", sublet_type(&slot));
    char *page = cap_vm_copyable(&slot);
    unsigned long address = cursor(page);

    require(page[0] == 0 && page[PAGE - 1] == 0, "a fresh page reads zero");
    memset(page, 0x5a, PAGE);
    require(page[0] == 0x5a && page[PAGE - 1] == 0x5a, "the page keeps what the grant wrote");
    pass("grant_reads_and_writes");

    char *small = malloc(48);
    char *large = malloc(LARGE);
    char *other = mmap(NULL, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(small && large, "malloc");
    require(other != MAP_FAILED, "second mmap");
    memset(small, 0x11, 48);
    memset(large, 0x22, LARGE);
    memset(other, 0x44, PAGE);
    pass("libc_accepts_copyable_grant");

    char *object = capstone_cap_derive(page, 64, 64);
    object[0] = 0x33;
    require(page[64] == 0x33, "the object lies in the page");
    stale = object;

    require(!munmap(page, PAGE), "munmap");
    require(small[47] == 0x11 && large[LARGE - 1] == 0x22 && other[PAGE - 1] == 0x44,
            "the other mappings keep their contents");
    other[0] = 0x45;
    large[0] = 0x23;
    require(other[0] == 0x45 && large[0] == 0x23, "the other mappings take a write");
    pass("other_mapping_survives_unmap");

    /* The launcher places the next mapping of this size at the address it unmapped last. */
    char *again = mmap(NULL, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(again != MAP_FAILED, "mmap after the unmap");
    again[64] = 0x77;
    printf("ROOT_REMAP same_address=%d\n", cursor(again) == address);
    fflush(stdout);

    char seen = *(volatile char *)stale;
    printf("ROOT_STALE_READ value=0x%x\n", seen);
    return 0;
}
