/* Root grants (module parameter root_grants=1): a mapping is a child of the one reference over
 * the address space, and a program's munmap ends it only through the reference the program hands
 * back (stage 2 of docs/plans/2026-11-12-address-space-root-adapter.md). The kernel accepts the
 * reference only if it is a live reference to the mapping itself whose bounds cover the whole
 * mapping, and then ends the mapping with CREVOKE. run-root-grants.py runs this program once per
 * mode (argv[1]) in one boot. Build with the virtual SDK.
 *
 * Every mode first reads the type of a grant as the adapter issues it (ROOT_GRANT_TYPE): 1
 * (non-linear) under root_grants=1, 0 (linear) without. A check that holds only with root grants
 * prints ROOT_SKIP with the reason when the type is 0, so the same program runs in both forms.
 * A check prints ROOT_CHECK:<name>:PASS; a failed requirement prints ROOT_FAIL and exits 1.
 *
 * unmap (the default; ends with a fault):
 *   grant_reads_and_writes          a mapped page reads zero and keeps what the program writes
 *   libc_accepts_copyable_grant     malloc of a small and of a large block and a second mmap work
 *   unmap_through_mapping_reference munmap through the mapping's reference succeeds
 *   other_mapping_survives_unmap    the other mappings keep their contents and take writes
 *   second_unmap_refused            a second munmap of the same mapping is refused
 *   stale_reference_at_reused_address_refused (root grants)
 *                                   the program maps the same address again; munmap through the
 *                                   old reference is refused and the new mapping still works
 *   stale access                    the last line reads through an object derived in the
 *                                   unmapped mapping: it must fault with cause 25, so the program
 *                                   must not reach ROOT_STALE_READ.
 * refuse (exit 0; nothing faults):
 *   address_refused (root grants)   munmap with the mapping's address as an integer
 *   nested_object_refused (root grants)
 *                                   munmap through an object derived over the whole page, one
 *                                   level below the mapping; the object and the mapping still work
 *   partial_unmap_refused           munmap of either page of a two-page mapping; the whole mapping
 *                                   then unmaps through its reference
 *   stale_reference_refused (root grants)
 *                                   the program ends a mapping itself with CREVOKE: munmap
 *                                   through the stale reference is refused, twice, without a fault
 *   stale_reference_after_collection_refused (root grants)
 *                                   the same after enough lifetimes ended for the adapter to
 *                                   collect identities (CAPSTONE_VM_STATS collections)
 * narrow (exit 0, or 1 under the negative control):
 *   narrowed_reference_refused (root grants)
 *                                   munmap through a reference narrowed to the first page of a
 *                                   two-page mapping is refused, and both pages still work
 *   whole_reference_after_narrowed  the whole mapping then unmaps through its own reference
 * renew (ends with a fault):
 *   renewed_old_reference_refused (root grants)
 *                                   after CRENEW, munmap through the reference from before it
 *   renewed_mapping_unmaps          munmap through the renewed reference succeeds; the last line
 *                                   reads through an object derived in the renewed mapping and
 *                                   must fault with 25
 * mremap (ends with a fault):
 *   public_mremap_moves             the libc's mremap (map, copy, unmap) moves a page to two
 *   mremap_old_reference_refused    munmap through the old reference afterwards is refused
 *                                   (the old mapping is gone); the last line reads through the old
 *                                   reference and must fault with 25.
 *
 * Before each stale read the program maps the same address again (ROOT_REMAP same_address=1).
 * An access that the processor lets through then reads the new page and does not fault at all,
 * so a missing CREVOKE shows as a read, not as a page fault. */
#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <capstone/capability.h>
#include "vm.h"

#define PAGE 4096UL
#define LARGE (1UL << 20)
#define ROUNDS 12000UL
#ifndef MREMAP_MAYMOVE
#define MREMAP_MAYMOVE 1
#endif

static char *volatile stale;
static int root_grants;

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

/* The reason for nearly every skip: without root grants the launcher resolves the address. */
#define BY_ADDRESS "root_grants=0 names a mapping by its address, so this unmap would succeed"

static void skip(const char *check, const char *reason)
{
    printf("ROOT_SKIP:%s:%s\n", check, reason);
    fflush(stdout);
}

static unsigned long cursor(void *p)
{
    return __builtin_capstone_cap_get_cursor(p);
}

/* munmap refused the call, with EINVAL. */
static int refused(int rc)
{
    return rc == -1 && errno == EINVAL;
}

static void *untagged(unsigned long address)
{
    void *p;
    __asm__("mv %0, %1" : "=r"(p) : "r"(address));
    return p;
}

static char *map(unsigned long bytes)
{
    char *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(p != MAP_FAILED, "mmap");
    p[0] = 1;
    return p;
}

/* A grant as the adapter issues it, its type printed and remembered, as a copyable reference. */
static char *grant(void)
{
    sublet_cap slot;
    require(!cap_vm_acquire(&slot, PAGE, PAGE, PROT_READ | PROT_WRITE, 6, CAP_VM_HEAP), "grant");
    unsigned long type = sublet_type(&slot);
    printf("ROOT_GRANT_TYPE:%lu\n", type);
    fflush(stdout);
    root_grants = type == SUBLET_TYPE_NONLIN;
    return cap_vm_copyable(&slot);
}

/* Map the address of the mapping that just ended again, then read through the stale reference.
 * The read must fault with 25 and never print. */
static void stale_read(unsigned long address)
{
    char *again = map(PAGE);
    again[64] = 0x77;
    printf("ROOT_REMAP same_address=%d\n", cursor(again) == address);
    fflush(stdout);
    char seen = *(volatile char *)stale;
    printf("ROOT_STALE_READ value=0x%x\n", seen);
    fflush(stdout);
}

static int mode_unmap(void)
{
    char *page = grant();
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

    require(!munmap(page, PAGE), "munmap through the mapping's reference");
    pass("unmap_through_mapping_reference");
    require(small[47] == 0x11 && large[LARGE - 1] == 0x22 && other[PAGE - 1] == 0x44,
            "the other mappings keep their contents");
    other[0] = 0x45;
    large[0] = 0x23;
    require(other[0] == 0x45 && large[0] == 0x23, "the other mappings take a write");
    pass("other_mapping_survives_unmap");

    require(refused(munmap(page, PAGE)), "a second munmap of the same mapping");
    pass("second_unmap_refused");

    /* The launcher places the next mapping of this size at the address it unmapped last. */
    char *again = map(PAGE);
    printf("ROOT_REMAP same_address=%d\n", cursor(again) == address);
    fflush(stdout);
    again[64] = 0x77;
    if (root_grants) {
        require(refused(munmap(page, PAGE)), "munmap through the old reference at the reused address");
        again[65] = 0x78;
        require(again[64] == 0x77 && again[65] == 0x78, "the new mapping after the refusal");
        pass("stale_reference_at_reused_address_refused");
    } else {
        skip("stale_reference_at_reused_address_refused", BY_ADDRESS);
    }

    char seen = *(volatile char *)stale;
    printf("ROOT_STALE_READ value=0x%x\n", seen);
    return 0;
}

static int mode_refuse(void)
{
    char *keep = grant();
    keep[0] = 1;

    if (root_grants) {
        char *page = map(PAGE);
        require(refused(munmap(untagged(cursor(page)), PAGE)), "munmap with an integer address");
        page[1] = 2;
        require(page[0] == 1 && page[1] == 2, "the mapping after the refusal of an address");
        pass("address_refused");

        char *object = capstone_cap_derive(page, 0, PAGE);
        require(refused(munmap(object, PAGE)), "munmap through an object over the whole page");
        object[2] = 3;
        page[3] = 4;
        require(page[2] == 3 && object[3] == 4, "the object and the mapping after the refusal");
        pass("nested_object_refused");
    } else {
        skip("address_refused", BY_ADDRESS);
        skip("nested_object_refused", BY_ADDRESS);
    }

    char *two = map(2 * PAGE);
    two[PAGE] = 2;
    require(refused(munmap(two, PAGE)), "munmap of the first page of two");
    require(refused(munmap(two + PAGE, PAGE)), "munmap of the second page of two");
    require(two[0] == 1 && two[PAGE] == 2, "the mapping after the partial munmaps");
    require(!munmap(two, 2 * PAGE), "munmap of the whole mapping afterwards");
    pass("partial_unmap_refused");

    if (root_grants) {
        char *page = map(PAGE);
        capstone_cap_release(page);
        require(refused(munmap(page, PAGE)), "munmap through a reference the program ended");
        require(refused(munmap(page, PAGE)), "a second munmap through that reference");
        require(keep[0] == 1, "the other mapping after those refusals");
        pass("stale_reference_refused");

        page = map(PAGE);
        capstone_cap_release(page);
        for (unsigned long i = 0; i < ROUNDS; ++i)
            capstone_cap_release(capstone_cap_derive(keep, 64, 64));
        require(refused(munmap(page, PAGE)), "munmap through an ended reference after a collection");
        keep[1] = 2;
        require(keep[0] == 1 && keep[1] == 2, "the other mapping after the collection");
        pass("stale_reference_after_collection_refused");
    } else {
        skip("stale_reference_refused", BY_ADDRESS);
        skip("stale_reference_after_collection_refused", BY_ADDRESS);
    }
    return 0;
}

static int mode_narrow(void)
{
    (void)grant();
    if (!root_grants) {
        skip("narrowed_reference_refused", BY_ADDRESS);
        skip("whole_reference_after_narrowed", "follows the narrowed case");
        return 0;
    }

    char *two = map(2 * PAGE);
    unsigned long base = cursor(two);
    two[PAGE] = 2;
    char *narrow = __builtin_capstone_cap_shrink(two, base, base + PAGE);
    require(cursor(narrow) == base, "the narrowed reference starts at the mapping");
    require(refused(munmap(narrow, 2 * PAGE)), "munmap through a narrowed reference");
    two[PAGE + 1] = 3;
    require(two[0] == 1 && two[PAGE] == 2 && two[PAGE + 1] == 3, "the mapping after the refusal");
    pass("narrowed_reference_refused");

    require(!munmap(two, 2 * PAGE), "munmap through the whole reference afterwards");
    pass("whole_reference_after_narrowed");
    return 0;
}

static int mode_renew(void)
{
    (void)grant();
    char *page = map(PAGE);
    unsigned long address = cursor(page);
    char *renewed = capstone_cap_renew(page);

    if (root_grants) {
        require(refused(munmap(page, PAGE)), "munmap through the reference from before CRENEW");
        pass("renewed_old_reference_refused");
    } else {
        skip("renewed_old_reference_refused", BY_ADDRESS);
    }

    char *object = capstone_cap_derive(renewed, 64, 64);
    object[0] = 0x55;
    require(renewed[64] == 0x55, "the object lies in the renewed mapping");
    stale = object;
    require(!munmap(renewed, PAGE), "munmap through the renewed reference");
    pass("renewed_mapping_unmaps");
    stale_read(address);
    return 0;
}

static int mode_mremap(void)
{
    (void)grant();
    char *p = map(PAGE);
    unsigned long address = cursor(p);
    memset(p, 0x61, PAGE);

    char *q = mremap(p, PAGE, 2 * PAGE, MREMAP_MAYMOVE);
    require(q != MAP_FAILED, "mremap");
    require(q[0] == 0x61 && q[PAGE - 1] == 0x61, "the moved page keeps its contents");
    q[PAGE] = 0x62;
    q[2 * PAGE - 1] = 0x63;
    require(q[PAGE] == 0x62 && q[2 * PAGE - 1] == 0x63, "the second page of the new mapping");
    pass("public_mremap_moves");

    require(refused(munmap(p, PAGE)), "munmap through the old reference after mremap");
    pass("mremap_old_reference_refused");
    stale = p;
    stale_read(address);
    return 0;
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "unmap";

    if (!strcmp(mode, "unmap")) return mode_unmap();
    if (!strcmp(mode, "refuse")) return mode_refuse();
    if (!strcmp(mode, "narrow")) return mode_narrow();
    if (!strcmp(mode, "renew")) return mode_renew();
    if (!strcmp(mode, "mremap")) return mode_mremap();
    fail("unknown mode");
    return 1;
}
