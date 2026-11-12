/* Root grants (module parameter root_grants=1): each mapping is a reservation, a record M that the
 * kernel derives from the one reference over the address space and keeps to itself, and the
 * program's reference P is a child of M. Every request of the program to end a mapping comes with
 * P: munmap, the stack at thread exit, the frame's hand-over at thread creation (stages 2 to 4 of
 * docs/plans/2026-11-12-address-space-root-adapter.md). The kernel accepts the reference
 * only if it is a live child of M whose bounds cover the mapping as mmap exposed it, and the call
 * names that whole length; then it ends the mapping with CREVOKE of M. An mprotect comes with P
 * too and may name part of the mapping, which P must cover. run-root-grants.py runs this program
 * once per mode (argv[1]) in one boot. Build with the virtual SDK.
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
 *                                   after CRENEW, munmap through the reference from before it is
 *                                   refused, and the renewed mapping still works
 *   renewed_mapping_unmaps          munmap through the renewed reference succeeds; the last line
 *                                   reads through an object derived in the renewed mapping and
 *                                   must fault with 25
 * renew-ref (ends with a fault): as renew, and the last line reads through the renewed reference
 *                                   itself.
 * mremap (ends with a fault): the libc's mremap (map, copy, unmap) of a two-page mapping that
 *                                   holds data and, at offset 64, a reference to another mapping
 *   mremap_wrong_length_refused     an old length that is not the mapping's: the unmap of the old
 *                                   mapping is refused, so mremap fails with EINVAL, and the old
 *                                   mapping keeps its contents and takes writes
 *   mremap_nested_object_refused (root grants)
 *                                   the same through an object over the whole mapping
 *   public_mremap_grows             mremap to three pages moves the data and the stored reference,
 *                                   which still reads its object; the third page reads zero
 *   mremap_old_reference_refused    munmap through the old reference afterwards is refused
 *                                   (the old mapping is gone)
 *   public_mremap_shrinks           mremap of the grown mapping to one page keeps the data of that
 *                                   page and the stored reference; the last line reads through the
 *                                   grown mapping's reference after the program mapped its address
 *                                   again, and must fault with 25.
 * exit-victim (exit 0; root grants):
 *   exit_with_own_stack_at_victim_refused
 *                                   a thread exits naming a victim mapping with its own stack's
 *                                   reference moved to the victim's address, which is the stack's
 *                                   end (the stack lies right below the victim), so the operand is
 *                                   a live reference to another mapping; the victim still works
 *                                   afterwards
 *   exit_with_victim_address_refused
 *                                   the same with the victim's address as an integer
 *   victim_unmaps_afterwards        the victim then unmaps through its own reference
 * exit-renew (ends with a fault):
 *   stack_renewed_before_exit       a thread renews its own stack's mapping with CRENEW and exits
 *                                   naming the renewed reference; the main thread kept a copy.
 *                                   The last line reads through that copy after the main thread
 *                                   mapped the stack's address again: it must fault with 25.
 * create (ends with a fault):
 *   create_with_stale_frame_refused (root grants)
 *                                   the creator renews the frame's mapping and passes the
 *                                   reference from before: the create is refused, and the renewed
 *                                   reference still works
 *   create_with_renewed_frame       the create through the renewed reference succeeds; the last
 *                                   line reads the frame through it and must fault with 25.
 * create-fail (exit 0):
 *   failed_create_releases_frame    the launcher's thread table is full (31 contexts exited and not
 *                                   joined), so the launcher fails the create after the module
 *                                   took the frame: the create answers EAGAIN and the frame's
 *                                   mapping is no longer registered (registered())
 * detached (exit 0):
 *   detached_exit_unmaps_stack      a detached pthread exits through musl's __unmapself: its stack
 *                                   mapping is registered before the exit and gone after it.
 * exit-length (exit 0):
 *   exit_with_wrong_length_completes
 *                                   a thread exits naming its own stack's mapping with a length
 *                                   that is not the mapping's: the joiner returns and the launcher
 *                                   wrote the exit word
 *   wrong_length_stack_stays_mapped the stack's mapping is still registered and works through the
 *                                   reference the main thread kept
 *   wrong_length_stack_unmaps_afterwards
 *                                   munmap through that reference then releases it
 * exit-huge (exit 0): the same with the length 2^64 - 1 (exit_with_huge_length_completes,
 *                                   huge_length_stack_stays_mapped,
 *                                   huge_length_stack_unmaps_afterwards). The exit word lies
 *                                   outside the stack, below the named base plus that length.
 * exit-frame (exit 0):
 *   exit_with_frame_as_stack_completes
 *                                   a thread exits naming its own frame's mapping as its stack:
 *                                   the joiner returns and the launcher wrote the exit word
 *   frame_as_stack_released_once    the frame's mapping is no longer registered, and the thread's
 *                                   stack still works and unmaps
 * protect (exit 0, or 1 under the negative control):
 *   protect_through_mapping_reference
 *                                   mprotect through the mapping's reference changes the
 *                                   protection of the whole mapping and of one page of it, as
 *                                   /proc/self/maps shows
 *   protect_integer_address_refused (root grants)
 *                                   mprotect with the mapping's address as an integer is refused
 *                                   and the protection is unchanged
 *   protect_stale_reference_refused (root grants)
 *                                   the same through the reference from before CRENEW
 *   protect_narrowed_reference_refused (root grants)
 *                                   the same through a reference narrowed to the first page of a
 *                                   two-page mapping, for both pages
 *   protect_nested_object_refused (root grants)
 *                                   the same through an object derived over the whole page
 *   protect_narrowed_covering_range a narrowed reference changes the page it covers: an mprotect
 *                                   may name part of a mapping (musl opens a stack past its guard)
 * protect-held (exit 0, or 1 under the negative control):
 *   protect_held_mapping_refused    a pthread's mprotect of its own exchange region, which the
 *                                   launcher holds while the thread is attached, is refused
 *                                   with EBUSY
 *   protect_after_detach            after the thread detaches its transport, the same mprotect
 *                                   succeeds
 * create-gate (exit 0, or 1 under the negative control):
 *   create_metadata_frame_refused   a frame in a one-page mapping of the allocator's metadata is
 *                                   refused (EINVAL), and the page still works
 *   create_launcher_page_refused    the launcher's own stack mapping, named at its base, is refused
 *   create_mismatched_frame_refused a frame named with its reference moved inside the page is
 *                                   refused, and the frame still works. The launcher refuses it;
 *                                   a launcher that names the mapping around the cursor
 *                                   (run-root-grants.py --launcher-frame-floor) leaves it to the
 *                                   module, which compares the frame with a0's cursor
 *   create_after_refusals           the create through the frame's own reference then succeeds
 *
 * registered() asks the launcher whether an address lies in a registered mapping, by integer: a
 * futex wake of no waiter there answers 0, outside every mapping EFAULT.
 *
 * Before each stale read the program maps the same address again (ROOT_REMAP same_address=1).
 * An access that the processor lets through then reads the new page and does not fault at all,
 * so a missing CREVOKE shows as a read, not as a page fault. */
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <capstone/capability.h>
#include <capstone/virtual.h>
#include "vm.h"

#define PAGE 4096UL
#define LARGE (1UL << 20)
#define ROUNDS 12000UL
#define STACK (64UL * 1024)
/* The launcher's thread table holds 32 contexts, the first one included. */
#define IDLE_THREADS 31

extern long __capstone_vm_thread_create_frame(void *frame);
extern long __capstone_vm_futex(void *word, int op, int value, long seconds, long nanoseconds,
                                int bitset);
extern _Noreturn void __capstone_vm_thread_exit_clear(uintptr_t clear, void *stack, size_t bytes,
                                                      long status);
extern uintptr_t __capstone_context_identity_take(void);
extern void __capstone_context_identity_give(uintptr_t identity);
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

/* Does address lie in a mapping the launcher has registered? */
static int registered(unsigned long address)
{
    return __capstone_vm_futex(untagged(address), 1 /* FUTEX_WAKE */, 0, -1, 0, 0) != -EFAULT;
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

/* Map the address of the mapping of `bytes` that just ended again, then read through the stale
 * reference. The read must fault with 25 and never print. */
static void stale_read_bytes(unsigned long address, unsigned long bytes)
{
    char *again = map(bytes);
    again[64] = 0x77;
    printf("ROOT_REMAP same_address=%d\n", cursor(again) == address);
    fflush(stdout);
    char seen = *(volatile char *)stale;
    printf("ROOT_STALE_READ value=0x%x\n", seen);
    fflush(stdout);
}

static void stale_read(unsigned long address)
{
    stale_read_bytes(address, PAGE);
}

static void pause_ms(long ms)
{
    struct timespec t = {ms / 1000, (ms % 1000) * 1000000};
    nanosleep(&t, NULL);
}

static void *thread_pointer(void)
{
    void *tp;
    __asm__ volatile("movc %0, tp" : "=r"(tp));
    return tp;
}

/* The type LCC answers for a reference in a register: 1 non-linear, 7 for an integer. */
static unsigned long type_of(void *p)
{
    unsigned long type;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x04, %0, %1, x1" : "=r"(type) : "r"(p));
    return type;
}

/* p with its cursor set to address (SCC). Outside what the bounds can represent the result is
 * an integer. */
static void *at_address(void *p, unsigned long address)
{
    void *q;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x05, %0, %1, %2" : "=r"(q) : "r"(p), "r"(address));
    return q;
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

/* through_object: the last read goes through an object derived in the renewed mapping, else
 * through the renewed reference itself. */
static int mode_renew(int through_object)
{
    (void)grant();
    char *page = map(PAGE);
    unsigned long address = cursor(page);
    char *renewed = capstone_cap_renew(page);

    if (root_grants) {
        require(refused(munmap(page, PAGE)), "munmap through the reference from before CRENEW");
        renewed[1] = 0x56;
        require(renewed[0] == 1 && renewed[1] == 0x56, "the renewed mapping after the refusal");
        pass("renewed_old_reference_refused");
    } else {
        skip("renewed_old_reference_refused", BY_ADDRESS);
    }

    char *object = capstone_cap_derive(renewed, 64, 64);
    object[0] = 0x55;
    require(renewed[64] == 0x55, "the object lies in the renewed mapping");
    stale = through_object ? object : renewed;
    require(!munmap(renewed, PAGE), "munmap through the renewed reference");
    pass("renewed_mapping_unmaps");
    stale_read(address);
    return 0;
}

/* mremap: the libc's mremap maps a new reservation, copies the contents with their tags and
 * unmaps the old reservation through the program's reference (mapping.c). The stored reference
 * at offset 64 names another mapping. */
static int mode_mremap(void)
{
    (void)grant();
    char *target = map(PAGE);
    target[1] = 0x71;
    char *p = map(2 * PAGE);
    unsigned long address = cursor(p);
    memset(p, 0x61, 2 * PAGE);
    *(char **)(p + 64) = target;

    errno = 0;
    require(mremap(p, PAGE, 3 * PAGE, MREMAP_MAYMOVE) == MAP_FAILED && errno == EINVAL,
            "mremap with an old length that is not the mapping's");
    p[2] = 0x62;
    require(p[0] == 0x61 && p[2] == 0x62 && p[2 * PAGE - 1] == 0x61 &&
            (*(char **)(p + 64))[1] == 0x71, "the old mapping after the refused mremap");
    pass("mremap_wrong_length_refused");

    if (root_grants) {
        char *object = capstone_cap_derive(p, 0, 2 * PAGE);
        errno = 0;
        require(mremap(object, 2 * PAGE, 3 * PAGE, MREMAP_MAYMOVE) == MAP_FAILED && errno == EINVAL,
                "mremap through an object over the whole mapping");
        object[3] = 0x63;
        require(p[3] == 0x63 && p[2] == 0x62 && p[2 * PAGE - 1] == 0x61,
                "the mapping and the object after the refused mremap");
        pass("mremap_nested_object_refused");
    } else {
        skip("mremap_nested_object_refused", BY_ADDRESS);
    }

    char *q = mremap(p, 2 * PAGE, 3 * PAGE, MREMAP_MAYMOVE);
    require(q != MAP_FAILED && cursor(q) != address, "mremap to three pages moves the mapping");
    char *moved = *(char **)(q + 64);
    printf("ROOT_MREMAP grown moved_type=%lu\n", type_of(moved));
    fflush(stdout);
    require(q[0] == 0x61 && q[2] == 0x62 && q[2 * PAGE - 1] == 0x61,
            "the grown mapping keeps the data");
    require(type_of(moved) == 1 && cursor(moved) == cursor(target) && moved[1] == 0x71,
            "the grown mapping keeps the stored reference");
    require(q[2 * PAGE] == 0 && q[3 * PAGE - 1] == 0, "the third page reads zero");
    q[3 * PAGE - 1] = 0x64;
    require(q[3 * PAGE - 1] == 0x64, "the third page takes a write");
    pass("public_mremap_grows");

    require(refused(munmap(p, 2 * PAGE)), "munmap through the old reference after mremap");
    pass("mremap_old_reference_refused");

    unsigned long grown = cursor(q);
    char *r = mremap(q, 3 * PAGE, PAGE, MREMAP_MAYMOVE);
    require(r != MAP_FAILED && cursor(r) != grown, "mremap to one page moves the mapping");
    moved = *(char **)(r + 64);
    require(r[0] == 0x61 && r[2] == 0x62 && r[PAGE - 1] == 0x61,
            "the shrunk mapping keeps the data of its page");
    require(type_of(moved) == 1 && moved[1] == 0x71, "the shrunk mapping keeps the stored reference");
    pass("public_mremap_shrinks");

    stale = q;
    stale_read_bytes(grown, 3 * PAGE);
    return 0;
}

/* ---- Threads ----
 *
 * The thread cases use raw contexts (__capstone_virtual_context_create, as the pthread contract
 * does) that share the main thread's TLS and touch none of it, so each case controls exactly
 * what its exit service names. */

/* exit-victim: the exit service's stack operand and its length, chosen by the main thread. */
static void *volatile exit_operand;
static volatile unsigned long exit_bytes;

static void *exit_naming_worker(void *arg)
{
    (void)arg;
    __capstone_vm_thread_exit_clear(0, exit_operand, exit_bytes, 0);
}

/* One raw context on its own stack mapping (own, own_bytes) that exits naming exit_operand;
 * joined. */
static void exit_naming(void *operand, unsigned long bytes, char *own, unsigned long own_bytes)
{
    exit_operand = operand;
    exit_bytes = bytes;
    long id = __capstone_virtual_context_create(exit_naming_worker, NULL, own + own_bytes,
                                                thread_pointer());
    require(id > 0, "create the exiting context");
    require(!capstone_virtual_thread_join(id), "join the exiting context");
}

static int mode_exit_victim(void)
{
    (void)grant();
    if (!root_grants) {
        skip("exit_with_own_stack_at_victim_refused", BY_ADDRESS);
        skip("exit_with_victim_address_refused", BY_ADDRESS);
        skip("victim_unmaps_afterwards", "follows the refused cases");
        return 0;
    }

    /* Two heap grants in a row: the launcher places each one right below the one before where
     * that range is free, so the thread's stack ends where the victim begins, and the stack's
     * reference with its cursor at the victim's address stays a reference (a cursor at the end
     * is inside what the bounds represent). A pair that did not land so is left mapped and the
     * next pair tried. */
    char *victim = NULL, *own = NULL;
    unsigned long address = 0;
    unsigned tries = 0;
    for (; tries < 8; ++tries) {
        sublet_cap slot;
        victim = grant();
        address = cursor(victim);
        require(!cap_vm_acquire(&slot, 2 * PAGE, PAGE, PROT_READ | PROT_WRITE, 6, CAP_VM_HEAP),
                "the thread's stack");
        own = cap_vm_copyable(&slot);
        if (cursor(own) + 2 * PAGE == address) break;
    }
    victim[0] = 1;
    void *moved = at_address(own, address);
    printf("ROOT_OPERAND own_stack_at_victim type=%lu adjacent=%d tries=%u\n", type_of(moved),
           cursor(own) + 2 * PAGE == address, tries + 1);
    fflush(stdout);
    exit_naming(moved, PAGE, own, 2 * PAGE);
    victim[1] = 2;
    require(victim[0] == 1 && victim[1] == 2, "the victim after an exit with the own stack's reference");
    pass("exit_with_own_stack_at_victim_refused");

    char *other = map(STACK);
    exit_naming(untagged(address), PAGE, other, STACK);
    victim[2] = 3;
    require(victim[0] == 1 && victim[2] == 3, "the victim after an exit with its address");
    pass("exit_with_victim_address_refused");

    require(!munmap(victim, PAGE), "munmap of the victim through its own reference");
    pass("victim_unmaps_afterwards");
    return 0;
}

/* exit-renew: the worker renews the reference to its own stack's mapping and exits naming the
 * renewed reference, of which the main thread keeps a copy. From the CRENEW on the worker's sp
 * names the old lifetime and is stale, so everything after it is one asm block that touches no
 * stack. */
static void *volatile exit_stack;
static void *volatile kept;

static void *renew_exit_worker(void *arg)
{
    (void)arg;
    __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%[from])\n"       /* LDC t0, the stack's mapping */
                     ".insn r 0x5b, 0x1, 0x54, a1, t0, x0\n"     /* CRENEW a1, t0 */
                     ".insn s 0x5b, 0x4, a1, 0(%[keep])\n"       /* STC a1, the main thread's copy */
                     "li a0, 0\n"
                     "mv a2, %[bytes]\n"
                     "li a3, 0\n"
                     "li a7, %[service]\n"
                     "ecall\n"
                     "ebreak\n"
                     :
                     : [from] "r"(&exit_stack), [keep] "r"(&kept), [bytes] "r"(STACK),
                       [service] "i"(CV_SERVICE_PTHREAD_EXIT)
                     : "t0", "a0", "a1", "a2", "a3", "a7", "memory");
    __builtin_unreachable();
}

static int mode_exit_renew(void)
{
    (void)grant();
    char *stack = map(STACK);
    unsigned long address = cursor(stack);

    exit_stack = stack;
    long id = __capstone_virtual_context_create(renew_exit_worker, NULL, stack + STACK,
                                                thread_pointer());
    require(id > 0, "create the worker");
    require(!capstone_virtual_thread_join(id), "join the worker");
    char *renewed = kept;
    require(type_of(renewed) == 1 && cursor(renewed) == address,
            "the main thread holds the worker's renewed stack reference");
    pass("stack_renewed_before_exit");
    stale = renewed;
    stale_read_bytes(address, STACK);
    return 0;
}

/* create and create-fail: a start frame as thread.c builds it, on its own one-page mapping. */
static volatile int child_ran;

static void *frame_child(void *arg)
{
    __atomic_store_n(&child_ran, 1, __ATOMIC_RELEASE);
    return arg;
}

static void *frame_return(void *unused)
{
    (void)unused;
    capstone_virtual_thread_exit(NULL);
}

static unsigned char *frame_fill(unsigned char *frame, void *(*entry)(void *), void *stack_top,
                                 uintptr_t *identity);

static unsigned char *frame_new(void *(*entry)(void *), void *stack_top, uintptr_t *identity)
{
    sublet_cap slot;
    require(!cap_vm_acquire(&slot, PAGE, PAGE, PROT_READ | PROT_WRITE, 6, CAP_VM_HEAP),
            "the frame's grant");
    return frame_fill(cap_vm_copyable(&slot), entry, stack_top, identity);
}

/* The page frame as a start frame for entry on stack_top, with a new context identity. */
static unsigned char *frame_fill(unsigned char *frame, void *(*entry)(void *), void *stack_top,
                                 uintptr_t *identity)
{
    memset(frame, 0, PAGE);
    ((uint64_t *)frame)[7] = 3;
    *identity = __capstone_context_identity_take();
    require(*identity != 0, "an identity for the context");
    capstone_cap_store((capstone_cap_slot *)(frame + 64 + 1 * 16), (void *)frame_return);
    capstone_cap_store((capstone_cap_slot *)(frame + 64 + 2 * 16), stack_top);
    __asm__ volatile("scc t0, gp, %1\n\tstc t0, 0(%0)"
                     : : "r"(frame + 64 + 3 * 16), "r"(*identity) : "t0", "memory");
    capstone_cap_store((capstone_cap_slot *)(frame + 64 + 4 * 16), thread_pointer());
    capstone_cap_store((capstone_cap_slot *)(frame + 64 + 10 * 16), NULL);
    capstone_cap_store((capstone_cap_slot *)(frame + 64), (void *)entry);
    return frame;
}

static int mode_create(void)
{
    (void)grant();
    char *stack = map(STACK);
    uintptr_t identity;
    unsigned char *frame = frame_new(frame_child, stack + STACK, &identity);
    unsigned char *renewed = capstone_cap_renew(frame);

    if (root_grants) {
        long id = __capstone_vm_thread_create_frame(frame);
        printf("ROOT_CREATE stale=%ld\n", id);
        fflush(stdout);
        require(id == -EINVAL, "thread create with the frame's reference from before CRENEW");
        renewed[PAGE - 1] = 0x5a;
        require(renewed[PAGE - 1] == 0x5a && ((uint64_t *)renewed)[7] == 3,
                "the renewed frame after the refused create");
        pass("create_with_stale_frame_refused");
    } else {
        skip("create_with_stale_frame_refused", BY_ADDRESS);
    }

    long id = __capstone_vm_thread_create_frame(renewed);
    printf("ROOT_CREATE renewed=%ld\n", id);
    fflush(stdout);
    require(id > 0, "thread create with the renewed frame reference");
    pass("create_with_renewed_frame");
    stale = (char *)renewed;
    char seen = *(volatile char *)stale;
    printf("ROOT_STALE_READ value=0x%x\n", seen);
    fflush(stdout);
    capstone_virtual_thread_join(id);
    return 0;
}

static volatile int idle_done;

static void *idle_entry(void *arg)
{
    __atomic_fetch_add(&idle_done, 1, __ATOMIC_RELEASE);
    return arg;
}

static int mode_create_fail(void)
{
    (void)grant();
    char *stacks = map((IDLE_THREADS + 1) * STACK);
    long ids[IDLE_THREADS];

    /* Fill the launcher's thread table: contexts that exit and are not joined keep their entries
     * there, while the module has let them go. */
    for (unsigned i = 0; i < IDLE_THREADS; ++i) {
        ids[i] = __capstone_virtual_context_create(idle_entry, NULL, stacks + (i + 1) * STACK,
                                                   thread_pointer());
        require(ids[i] > 0, "create an idle context");
    }
    for (unsigned spins = 0; __atomic_load_n(&idle_done, __ATOMIC_ACQUIRE) < IDLE_THREADS; ++spins) {
        require(spins < 1000, "the idle contexts finish");
        pause_ms(10);
    }
    pause_ms(200);

    uintptr_t identity;
    unsigned char *frame = frame_new(frame_child, stacks + (IDLE_THREADS + 1) * STACK, &identity);
    unsigned long address = cursor(frame);
    long id;
    for (unsigned tries = 0;; ++tries) {
        id = __capstone_vm_thread_create_frame(frame);
        if ((id == -EBUSY || id == -ENOSPC) && tries < 100) { pause_ms(20); continue; }
        break;
    }
    printf("ROOT_CREATE_FAIL result=%ld\n", id);
    fflush(stdout);
    require(id == -EAGAIN, "the launcher fails the create after the module took the frame");
    __capstone_context_identity_give(identity);
    int probe = registered(address);
    printf("ROOT_FRAME_PROBE registered=%d\n", probe);
    fflush(stdout);
    require(!probe, "the frame's mapping is no longer registered");
    pass("failed_create_releases_frame");

    for (unsigned i = 0; i < IDLE_THREADS; ++i)
        require(!capstone_virtual_thread_join(ids[i]), "join an idle context");
    return 0;
}

/* detached: a pthread's stack mapping is registered while it runs and gone after its exit, which
 * unmaps it through musl's __unmapself (pthread.c) with the mapping's reference. */
static volatile unsigned long detached_page;
static volatile int detached_ready, detached_go;

static void *detached_worker(void *arg)
{
    pthread_attr_t attr;
    void *base = NULL;
    size_t bytes = 0;
    if (!pthread_getattr_np(pthread_self(), &attr) && !pthread_attr_getstack(&attr, &base, &bytes))
        detached_page = (cursor(base) + bytes / 2) & ~(PAGE - 1);
    __atomic_store_n(&detached_ready, 1, __ATOMIC_RELEASE);
    while (!__atomic_load_n(&detached_go, __ATOMIC_ACQUIRE)) pause_ms(5);
    return arg;
}

static int stack_probe(void)
{
    return registered(detached_page) ? 0 : EINVAL;
}

static int mode_detached(void)
{
    (void)grant();
    pthread_attr_t attr;
    pthread_t thread;
    require(!pthread_attr_init(&attr) &&
            !pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED), "detached attribute");
    require(!pthread_create(&thread, &attr, detached_worker, NULL), "pthread_create");
    for (unsigned spins = 0; !__atomic_load_n(&detached_ready, __ATOMIC_ACQUIRE); ++spins) {
        require(spins < 1000, "the detached thread starts");
        pause_ms(10);
    }
    require(detached_page, "the detached thread's stack");
    int before = stack_probe();
    __atomic_store_n(&detached_go, 1, __ATOMIC_RELEASE);
    int after = 0;
    for (unsigned spins = 0; spins < 200 && !after; ++spins) {
        pause_ms(20);
        after = stack_probe();
    }
    printf("ROOT_DETACHED page=0x%lx before=%d after=%d\n", detached_page, before, after);
    fflush(stdout);
    require(before == 0, "the stack's mapping is registered while the thread runs");
    require(after == EINVAL, "the stack's mapping is gone after the thread's exit");
    pass("detached_exit_unmaps_stack");
    return 0;
}

/* exit-length and exit-huge: a thread exits naming its own stack's mapping with a length that is
 * not the mapping's, STACK - PAGE or 2^64 - 1. The exit completes and writes the exit word; the
 * mapping stays until the program unmaps it through the reference it kept. The exit word lies
 * outside the stack, so a launcher that checks it against the base and the length the program
 * names (and not against the mapping it found) stops at the huge length. */
static volatile int exit_word;

static void *exit_word_worker(void *arg)
{
    (void)arg;
    __capstone_vm_thread_exit_clear(cursor((void *)&exit_word), exit_operand, exit_bytes, 0);
}

static int mode_exit_length(unsigned long bytes, const char *tag)
{
    char name[80];
    (void)grant();
    char *stack = map(STACK);
    unsigned long address = cursor(stack);

    printf("ROOT_EXIT_WORD in_stack=%d length=0x%lx\n",
           cursor((void *)&exit_word) - address < STACK, bytes);
    fflush(stdout);
    require(cursor((void *)&exit_word) - address >= STACK, "the exit word lies outside the stack");
    exit_operand = stack;
    exit_bytes = bytes;
    exit_word = 1;
    long id = __capstone_virtual_context_create(exit_word_worker, NULL, stack + STACK,
                                                thread_pointer());
    require(id > 0, "create the exiting context");
    require(!capstone_virtual_thread_join(id), "join the context that named a wrong length");
    require(exit_word == 0, "the launcher wrote the exit word");
    snprintf(name, sizeof name, "exit_with_%s_length_completes", tag);
    pass(name);

    require(registered(address), "the stack's mapping is still registered");
    stack[0] = 0x31;
    stack[STACK - 1] = 0x32;
    require(stack[0] == 0x31 && stack[STACK - 1] == 0x32,
            "the stack's mapping works through the kept reference");
    snprintf(name, sizeof name, "%s_length_stack_stays_mapped", tag);
    pass(name);

    require(!munmap(stack, STACK), "munmap of the stack through the kept reference");
    require(!registered(address), "the stack's mapping is gone after the munmap");
    snprintf(name, sizeof name, "%s_length_stack_unmaps_afterwards", tag);
    pass(name);
    return 0;
}

/* exit-frame: a thread exits naming its own frame's mapping as its stack, through the frame's
 * reference from before the create (stale since the hand-over). The module refuses the stack
 * because it is the thread's frame and releases the arena once, as the frame. */
static void *volatile frame_operand;

static void *frame_as_stack_child(void *arg)
{
    (void)arg;
    __capstone_vm_thread_exit_clear(cursor((void *)&exit_word), frame_operand, PAGE, 0);
}

static int mode_exit_frame(void)
{
    (void)grant();
    char *stack = map(STACK);
    uintptr_t identity;
    unsigned char *frame = frame_new(frame_as_stack_child, stack + STACK, &identity);
    unsigned long address = cursor(frame);

    frame_operand = frame;
    exit_word = 1;
    long id = __capstone_vm_thread_create_frame(frame);
    require(id > 0, "create the thread on its frame");
    require(!capstone_virtual_thread_join(id), "join the thread that named its frame as its stack");
    require(exit_word == 0, "the launcher wrote the exit word");
    pass("exit_with_frame_as_stack_completes");

    int probe = registered(address);
    printf("ROOT_FRAME_PROBE registered=%d\n", probe);
    fflush(stdout);
    require(!probe, "the frame's mapping is released");
    stack[0] = 0x33;
    require(stack[0] == 0x33 && !munmap(stack, STACK), "the thread's stack afterwards");
    pass("frame_as_stack_released_once");
    return 0;
}

/* The protection of the page at address as /proc/self/maps shows it ("rw-", "r--", "---"), or
 * "?" where no line holds it. The launcher's process is the program's address space. */
static void protection(unsigned long address, char out[4])
{
    char line[256];
    FILE *maps = fopen("/proc/self/maps", "r");

    strcpy(out, "?");
    if (!maps)
        return;
    while (fgets(line, sizeof line, maps)) {
        unsigned long low, high;
        char rights[8];
        if (sscanf(line, "%lx-%lx %7s", &low, &high, rights) == 3 && low <= address &&
            address < high) {
            memcpy(out, rights, 3);
            out[3] = 0;
            break;
        }
    }
    fclose(maps);
}

/* Is the page at address protected as rights says? Printed for the record. */
static int protected_as(unsigned long address, const char *rights)
{
    char seen[4];
    protection(address, seen);
    printf("ROOT_PROTECTION 0x%lx %s\n", address, seen);
    fflush(stdout);
    return !strcmp(seen, rights);
}

static int mode_protect(void)
{
    (void)grant();
    char *two = map(2 * PAGE);
    unsigned long base = cursor(two);
    two[PAGE] = 2;

    require(!mprotect(two, 2 * PAGE, PROT_READ), "mprotect through the mapping's reference");
    require(protected_as(base, "r--") && protected_as(base + PAGE, "r--"),
            "the whole mapping is read-only");
    require(two[0] == 1 && two[PAGE] == 2, "the read-only mapping reads");
    require(!mprotect(two, 2 * PAGE, PROT_READ | PROT_WRITE), "restore write rights");
    require(!mprotect(two + PAGE, PAGE, PROT_NONE), "mprotect of the second page");
    require(protected_as(base, "rw-") && protected_as(base + PAGE, "---"),
            "only the second page is inaccessible");
    require(!mprotect(two + PAGE, PAGE, PROT_READ | PROT_WRITE), "restore the second page");
    two[1] = 3;
    two[PAGE + 1] = 4;
    require(two[1] == 3 && two[PAGE + 1] == 4, "the mapping after the restores");
    pass("protect_through_mapping_reference");

    if (root_grants) {
        char *page = map(PAGE);
        unsigned long address = cursor(page);
        require(refused(mprotect(untagged(address), PAGE, PROT_NONE)),
                "mprotect with an integer address");
        require(protected_as(address, "rw-"), "the protection after the refusal of an address");
        page[1] = 5;
        require(page[0] == 1 && page[1] == 5, "the mapping after the refusal of an address");
        pass("protect_integer_address_refused");

        char *renewed = capstone_cap_renew(page);
        require(refused(mprotect(page, PAGE, PROT_NONE)),
                "mprotect through the reference from before CRENEW");
        require(protected_as(address, "rw-"), "the protection after the refusal of a stale reference");
        renewed[2] = 6;
        require(renewed[1] == 5 && renewed[2] == 6, "the renewed mapping after the refusal");
        pass("protect_stale_reference_refused");

        char *narrow = __builtin_capstone_cap_shrink(two, base, base + PAGE);
        require(refused(mprotect(narrow, 2 * PAGE, PROT_NONE)),
                "mprotect of two pages through a reference narrowed to one");
        require(protected_as(base, "rw-") && protected_as(base + PAGE, "rw-"),
                "the protection after the refusal of a narrowed reference");
        pass("protect_narrowed_reference_refused");

        char *object = capstone_cap_derive(renewed, 0, PAGE);
        require(refused(mprotect(object, PAGE, PROT_NONE)),
                "mprotect through an object over the whole page");
        require(protected_as(address, "rw-"), "the protection after the refusal of an object");
        object[3] = 7;
        require(renewed[3] == 7, "the object and the mapping after the refusal");
        pass("protect_nested_object_refused");
    } else {
        skip("protect_integer_address_refused", BY_ADDRESS);
        skip("protect_stale_reference_refused", BY_ADDRESS);
        skip("protect_narrowed_reference_refused", BY_ADDRESS);
        skip("protect_nested_object_refused", BY_ADDRESS);
    }

    char *first = __builtin_capstone_cap_shrink(two, base, base + PAGE);
    require(!mprotect(first, PAGE, PROT_READ), "mprotect of the page a narrowed reference covers");
    require(protected_as(base, "r--") && protected_as(base + PAGE, "rw-"),
            "only the covered page is read-only");
    require(!mprotect(two, PAGE, PROT_READ | PROT_WRITE), "restore the first page");
    pass("protect_narrowed_covering_range");
    return 0;
}

/* protect-held: a pthread's exchange region is a mapping of its own that the launcher holds
 * while the thread is attached, since the launcher writes it through Linux. mprotect of it through
 * the thread's own reference is refused with EBUSY. After the thread detaches its transport, the
 * same mprotect succeeds. The worker restores each protection it changed before its next
 * delegated call writes the region again. */
extern void *__capstone_virtual_exchange(void);
extern long __capstone_vm_transport_detach(void);
static int held_errno = -1, detached_errno = -1;

static void *held_worker(void *arg)
{
    printf("ROOT_HELD_WORKER attached\n");      /* a delegated call: the worker attaches */
    fflush(stdout);
    char *exchange = __capstone_virtual_exchange();
    if (!exchange)
        return arg;
    held_errno = mprotect(exchange, PAGE, PROT_READ) ? errno : 0;
    if (!held_errno)
        mprotect(exchange, PAGE, PROT_READ | PROT_WRITE);
    __capstone_vm_transport_detach();
    detached_errno = mprotect(exchange, PAGE, PROT_READ) ? errno : 0;
    if (!detached_errno)
        mprotect(exchange, PAGE, PROT_READ | PROT_WRITE);
    return arg;
}

static int mode_protect_held(void)
{
    (void)grant();
    pthread_t thread;
    require(!pthread_create(&thread, NULL, held_worker, NULL), "pthread_create");
    require(!pthread_join(thread, NULL), "pthread_join");
    printf("ROOT_HELD held_errno=%d detached_errno=%d\n", held_errno, detached_errno);
    fflush(stdout);
    require(held_errno == EBUSY, "mprotect of a held mapping");
    pass("protect_held_mapping_refused");
    require(detached_errno == 0, "mprotect of the same mapping after the detach");
    pass("protect_after_detach");
    return 0;
}

/* The base of a reference's bounds (LCC selector 3). */
static unsigned long base_of(void *p)
{
    unsigned long base;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x04, %0, %1, x3" : "=r"(base) : "r"(p));
    return base;
}

static void *stack_pointer(void)
{
    void *sp;
    __asm__ volatile("movc %0, sp" : "=r"(sp));
    return sp;
}

/* create-gate: the launcher hands the module only a frame that the program may end, a whole
 * one-page heap mapping named at its address, and the module takes only the frame that a0
 * names. */
static int mode_create_gate(void)
{
    (void)grant();
    char *stack = map(STACK);
    uintptr_t identity;
    sublet_cap slot;

    require(!cap_vm_acquire(&slot, PAGE, PAGE, PROT_READ | PROT_WRITE, 6, CAP_VM_METADATA),
            "a page of the allocator's metadata");
    unsigned char *metadata = frame_fill(cap_vm_copyable(&slot), frame_child, stack + STACK,
                                         &identity);
    long id = __capstone_vm_thread_create_frame(metadata);
    printf("ROOT_CREATE metadata=%ld\n", id);
    fflush(stdout);
    require(id == -EINVAL, "thread create with a frame in a metadata page");
    __capstone_context_identity_give(identity);
    metadata[PAGE - 1] = 0x5b;
    require(metadata[PAGE - 1] == 0x5b && ((uint64_t *)metadata)[7] == 3,
            "the metadata page after the refused create");
    pass("create_metadata_frame_refused");

    void *sp = stack_pointer();
    void *launcher_stack = at_address(sp, base_of(sp));
    id = __capstone_vm_thread_create_frame(launcher_stack);
    printf("ROOT_CREATE launcher_stack=%ld base=0x%lx type=%lu\n", id, base_of(sp),
           type_of(launcher_stack));
    fflush(stdout);
    require(id == -EINVAL, "thread create with the launcher's stack mapping");
    pass("create_launcher_page_refused");

    unsigned char *frame = frame_new(frame_child, stack + STACK, &identity);
    void *inside = at_address(frame, cursor(frame) + 64);
    id = __capstone_vm_thread_create_frame(inside);
    printf("ROOT_CREATE inside=%ld\n", id);
    fflush(stdout);
    require(id == -EINVAL,
            "thread create with a frame reference whose cursor is not the frame's address");
    frame[PAGE - 1] = 0x5c;
    require(frame[PAGE - 1] == 0x5c && ((uint64_t *)frame)[7] == 3,
            "the frame after the refused create");
    pass("create_mismatched_frame_refused");

    id = __capstone_vm_thread_create_frame(frame);
    printf("ROOT_CREATE own=%ld\n", id);
    fflush(stdout);
    require(id > 0, "thread create through the frame's own reference afterwards");
    require(!capstone_virtual_thread_join(id) && __atomic_load_n(&child_ran, __ATOMIC_ACQUIRE),
            "the created thread ran");
    pass("create_after_refusals");
    return 0;
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "unmap";

    if (!strcmp(mode, "unmap")) return mode_unmap();
    if (!strcmp(mode, "refuse")) return mode_refuse();
    if (!strcmp(mode, "narrow")) return mode_narrow();
    if (!strcmp(mode, "renew")) return mode_renew(1);
    if (!strcmp(mode, "renew-ref")) return mode_renew(0);
    if (!strcmp(mode, "mremap")) return mode_mremap();
    if (!strcmp(mode, "exit-victim")) return mode_exit_victim();
    if (!strcmp(mode, "exit-renew")) return mode_exit_renew();
    if (!strcmp(mode, "create")) return mode_create();
    if (!strcmp(mode, "create-fail")) return mode_create_fail();
    if (!strcmp(mode, "detached")) return mode_detached();
    if (!strcmp(mode, "exit-length")) return mode_exit_length(STACK - PAGE, "wrong");
    if (!strcmp(mode, "exit-huge")) return mode_exit_length(~0UL, "huge");
    if (!strcmp(mode, "exit-frame")) return mode_exit_frame();
    if (!strcmp(mode, "protect-held")) return mode_protect_held();
    if (!strcmp(mode, "protect")) return mode_protect();
    if (!strcmp(mode, "create-gate")) return mode_create_gate();
    fail("unknown mode");
    return 1;
}
