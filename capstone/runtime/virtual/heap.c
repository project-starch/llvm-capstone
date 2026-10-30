/* mallocng runs in the application's Capstone context. This file supplies
 * ownership operations, not allocation policy. Only VM growth/release and
 * node pressure cross the existing service boundary. */
#include <errno.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <malloc_capstone.h>
#include <capstone/capability.h>
#include "vm.h"
#include "vm-abi.h"
#include <capstone/representable.h>

/* The two switches of the spatial-versus-temporal measurement (runtime/application CMake
 * options CAPSTONE_HEAP_LIFETIMES and CAPSTONE_HEAP_SCRUB_FREE, recorded in sdk.json). The
 * default, LIFETIMES=1 and SCRUB_FREE=0, is the profile. LIFETIMES=0: a public object, and a
 * nested group, is a SHRINK copy of the slot's authority over the same bytes instead of a
 * CDERIVE child, so free revokes nothing and only a mapping's retirement (munmap, mremap)
 * ends a stale capability; the lifetime per mapping (mapping_group) remains, so ensure_ids
 * stays. SCRUB_FREE=0: free leaves the payload's bytes and tags in place (calloc still
 * zeroes), so a reused slot's previous contents are the next owner's to read; a stale
 * capability among them still faults on use, because its lifetime ended. SCRUB_FREE=1 zeroes
 * the payload before the CREVOKE. */
#ifndef CAPSTONE_HEAP_LIFETIMES
#define CAPSTONE_HEAP_LIFETIMES 1
#endif
#ifndef CAPSTONE_HEAP_SCRUB_FREE
#define CAPSTONE_HEAP_SCRUB_FREE 0
#endif

#define INDEX_BUCKETS 4096U
static struct capstone_malloc_slot *index_heads[INDEX_BUCKETS];
static sublet_cap pending_mapping;
static unsigned heap_mutex;
enum { SLOT_INTERNAL, SLOT_PUBLIC, SLOT_NESTED };
static void lock_heap(void)
{
    while (__atomic_exchange_n(&heap_mutex, 1, __ATOMIC_ACQUIRE))
        if (__capstone_vm_wait()) __builtin_trap();
}
static void unlock_heap(void)
{ __atomic_store_n(&heap_mutex, 0, __ATOMIC_RELEASE); }
static unsigned bucket(size_t key)
{ return ((key >> 4) * 11400714819323198485UL) >> 52; }
static void unindex(struct capstone_malloc_slot *r)
{
    if (!r->key) return;
    struct capstone_malloc_slot **p = &index_heads[bucket(r->key)];
    while (*p && *p != r) p = &(*p)->next;
    if (!*p) __builtin_trap();
    *p = r->next;
    r->next = NULL; r->key = 0;
}
struct capstone_malloc_slot *__capstone_malloc_find(const void *p)
{
    size_t key = __builtin_capstone_cap_get_cursor((void *)p);
    for (struct capstone_malloc_slot *r = index_heads[bucket(key)]; r; r = r->next)
        if (r->key == key) return r;
    return NULL;
}
void __capstone_malloc_slot_key(struct capstone_malloc_slot *r, void *p, size_t n)
{
    unindex(r);
    r->key = __builtin_capstone_cap_get_cursor(p); r->nominal = n;
    unsigned b = bucket(r->key);
    r->next = index_heads[b]; index_heads[b] = r;
}
/* Q-21: THE APPLICATION RUNTIME NEVER READS A CSR FOR AVAILABILITY.
 *
 * This used to begin `csrr n, 0xcc0` (urevavail) and return early when n >= need +
 * CV_NODE_RESERVE, purely to SKIP the service call below. The read is a QEMU-only construct:
 * urevavail does not exist on the RTL (csr_regfile.sv falls through to illegal), so every image
 * carrying it passes under QEMU and takes an illegal-instruction fault on silicon at its FIRST
 * malloc. vp2's boot 3-T1 read exactly that -- cause 2 at contract.dom VA 0x20ccc, which is this
 * instruction inside malloc, with tval the instruction's own encoding. Q-21's rule had been applied
 * to the kernel module and not here, and the application runtime cannot see the table header at
 * all, so it had no second source to fall back on.
 *
 * Dropping the pre-check is BEHAVIOUR-PRESERVING, not a trade: the module's ensure_nodes() opens
 * with `if (available >= required) return 0;` over the table header, which is the authoritative
 * copy of the same quantity the CSR mirrored. So the decision is unchanged and merely made one
 * level in. The cost is a service call on the path that previously returned without one.
 *
 * CORRECTION (2026-10-26): that cost was PER ALLOCATION, not per mapping as this comment used to say.
 * malloc, realloc and aligned_alloc each called ensure_ids(256) unconditionally, so every one of them
 * left the context for a full delegated round (thousands of instructions; #282 profiles one round at
 * 6,890) in both measurement arms, T1 included, where a malloc takes no record at all.
 *
 * So the heap now counts instead of asking. A successful call leaves at least IDS_ASK records
 * available (the module's ensure_nodes guarantees available >= required); the heap then takes up to
 * IDS_ASK - CV_NODE_RESERVE - 1 of them, one per CDERIVE it issues (take_id), before it asks again,
 * so it never drives the table to the hardware's pressure threshold on its own. That is one service
 * call per ~255 records taken, and none for an allocation that takes none (T1's SHRINK). Records
 * taken outside the heap (a ported program's own nested allocator, CSNODE) are not counted here;
 * they are what the hardware's kind-5 pressure escape is for, which the module answers with
 * ensure_nodes and a retry (capstone_vm.c, CV_STEP, event 5) -- the backstop this hint never
 * replaced. Frees do not refill the count: a CREVOKE retires a record and frees it only at a
 * collection, so counting takes alone errs on the side of asking early.
 *
 * Availability is not a thing the application polls. The signals are the hardware's kind-5 pressure
 * escape and the module's collection; a hint, if one is ever wanted, is a service call like the one
 * below and never a CSR. */
#define IDS_ASK (CV_NODE_RESERVE + 256)
/* The most records one heap call takes before its next ensure_ids: a mapping group, a nested group
 * and the object's own child. */
#define IDS_PER_CALL 3
static size_t ids_budget;   /* records the heap may still take before asking; under the heap lock */
static int ensure_ids(size_t need)
{
    if (ids_budget >= need) return 0;
    long rc = __capstone_vm_nodes(IDS_ASK);
    if (rc < 0) { errno = -rc; return -1; }
    ids_budget = IDS_ASK - CV_NODE_RESERVE - 1;
    return 0;
}
static inline void take_id(void)
{
    if (ids_budget) --ids_budget;
}
/* Keep musl's group geometry. Slot parents are narrowed copies of one
 * managed group, not linear partitions or separate lifetime nodes. The
 * caller receives only a freshly derived child over its requested bytes. */
/* The group header bind_slots places slots after. Derived from the MAPPING size, so it is an
 * over-approximation of the header the stride itself needs -- granule() is monotone -- which makes
 * the single slot's stride below provably small enough rather than exactly right. Measured over
 * 300,003 sizes, the over-approximation costs 0 bytes. */
static size_t group_header(size_t bytes)
{
    size_t g = capstone_repr_granule(bytes > 4 ? bytes - 4 : 1);
    return g > 16 ? g : 16;
}
/* Cause B probe. A refused shrink leaves an unusable slot, mallocng then fails to find one, and
 * malloc returns NULL -- indistinguishable at the call site from any other out-of-memory, which is
 * what made the first run uninformative. Counted, never printed: a print from inside the allocator
 * can re-enter it. */
unsigned long __capstone_malloc_bind_refused;
unsigned long __capstone_malloc_bind_first[6];
unsigned long __capstone_malloc_ng_null;
/* Split from __capstone_malloc_ng_null, which conflated them and left the first run ambiguous. */
unsigned long __capstone_malloc_ids_failed;
/* The choke point. map_calls not advancing across a failed malloc means mallocng never asked for
 * a mapping, which no errno can tell you. */
unsigned long __capstone_malloc_map_calls;
unsigned long __capstone_malloc_map_refused_early;
unsigned long __capstone_malloc_map_refused_acquire;
unsigned long __capstone_malloc_map_last[4];   /* bytes, align, errno, rounded request */
/* The BASE, which is what add()'s second disjunct tests: (address & (bytes - 1)). Without it a
 * power-of-two size refused with EINVAL cannot be told from a misaligned one -- the reading that
 * refuted "the predicate is is_power_of_2". */
unsigned long __capstone_malloc_map_last_base;
unsigned long __capstone_malloc_map_ok_base;
/* The last ADMITTED mapping. The exact-bounds-ON arm cannot show a non-power-of-two mapping being
 * accepted while the gate is armed -- turning the profile on disables the gate -- so the only way
 * to see an admission is to record it on the success path. */
unsigned long __capstone_malloc_map_ok[2];     /* bytes, align */
unsigned long __capstone_malloc_map_admitted;
static void *bind_slots(struct capstone_malloc_slot *slots, void *group,
                        size_t stride, unsigned count)
{
    size_t base = __builtin_capstone_cap_get_cursor(group);
    /* Option (iii): the slot array starts after a GRANULE-sized group header, not a 16-byte one,
     * so each slot's base is a multiple of its own bounds granule and the capability derived over
     * an object in it is exactly representable. mallocng's enframe pads identically
     * (meta.h cap_hdr_pad), and the pad fits in the page rounding alloc_group already does --
     * asserted there, because the margin is exact for some class and count pairs. */
    size_t granule = capstone_repr_granule(stride > 4 ? stride - 4 : 1);
    size_t hdr = granule > 16 ? granule : 16;
    for (unsigned i = 0; i < count; ++i) {
        size_t start = base + hdr + i * stride;
        size_t end = base + hdr + (i + 1) * stride;
        /* Offset in integers first (lane runtime/cursor-window, 2026-10-19). `group + start - base`
         * let the compiler keep a capability with its cursor near 0, far outside its bounds, live
         * across this loop. On QEMU's compact profile a preemption saves it compressed and the resume
         * decodes different bounds (capstone_supervisor.c virtual_store_reg/virtual_load_reg); on
         * silicon a bounds-deriving op issued while it is out of window refuses with 29 (RTL arm
         * 4eecdf450). Forming `start - base` first keeps every intermediate cursor in the group. */
        slots[i].parent = __builtin_capstone_cap_shrink(
            (char *)group + (start - base), start, end);
        if (__builtin_capstone_cap_get_base(slots[i].parent) != start ||
            __builtin_capstone_cap_get_end(slots[i].parent) != end) {
            if (!__capstone_malloc_bind_refused++) {
                __capstone_malloc_bind_first[0] = i;
                __capstone_malloc_bind_first[1] = start;
                __capstone_malloc_bind_first[2] = end;
                __capstone_malloc_bind_first[3] = stride;
                __capstone_malloc_bind_first[4] = base;
                __capstone_malloc_bind_first[5] =
                    __builtin_capstone_cap_get_end(group);
            }
        }
        slots[i].issued = NULL;
    }
    return group;
}
/* A child of a slot's authority over [base + offset, base + offset + size): the CDERIVE
 * child with a lifetime of its own, or without lifetimes the narrowed copy that bind_slots
 * makes for the slot itself (the same rights, the slot's lifetime). */
static void *issue_child(void *parent, size_t offset, size_t size)
{
#if CAPSTONE_HEAP_LIFETIMES
    take_id();
    return capstone_cap_derive(parent, offset, size);
#else
    size_t start = __builtin_capstone_cap_get_base(parent) + offset;
    void *at = (char *)parent + (ptrdiff_t)(start - __builtin_capstone_cap_get_cursor(parent));
    /* CDERIVE refuses bounds the encoder cannot represent (cause 29); SHRINK rounds outward on
     * silicon. A silently widened child would reach into its neighbours where the qualified
     * arm faults, so the spatial arm would under-report the overflows it exists to measure. */
    void *child = __builtin_capstone_cap_shrink(at, start, start + size);
    if (__builtin_capstone_cap_get_base(child) != start ||
        __builtin_capstone_cap_get_end(child) != start + size)
        __builtin_trap();
    return child;
#endif
}
static void end_child(void *parent, void *child)
{
#if CAPSTONE_HEAP_LIFETIMES
    capstone_cap_revoke_child(parent, child);
#else
    (void)parent; (void)child;
#endif
}
static void *mapping_group(sublet_cap *grant, size_t bytes)
{
    take_id();
    return capstone_cap_derive(cap_vm_copyable(grant), 0, bytes);
}
void *__capstone_malloc_map(size_t bytes)
{
    __capstone_malloc_map_calls++;
    __capstone_malloc_map_last[3] = bytes;
    if (!bytes || bytes > CV_MAX_REGION_BYTES) {
        __capstone_malloc_map_refused_early++;
        errno = ENOMEM; return MAP_FAILED;
    }
    bytes = (bytes + 4095) & -4096UL;
    /* Option (iii): above 4 MiB the bounds granule EXCEEDS a page (8 KiB at 4 MiB, 128 KiB at
     * 64 MiB), so a page-aligned mapping is not enough -- the encoder truncates an unaligned base
     * downward and the object would gain authority below itself. Measured: 4, 8, 16, 64 and 88 MiB
     * all decode inexact at a page-aligned base, while 2 MiB and 4 MiB - 4 KiB decode exact.
     * cap_vm_acquire already takes an alignment, so this is the alignment argument rather than an
     * over-map and trim, and `bytes` is rounded to the granule so both edges are multiples. */
    bytes = capstone_repr_round(bytes);
    size_t align = capstone_repr_granule(bytes);
    if (align < 4096UL) align = 4096UL;
    if (cap_vm_acquire(&pending_mapping, bytes, align,
                       PROT_READ | PROT_WRITE, 6, CAP_VM_HEAP)) {
        __capstone_malloc_map_refused_acquire++;
        __capstone_malloc_map_last[0] = bytes;
        __capstone_malloc_map_last[1] = align;
        __capstone_malloc_map_last[2] = (unsigned long)errno;
        return MAP_FAILED;
    }
    __capstone_malloc_map_admitted++;
    __capstone_malloc_map_ok[0] = bytes;
    __capstone_malloc_map_ok[1] = align;
    __capstone_malloc_map_ok_base = sublet_base(&pending_mapping);
    /* Does the ADMITTED mapping satisfy add()'s second disjunct? If a mapping is admitted whose
     * base is NOT naturally aligned to its own size, the alignment half cannot be what refuses. */
    __capstone_malloc_map_last_base =
        (__capstone_malloc_map_ok_base & (bytes - 1)) ? 1UL : 0UL;
    /* A scalar placeholder crosses only the private backend, until attach.
     * The linear grant remains in its consuming slot throughout alloc_meta. */
    return (void *)sublet_base(&pending_mapping);
}
void *__capstone_malloc_group_attach(struct capstone_malloc_slot *slots,
                                    void *p, size_t bytes)
{
    size_t base = sublet_base(&pending_mapping);
    if (base != (size_t)p) __builtin_trap();
    /* stride = bytes - header, NOT bytes - 16. The single slot starts after the granule-sized
     * header, so a stride measured from a 16-byte one ran past the end of the mapping and the
     * shrink in bind_slots could not be derived -- which is what returned NULL for every
     * allocation on this path. */
    return bind_slots(slots, mapping_group(&pending_mapping, bytes),
                      bytes - group_header(bytes), 1);
}
void *__capstone_malloc_group_map(struct capstone_malloc_slot *slots,
                                 size_t bytes, size_t stride, unsigned count)
{
    void *p = __capstone_malloc_map(bytes);
    if (p == MAP_FAILED) return p;
    size_t mapped = sublet_end(&pending_mapping) - sublet_base(&pending_mapping);
    return bind_slots(slots, mapping_group(&pending_mapping, mapped), stride, count);
}
void *__capstone_malloc_group_nested(struct capstone_malloc_slot *slots,
                                    size_t stride, unsigned count,
                                    struct capstone_malloc_slot *parent)
{
    size_t base = parent->key;
    parent->issued = issue_child(parent->parent,
        base - __builtin_capstone_cap_get_base(parent->parent), 16 + stride * count);
    parent->kind = SLOT_NESTED;
    return bind_slots(slots, parent->issued, stride, count);
}
void *__capstone_malloc_slot_take(struct capstone_malloc_slot *r)
{
    void *p = r->parent;
    __capstone_malloc_slot_key(r, p, r->nominal);
    return p;
}
void *__capstone_malloc_slot_start(struct capstone_malloc_slot *r)
{
    return r->parent;
}
static void *backend_pointer(struct capstone_malloc_slot *r)
{ return (char *)r->parent + (r->key - __builtin_capstone_cap_get_base(r->parent)); }  /* offset first; see bind_slots */
void __capstone_malloc_slot_release(struct capstone_malloc_slot *r)
{ unindex(r); r->kind = SLOT_INTERNAL; }
void __capstone_malloc_nested_release(struct capstone_malloc_slot *r)
{
    end_child(r->parent, r->issued);
    r->issued = NULL;
    __capstone_malloc_slot_release(r);
}
void *__capstone_malloc_prepare_free(void *p)
{
    struct capstone_malloc_slot *r = __capstone_malloc_find(p);
    if (!r || r->kind == SLOT_NESTED) __builtin_trap();
    if (r->kind == SLOT_PUBLIC) {
#if CAPSTONE_HEAP_SCRUB_FREE
        /* Remove residual payload capabilities before returning ownership.
         * The slot prefix and musl's offset-cycle state are preserved. */
        memset(backend_pointer(r), 0, r->nominal);
#endif
        end_child(r->parent, r->issued);
        r->issued = NULL;
        r->kind = SLOT_INTERNAL;
    }
    return p;
}
/* A real Linux mremap is a VM slow path. The returned grant is new authority
 * over the retained physical frames; the old mapping's descendants are dead. */
/* Option (iii): how often a remap had to fall back to a copy because the service returned a
 * mapping that was page-aligned but not granule-aligned. __capstone_vm_remap takes an address and
 * two lengths and has NO alignment parameter, unlike cap_vm_acquire, so this cannot be asked for.
 * Read by malloc-contract.c and reported from the suite runs: mremap returns page-aligned
 * addresses, so at a 128 KiB granule most huge reallocs are expected to take the copy, and that
 * cost should be a measured number rather than an inference. If the acceptance workloads hit it
 * often, an aligned remap (reserve an aligned range, then MREMAP_FIXED into it, inside the module)
 * is the follow-up, decided on this count. */
unsigned long __capstone_malloc_remap_unaligned;

/* Is a remapped mapping usable, i.e. will every bound derived inside it be representable? */
int __capstone_malloc_remap_usable(unsigned long base, size_t bytes)
{
    /* Check the length ACTUALLY mapped, not a rounded-up one: repr_round(bytes) can be
     * representable while `bytes` itself is not, which would pass a mapping whose real end the
     * encoder cannot express. */
    return bytes && capstone_repr_ok(base, bytes);
}

void *__capstone_malloc_group_remap(struct capstone_malloc_slot *slots,
                                   void *p, size_t old, size_t bytes)
{
    struct capstone_malloc_slot *r = slots;
    size_t offset = r->key - __builtin_capstone_cap_get_base(r->parent);
    sublet_cap root;
    sublet_store(&root, __capstone_vm_remap((size_t)p, old, bytes));
    unsigned long result;
    __asm__ volatile("ld %0, 0(%1)" : "=r"(result) : "r"(&root) : "memory");
    if ((long)result < 0) { errno = -(long)result; sublet_clear(&root); return MAP_FAILED; }
    /* CORRECTION (2026-10-17): this was written as a safe fallback -- "costs a copy and never a
     * wrong bound" -- and it is not one. __capstone_vm_remap() above has ALREADY moved the
     * mapping and re-registered it by the time we get here, so returning MAP_FAILED sends
     * realloc.c down a malloc-plus-copy path that reads from an address which no longer maps.
     * The launcher's remap_mapping() now guarantees a granule-aligned destination, which makes
     * this an ASSERTION rather than a fallback: a nonzero counter is a launcher bug to fix, not
     * a cost to measure. It is kept, and kept counted, precisely so that bug would be visible
     * rather than silent. */
    if (!__capstone_malloc_remap_usable(sublet_base(&root), bytes)) {
        __capstone_malloc_remap_unaligned++;
        sublet_clear(&root);
        errno = ENOMEM;
        return MAP_FAILED;
    }
    unindex(r);
    /* bytes - group_header(bytes), NOT bytes - 16, for the same reason as group_attach: the slot
     * begins after a granule-sized header, so a stride measured from a 16-byte one runs past the
     * end of the mapping. Missing it here while fixing the attach path cost a SIGSEGV that only
     * appeared once growth began completing at all -- the overrun is 496 bytes on a 401408-byte
     * mapping, and get_stride() (which realloc.c now uses) disagrees with it. */
    void *header = bind_slots(slots, mapping_group(&root, bytes),
                              bytes - group_header(bytes), 1);
    void *raw = (char *)r->parent + offset;
    __capstone_malloc_slot_key(r, raw, r->nominal);
    *(unsigned long *)header = (unsigned long)r->group;
    ((unsigned char *)header)[8] = 0;
    return header;
}
static void *publish(void *p, size_t n)
{
    if (!p) return NULL;
    struct capstone_malloc_slot *r = __capstone_malloc_find(p);
    if (!r) __builtin_trap();
    /* Successful in-place realloc replaces the public lifetime; failure
     * never reaches here. Remapping already retired the old mapping. */
    if (r->issued) end_child(r->parent, r->issued);
    r->kind = SLOT_PUBLIC; r->nominal = n;
    r->issued = issue_child(r->parent,
        r->key - __builtin_capstone_cap_get_base(r->parent), n ? n : 1);
    return r->issued;
}
void *malloc(size_t n)
{
    /* Option (iii): publish() derives the child over the nominal size, so that size has to be
     * representable -- a granule multiple, at a granule-aligned base. Rounding HERE rather than
     * inside mallocng keeps musl's own decision code untouched (check-mallocng-policy.py holds
     * size_to_class and the slot/class selection token-identical to upstream), and it means
     * mallocng's set_size records exactly what the capability covers, so malloc_usable_size and
     * an in-place realloc never report bytes the program cannot reach. The seven raised stride
     * entries are what make the resulting slot's base granule-aligned. */
    size_t want = capstone_repr_round(n ? n : 1);
    if (want < n) { errno = ENOMEM; return NULL; }   /* overflow on a hostile size */
    lock_heap();
    void *raw;
    if (ensure_ids(IDS_PER_CALL)) { __capstone_malloc_ids_failed++; raw = NULL; }
    else { raw = __capstone_mallocng_malloc(want); if (!raw) __capstone_malloc_ng_null++; }
    void *p = publish(raw, want);
    unlock_heap(); return p;
}
void *__libc_malloc(size_t n) { return malloc(n); }
void *__simple_malloc(size_t n) { return malloc(n); }
__attribute__((noinline, noreturn)) static void invalid_free(void)
{
    __asm__ volatile(".global cap_malloc_invalid_free\ncap_malloc_invalid_free:\nunimp" ::: "memory");
    __builtin_unreachable();
}
__attribute__((noinline)) static struct capstone_malloc_slot *checked(void *p)
{
    __asm__ volatile(".global cap_malloc_validate\ncap_malloc_validate:\nlbu zero, 0(%0)"
                     : : "r"(p) : "memory");
    struct capstone_malloc_slot *r = __capstone_malloc_find(p);
    if (!r || r->kind != SLOT_PUBLIC) invalid_free();
    return r;
}
void free(void *p)
{
    if (!p) return;
    int e = errno;
    lock_heap(); __capstone_mallocng_free(backend_pointer(checked(p))); unlock_heap();
    errno = e;
}
void __libc_free(void *p) { free(p); }

/* THE NESTED-ALLOCATOR TRIO: what an allocator layered on malloc needs at its own alloc and free,
 * and nothing else. An application allocator that hands out sub-objects of a malloc'd block (PHP's
 * emalloc: a 48-byte header, then the payload) must at ITS free find the block's own authority
 * from the caller's pointer and keep a live reference to the block after the caller's lifetime
 * has ended. Both are in this heap's slot record already -- the issued object with MANAGE, indexed
 * by address -- so every such allocator was rebuilding this index for itself (ports/php's seam
 * carried 886 lines of it). These calls are the index, exported.
 *
 * derive: a child of the issued object `object`, over [object + offset, + size), on this heap's
 * node budget (the same ensure_ids/take_id path malloc's own child takes, so a nested allocator
 * cannot exhaust the table the heap reserved against). NULL when the table cannot be grown, never
 * a fault: the caller decides what an unprotected object means in its arm.
 *
 * object: the issued object that holds `child`, cursor at its start, or NULL when no slot record
 * has that block. Nothing ends. The block is found by the child's BASE minus `offset`, the same
 * offset derive was given, so a child narrowed from below (its base moved) is not found.
 *
 * reclaim: object, and the caller's child ends. A missing block is refused like an interior free
 * (cap_malloc_invalid_free), after one diagnostic line. CREVOKE's direct-parent rule is the
 * identity check: a grandchild the client derived for itself is cause 29, a child already ended is
 * cause 25, both at cap_malloc_reclaim. The parent is the record's issued object, never the
 * caller's reference, so the authority that ends the child is the heap's. Lookup, validation and
 * the CREVOKE run under the heap lock, so a concurrent free or realloc of the same object cannot
 * end the parent between the two. */
static struct capstone_malloc_slot *slot_of_child(void *child, size_t offset)
{
    size_t key = __builtin_capstone_cap_get_base(child) - offset;
    for (struct capstone_malloc_slot *q = index_heads[bucket(key)]; q; q = q->next)
        if (q->key == key) return (q->kind == SLOT_PUBLIC && q->issued) ? q : NULL;
    return NULL;
}
void *__capstone_malloc_derive(void *object, size_t offset, size_t size)
{
    if (!object || !size) return NULL;
    lock_heap();
    void *child = NULL;
    if (ensure_ids(1) == 0) { take_id(); child = capstone_cap_derive(object, offset, size); }
    unlock_heap();
    return child;
}
void *__capstone_malloc_object(void *child, size_t offset)
{
    if (!child) return NULL;
    lock_heap();
    struct capstone_malloc_slot *r = slot_of_child(child, offset);
    void *object = r ? r->issued : NULL;
    unlock_heap();
    return object;
}
__attribute__((noinline)) static void reclaim_child(void *parent, void *child)
{
    __asm__ volatile(".global cap_malloc_reclaim\ncap_malloc_reclaim:" ::: "memory");
    capstone_cap_revoke_child(parent, child);
}
void *__capstone_malloc_reclaim(void *child, size_t offset)
{
    if (!child) return NULL;
    lock_heap();
    struct capstone_malloc_slot *r = slot_of_child(child, offset);
    void *object = r ? r->issued : NULL;
    if (object) reclaim_child(object, child);
    unlock_heap();
    if (!object) {
        /* One line before the refusal, so a run that dies here says which pointer: a free of a
         * pointer this heap did not issue reads otherwise exactly like a port defect. */
        char line[160];
        int n = snprintf(line, sizeof line,
                         "CAPSTONE_RECLAIM_MISS base=0x%lx cursor=0x%lx end=0x%lx offset=%lu tag=%d\n",
                         (unsigned long)__builtin_capstone_cap_get_base(child),
                         (unsigned long)__builtin_capstone_cap_get_cursor(child),
                         (unsigned long)__builtin_capstone_cap_get_end(child), (unsigned long)offset,
                         (int)__builtin_capstone_cap_get_tag(child));
        if (n > 0) write(2, line, (size_t)n < sizeof line ? (size_t)n : sizeof line - 1);
        invalid_free();
    }
    return object;
}
void *calloc(size_t n, size_t size)
{
    if (size && n > SIZE_MAX / size) { errno = ENOMEM; return NULL; }
    void *p = malloc(n * size);
    if (p) memset(p, 0, n * size);
    return p;
}
/* mallocng keeps an object's offset when it grows it in place, or through the remap of a
 * single-slot mapping (the no-op one included). enframe aligned that offset only to the granule of
 * the size the object was allocated at, and the size published here can have a larger one, so the
 * child would not be exactly representable: the spatial heap's exactness check traps (cause 2)
 * and CDERIVE refuses it (cause 29). Measured on the twin, both arms: 32720 -> 32768 in place from
 * a base = 32 mod 64, and 261888 -> 262144 through the no-op remap from a base = 256 mod 512,
 * faulted on every misaligned round and passed on every aligned one; SPEC's gcc train (200.c,
 * scilab.c) and perlbench test (japh/abigail, re/pat_rt_report) stop on the same instructions.
 * Such a block is moved here, malloc-copy-free as mallocng's own moving path does, to a slot
 * enframed at the new size. Done in the runtime and not in realloc.c because
 * check-mallocng-policy.py holds mallocng's in-place decision token-identical to upstream, and
 * the decision is not wrong: only the published bounds are. The copy reads the old block through
 * its slot, which spans it whichever route kept it. If the new block cannot be had, NULL: the
 * caller keeps the old object, which a remap has already moved and whose old child is dead. */
static void *exact_or_moved(void *raw, size_t want, size_t old)
{
    if (!raw) return NULL;
    struct capstone_malloc_slot *r = __capstone_malloc_find(raw);
    if (!r) __builtin_trap();
    if (capstone_repr_ok(r->key, want)) return raw;
    void *fresh = __capstone_mallocng_malloc(want);
    if (!fresh) { __capstone_malloc_ng_null++; return NULL; }
    memcpy(fresh, raw, old < want ? old : want);
    __capstone_mallocng_free(raw);
    return fresh;
}
void *realloc(void *p, size_t n)
{
    if (!p) return malloc(n);
    lock_heap();
    struct capstone_malloc_slot *r = checked(p);
    /* Round the REQUEST as malloc does, not just the published size: publish() derives the child
     * over the nominal size, so that size must be representable, and asking mallocng for the
     * rounded size is what makes the block actually that big. Publishing a rounded size over an
     * unrounded block would hand out a capability past the allocation. */
    size_t want = capstone_repr_round(n ? n : 1);
    if (want < n) { unlock_heap(); errno = ENOMEM; return NULL; }
    size_t old = r->nominal;
    void *q = ensure_ids(IDS_PER_CALL) ? NULL
            : publish(exact_or_moved(__capstone_mallocng_realloc(backend_pointer(r), want), want, old), want);
    unlock_heap(); return q;
}
void *__libc_realloc(void *p, size_t n) { return realloc(p, n); }
void *aligned_alloc(size_t a, size_t n)
{
    lock_heap();
    /* Same as malloc and realloc. This one is why posix_memalign(4096, 5001) faulted: 5001 has
     * granule 8 and 5001 % 8 == 1, so the child derived over the raw size is unrepresentable and
     * the CDERIVE in capstone_cap_derive raises ILLEGAL_OP_VAL. The arena itself was fine; the
     * rounding added to malloc had simply never been applied to its two siblings. */
    size_t want = capstone_repr_round(n ? n : 1);
    if (want < n) { unlock_heap(); errno = ENOMEM; return NULL; }
    void *p = ensure_ids(IDS_PER_CALL) ? NULL
            : publish(__capstone_mallocng_aligned_alloc(a, want), want);
    unlock_heap(); return p;
}
int posix_memalign(void **out, size_t a, size_t n)
{
    if (!a || (a & (a-1)) || a < sizeof(void *)) return EINVAL;
    int e = errno;
    void *p = aligned_alloc(a, n);
    int error = p ? 0 : errno;
    errno = e;
    if (!error) *out = p;
    return error;
}
void *memalign(size_t a, size_t n) { return aligned_alloc(a, n); }
size_t malloc_usable_size(void *p)
{
    if (!p) return 0;
    lock_heap(); size_t n = checked(p)->nominal; unlock_heap(); return n;
}
int malloc_trim(size_t pad) { (void)pad; return 0; }
