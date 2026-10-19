/* mallocng runs in the application's Capstone context. This file supplies
 * ownership operations, not allocation policy. Only VM growth/release and
 * node pressure cross the existing service boundary. */
#include <errno.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <malloc_capstone.h>
#include <capstone/capability.h>
#include "vm.h"
#include <capstone/representable.h>

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
 * level in. The cost is a service call on the path that previously returned without one; it is
 * per-MAPPING (three call sites, all ensure_ids(256), on mallocng's group-mapping paths) and not
 * per-allocation, and it is measured rather than assumed in the record beside this change.
 *
 * Availability is not a thing the application polls. The signals are the hardware's kind-5 pressure
 * escape and the module's collection; a hint, if one is ever wanted, is a service call like the one
 * below and never a CSR. */
static int ensure_ids(size_t need)
{
    long rc = __capstone_vm_nodes(need);
    if (rc < 0) { errno = -rc; return -1; }
    return 0;
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
        slots[i].parent = __builtin_capstone_cap_shrink(
            (char *)group + start - base, start, end);
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
static void *mapping_group(sublet_cap *grant, size_t bytes)
{
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
    parent->issued = capstone_cap_derive(parent->parent,
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
{ return (char *)r->parent + r->key - __builtin_capstone_cap_get_base(r->parent); }
void __capstone_malloc_slot_release(struct capstone_malloc_slot *r)
{ unindex(r); r->kind = SLOT_INTERNAL; }
void __capstone_malloc_nested_release(struct capstone_malloc_slot *r)
{
    capstone_cap_revoke_child(r->parent, r->issued);
    r->issued = NULL;
    __capstone_malloc_slot_release(r);
}
void *__capstone_malloc_prepare_free(void *p)
{
    struct capstone_malloc_slot *r = __capstone_malloc_find(p);
    if (!r || r->kind == SLOT_NESTED) __builtin_trap();
    if (r->kind == SLOT_PUBLIC) {
        /* Remove residual payload capabilities before returning ownership.
         * The slot prefix and musl's offset-cycle state are preserved. */
        memset(backend_pointer(r), 0, r->nominal);
        capstone_cap_revoke_child(r->parent, r->issued);
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
    if (r->issued) capstone_cap_revoke_child(r->parent, r->issued);
    r->kind = SLOT_PUBLIC; r->nominal = n;
    r->issued = capstone_cap_derive(r->parent,
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
    if (ensure_ids(256)) { __capstone_malloc_ids_failed++; raw = NULL; }
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
void *calloc(size_t n, size_t size)
{
    if (size && n > SIZE_MAX / size) { errno = ENOMEM; return NULL; }
    void *p = malloc(n * size);
    if (p) memset(p, 0, n * size);
    return p;
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
    void *q = ensure_ids(256) ? NULL
            : publish(__capstone_mallocng_realloc(backend_pointer(r), want), want);
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
    void *p = ensure_ids(256) ? NULL
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
