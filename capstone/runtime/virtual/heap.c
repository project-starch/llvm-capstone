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
static int ensure_ids(size_t need)
{
    unsigned long n;
    __asm__ volatile("csrr %0, 0xcc0" : "=r"(n));
    if (n >= need + CV_NODE_RESERVE) return 0;
    long rc = __capstone_vm_nodes(need);
    if (rc < 0) { errno = -rc; return -1; }
    return 0;
}
/* Keep musl's group geometry. Slot parents are narrowed copies of one
 * managed group, not linear partitions or separate lifetime nodes. The
 * caller receives only a freshly derived child over its requested bytes. */
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
    if (!bytes || bytes > CAP_VM_MAX_BYTES) { errno = ENOMEM; return MAP_FAILED; }
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
                       PROT_READ | PROT_WRITE, 6, CAP_VM_HEAP)) return MAP_FAILED;
    /* A scalar placeholder crosses only the private backend, until attach.
     * The linear grant remains in its consuming slot throughout alloc_meta. */
    return (void *)sublet_base(&pending_mapping);
}
void *__capstone_malloc_group_attach(struct capstone_malloc_slot *slots,
                                    void *p, size_t bytes)
{
    size_t base = sublet_base(&pending_mapping);
    if (base != (size_t)p) __builtin_trap();
    return bind_slots(slots, mapping_group(&pending_mapping, bytes), bytes - 16, 1);
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
    return capstone_repr_ok(base, capstone_repr_round(bytes ? bytes : 1));
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
    /* Refuse an unrepresentable mapping rather than derive a widened bound inside it. realloc.c
     * falls through to its malloc-plus-copy path on a failed mremap, so this costs a copy and
     * never a wrong bound. */
    if (!__capstone_malloc_remap_usable(sublet_base(&root), bytes)) {
        __capstone_malloc_remap_unaligned++;
        sublet_clear(&root);
        errno = ENOMEM;
        return MAP_FAILED;
    }
    unindex(r);
    void *header = bind_slots(slots, mapping_group(&root, bytes), bytes - 16, 1);
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
    void *p = ensure_ids(256) ? NULL : publish(__capstone_mallocng_malloc(want), want);
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
    void *q = ensure_ids(256) ? NULL : publish(__capstone_mallocng_realloc(backend_pointer(r), n), n);
    unlock_heap(); return q;
}
void *__libc_realloc(void *p, size_t n) { return realloc(p, n); }
void *aligned_alloc(size_t a, size_t n)
{
    lock_heap();
    void *p = ensure_ids(256) ? NULL : publish(__capstone_mallocng_aligned_alloc(a, n), n);
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
