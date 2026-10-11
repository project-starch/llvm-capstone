/* mallocng runs in the application's Capstone context. This file supplies
 * ownership operations, not allocation policy. Only VM growth/release and
 * node pressure cross the existing service boundary. */
#include <errno.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <malloc_capstone.h>
#include "vm.h"

#define INDEX_BUCKETS 4096U
static struct capstone_malloc_slot *index_heads[INDEX_BUCKETS];
static sublet_cap pending_mapping;
static unsigned heap_mutex;
static unsigned long allocations, frees, live_objects, peak_objects;
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
    r->raw = p; r->key = __builtin_capstone_cap_get_cursor(p); r->nominal = n;
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
static sublet_cap *authority(struct capstone_malloc_slot *r)
{ return (sublet_cap *)&r->authority; }

/* Partition the exact upstream group geometry: a 16-byte header followed
 * by count slots of stride bytes. The initial four prefix bytes of each
 * slot are represented in its record, never in the preceding owner's slot. */
static void *partition(struct capstone_malloc_slot *slots, void **tail,
                       sublet_cap *root, size_t base, size_t stride, unsigned count)
{
    sublet_cap rest;
    sublet_split(root, base + 16, &rest);
    void *header = cap_vm_copyable(root);
    for (unsigned i = 0; i < count; ++i) {
        size_t end = base + 16 + (i + 1) * stride;
        if (end < sublet_end(&rest)) {
            sublet_cap next;
            sublet_split(&rest, end, &next);
            sublet_move(&rest, authority(&slots[i]));
            sublet_move(&next, &rest);
        } else sublet_move(&rest, authority(&slots[i]));
        if (sublet_base(authority(&slots[i])) != end - stride ||
            sublet_end(authority(&slots[i])) != end) __builtin_trap();
    }
    sublet_move(&rest, (sublet_cap *)tail);
    return header;
}
void *__capstone_malloc_map(size_t bytes)
{
    if (!bytes || bytes > CAP_VM_MAX_BYTES) { errno = ENOMEM; return MAP_FAILED; }
    bytes = (bytes + 4095) & -4096UL;
    if (cap_vm_acquire(&pending_mapping, bytes, 4096,
                       PROT_READ | PROT_WRITE, 6, CAP_VM_HEAP)) return MAP_FAILED;
    /* A scalar placeholder crosses only the private backend, until attach.
     * The linear grant remains in its consuming slot throughout alloc_meta. */
    return (void *)sublet_base(&pending_mapping);
}
void *__capstone_malloc_group_attach(struct capstone_malloc_slot *slots, void **tail,
                                    void *p, size_t bytes)
{
    size_t base = sublet_base(&pending_mapping);
    if (base != (size_t)p) __builtin_trap();
    return partition(slots, tail, &pending_mapping, base, bytes - 16, 1);
}
void *__capstone_malloc_group_map(struct capstone_malloc_slot *slots, void **tail,
                                 size_t bytes, size_t stride, unsigned count)
{
    void *p = __capstone_malloc_map(bytes);
    if (p == MAP_FAILED) return p;
    return partition(slots, tail, &pending_mapping, (size_t)p, stride, count);
}
void *__capstone_malloc_group_nested(struct capstone_malloc_slot *slots, void **tail,
                                    size_t stride, unsigned count,
                                    struct capstone_malloc_slot *parent)
{
    size_t base = parent->key;
    sublet_give(authority(parent));
    sublet_cap root;
    sublet_take_linear(authority(parent), &root);
    parent->kind = 3; parent->raw = NULL;
    return partition(slots, tail, &root, base, stride, count);
}
void *__capstone_malloc_slot_take(struct capstone_malloc_slot *r)
{
    if (sublet_type(authority(r)) != SUBLET_TYPE_LIN) sublet_give(authority(r));
    void *p = sublet_take(authority(r));
    __capstone_malloc_slot_key(r, p, r->nominal);
    return p;
}
void *__capstone_malloc_slot_start(struct capstone_malloc_slot *r)
{
    return (char *)r->raw - (r->key - __builtin_capstone_cap_get_base(r->raw));
}
void __capstone_malloc_slot_release(struct capstone_malloc_slot *r)
{ unindex(r); r->kind = 0; r->raw = NULL; }
void __capstone_malloc_nested_release(struct capstone_malloc_slot *r)
{
    sublet_give(authority(r));
    __capstone_malloc_slot_release(r);
}
static void *rotate(struct capstone_malloc_slot *r)
{
    size_t offset = r->key - sublet_base(authority(r));
    sublet_give(authority(r));
    void *p = (char *)sublet_take(authority(r)) + offset;
    r->raw = p;
    return p;
}
void *__capstone_malloc_prepare_free(void *p)
{
    struct capstone_malloc_slot *r = __capstone_malloc_find(p);
    if (!r || r->kind == 3) __builtin_trap();
    if (r->kind == 1) {
        /* Remove residual payload capabilities before returning ownership.
         * The slot prefix and musl's offset-cycle state are preserved. */
        memset(r->raw, 0, r->nominal);
        p = rotate(r);
        ++frees; --live_objects; r->kind = 0;
    }
    return p;
}
/* A real Linux mremap is a VM slow path. The returned grant is new authority
 * over the retained physical frames; the old mapping's descendants are dead. */
void *__capstone_malloc_group_remap(struct capstone_malloc_slot *slots, void **tail,
                                   void *p, size_t old, size_t bytes)
{
    struct capstone_malloc_slot *r = slots;
    size_t offset = r->key - __builtin_capstone_cap_get_base(r->raw);
    sublet_cap root;
    sublet_store(&root, __capstone_vm_remap((size_t)p, old, bytes));
    unsigned long result;
    __asm__ volatile("ld %0, 0(%1)" : "=r"(result) : "r"(&root) : "memory");
    if ((long)result < 0) { errno = -(long)result; sublet_clear(&root); return MAP_FAILED; }
    unindex(r);
    void *header = partition(slots, tail, &root, result, bytes - 16, 1);
    void *raw = (char *)sublet_take(authority(r)) + offset;
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
    if (r->kind == 1) p = rotate(r);
    else {
        ++allocations;
        if (++live_objects > peak_objects) peak_objects = live_objects;
    }
    r->kind = 1; r->nominal = n;
    return __builtin_capstone_cap_shrink(p, r->key, r->key + (n ? n : 1));
}
void *malloc(size_t n)
{
    lock_heap();
    void *p = ensure_ids(256) ? NULL : publish(__capstone_mallocng_malloc(n), n);
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
    if (!r || r->kind != 1) invalid_free();
    return r;
}
void free(void *p)
{
    if (!p) return;
    int e = errno;
    lock_heap(); __capstone_mallocng_free(checked(p)->raw); unlock_heap();
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
    void *q = ensure_ids(256) ? NULL : publish(__capstone_mallocng_realloc(r->raw, n), n);
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
unsigned long __capstone_sublet_malloc_linear(size_t n, sublet_cap *out)
{
    lock_heap();
    sublet_clear(out);
    void *p = ensure_ids(256) ? NULL : __capstone_mallocng_malloc(n);
    size_t base = 0;
    if (p) {
        struct capstone_malloc_slot *r = __capstone_malloc_find(p);
        sublet_give(authority(r));
        sublet_cap owned;
        sublet_take_linear(authority(r), &owned);
        base = r->key;
        size_t end = base + (n ? n : 1);
        __asm__ volatile("ldc t0, 0(%0)\nshrink t0, %1, %2\nstc t0, 0(%3)"
                         : : "r"(&owned), "r"(base), "r"(end), "r"(out)
                         : "t0", "memory");
        r->kind = 2; r->raw = NULL;
        ++allocations;
        if (++live_objects > peak_objects) peak_objects = live_objects;
    }
    unlock_heap(); return base;
}
/* Restore prefix/footer bytes after UNINIT reclamation of an explicit linear
 * loan. The backend still makes the ordinary free-list/release decision. */
extern void __capstone_mallocng_restore(struct capstone_malloc_slot *, void *);
void __capstone_sublet_free_linear(unsigned long base)
{
    lock_heap();
    struct capstone_malloc_slot *r = __capstone_malloc_find((void *)base);
    if (!r || r->kind != 2) invalid_free();
    void *p = rotate(r);
    __capstone_mallocng_restore(r, p);
    r->kind = 0; ++frees; --live_objects;
    __capstone_mallocng_free(p);
    unlock_heap();
}
void __capstone_sublet_heap_stats(unsigned long out[9])
{
    lock_heap();
    out[0]=allocations; out[1]=frees; out[2]=0; out[3]=peak_objects;
    out[4]=sublet_stats.split; out[5]=sublet_stats.mrev; out[6]=sublet_stats.delin;
    out[7]=sublet_stats.revoke; out[8]=sublet_stats.init;
    unlock_heap();
}
