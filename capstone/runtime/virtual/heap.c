/* Grow through Linux mappings; keep object lifetimes in the existing Sublet
 * slot discipline. Power-of-two blocks keep stored bounds representable.
 * A scalar atomic mutex protects metadata across guest preemption. Contended
 * callers yield through the OS adapter; uncontended allocations stay local. */
#include <errno.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "vm.h"

#ifndef CAPSTONE_VIRTUAL_ARENAS
#define CAPSTONE_VIRTUAL_ARENAS 256
#endif
#ifndef CAPSTONE_VIRTUAL_BLOCKS
#define CAPSTONE_VIRTUAL_BLOCKS 65536
#endif
#define ARENAS CAPSTONE_VIRTUAL_ARENAS
#define BLOCKS CAPSTONE_VIRTUAL_BLOCKS
#define MIN_BLOCK 256UL
#define MIN_ARENA 65536UL
struct arena { unsigned long base, bytes, block; unsigned used, free_head; };
struct block {
    sublet_cap slot;
    unsigned long base, requested;
    unsigned arena, live, next_free, index;
};
#define BLOCK_SLAB 256U
#define ARENA_SLAB 32U
_Static_assert(BLOCKS > 0 && ARENAS > 0, "positive virtual allocator limits");
static struct block initial_blocks[BLOCK_SLAB];
static struct arena initial_arenas[ARENA_SLAB];
static struct block *block_slabs[(BLOCKS + BLOCK_SLAB - 1) / BLOCK_SLAB] = {initial_blocks};
static struct arena *arena_slabs[(ARENAS + ARENA_SLAB - 1) / ARENA_SLAB] = {initial_arenas};
static unsigned block_capacity = BLOCKS < BLOCK_SLAB ? BLOCKS : BLOCK_SLAB;
static unsigned arena_capacity = ARENAS < ARENA_SLAB ? ARENAS : ARENA_SLAB;
static unsigned long allocations, frees, live_objects, peak_objects;
static struct block *block_at(unsigned i)
{ return &block_slabs[i / BLOCK_SLAB][i % BLOCK_SLAB]; }
static struct arena *arena_at(unsigned i)
{ return &arena_slabs[i / ARENA_SLAB][i % ARENA_SLAB]; }
/* Metadata grows through the same VM service without recursing into malloc.
 * These mappings have a separate lifetime from every payload arena. */
static void *metadata(size_t bytes)
{
    sublet_cap slot;
    bytes = (bytes + 4095) & ~4095UL;
    if (cap_vm_acquire(&slot, bytes, 4096, PROT_READ | PROT_WRITE,
                       6, CAP_VM_METADATA)) return NULL;
    void *p = cap_vm_copyable(&slot);
    memset(p, 0, bytes);
    return p;
}
static int grow_blocks(void)
{
    if (block_capacity == BLOCKS) return -1;
    void *p = metadata(BLOCK_SLAB * sizeof(struct block));
    if (!p) return -1;
    block_slabs[block_capacity / BLOCK_SLAB] = p;
    unsigned n = BLOCKS - block_capacity;
    block_capacity += n < BLOCK_SLAB ? n : BLOCK_SLAB;
    return 0;
}
static int grow_arenas(void)
{
    if (arena_capacity == ARENAS) return -1;
    void *p = metadata(ARENA_SLAB * sizeof(struct arena));
    if (!p) return -1;
    arena_slabs[arena_capacity / ARENA_SLAB] = p;
    unsigned n = ARENAS - arena_capacity;
    arena_capacity += n < ARENA_SLAB ? n : ARENA_SLAB;
    return 0;
}
static unsigned heap_mutex;
static void cap_vm_heap_lock(void)
{
    while (__atomic_exchange_n(&heap_mutex, 1, __ATOMIC_ACQUIRE))
        if (__capstone_vm_wait()) __builtin_trap();
}
static void cap_vm_heap_unlock(void)
{ __atomic_store_n(&heap_mutex, 0, __ATOMIC_RELEASE); }
static unsigned long free_ids(void)
{
    unsigned long n;
    __asm__ volatile("csrr %0, 0xcc0" : "=r"(n));
    return n;
}
static int ensure_ids(unsigned long need)
{
    if (free_ids() >= need + CV_NODE_RESERVE) return 0;
    long rc = __capstone_vm_nodes(need);
    if (rc < 0) { errno = -rc; return -1; }
    if (free_ids() < need + CV_NODE_RESERVE) { errno = ENOMEM; return -1; }
    return 0;
}

static unsigned long power(unsigned long n)
{
    unsigned long p = MIN_BLOCK;
    if (n > (256UL << 20)) return 0;
    while (p < n) p <<= 1;
    return p;
}
static int grow(unsigned long size)
{
    unsigned a, ids[256], count = 0, need;
    unsigned long bytes = size < MIN_ARENA ? MIN_ARENA : size;
    sublet_cap rest;
    for (a = 0; a < arena_capacity && arena_at(a)->bytes; ++a) {}
    if (a == arena_capacity && grow_arenas()) return -1;
    need = bytes / size;
    for (unsigned i = 0; count < need; ++i) {
        if (i == block_capacity && grow_blocks()) return -1;
        if (!block_at(i)->base) ids[count++] = i;
    }
    /* Account after metadata growth: two roots, need-1 splits and MREV.
     * The heap mutex protects the allocator state across service calls. */
    if (ensure_ids(need + 2)) return -1;
    if (cap_vm_acquire(&rest, bytes, bytes, PROT_READ | PROT_WRITE,
                       6, CAP_VM_HEAP)) return -1;
    unsigned long base;
    base = sublet_base(&rest);
    *arena_at(a) = (struct arena){base, bytes, size, 0, 0};
    for (unsigned i = 0; i < need; ++i) {
        struct block *b = block_at(ids[i]);
        b->base = base + i * size; b->arena = a; b->index = ids[i];
        b->live = 0; b->requested = 0;
        b->next_free = arena_at(a)->free_head;
        arena_at(a)->free_head = ids[i] + 1;
        if (i + 1 == need) sublet_move(&rest, &b->slot);
        else {
            sublet_cap tail;
            sublet_split(&rest, b->base + size, &tail);
            sublet_move(&rest, &b->slot);
            sublet_move(&tail, &rest);
        }
    }
    return 0;
}
static struct block *reserve_block(size_t n, size_t alignment)
{
    unsigned long size = power(n > alignment ? n : alignment);
    if (!size || ensure_ids(1)) { errno = ENOMEM; return NULL; }
    for (unsigned retry = 0; retry < 2; ++retry) {
        for (unsigned a = 0; a < arena_capacity; ++a) {
            struct arena *arena = arena_at(a);
            if (arena->bytes && arena->block == size && arena->free_head) {
                unsigned index = arena->free_head - 1;
                struct block *b = block_at(index);
                arena->free_head = b->next_free;
                b->next_free = 0; b->live = 1; b->requested = n;
                ++arena->used;
                ++allocations;
                if (++live_objects > peak_objects) peak_objects = live_objects;
                return b;
            }
        }
        if (grow(size)) break;
    }
    errno = ENOMEM; return NULL;
}
static void *allocate(size_t n, size_t alignment)
{
    struct block *b = reserve_block(n, alignment);
    if (!b) return NULL;
    void *p = sublet_take(&b->slot);
    unsigned long length = n ? n : 1;
    if (length >= 4096) {
        unsigned long grain = 1UL << (63 - __builtin_clzl(length) - 9);
        length = (length + grain - 1) & ~(grain - 1);
    }
    return __builtin_capstone_cap_shrink(p, b->base, b->base + length);
}
void *malloc(size_t n)
{
    cap_vm_heap_lock();
    void *p = allocate(n ? n : 1, 16);
    cap_vm_heap_unlock();
    return p;
}
void *__libc_malloc(size_t n) { return malloc(n); }
void *__simple_malloc(size_t n) { return malloc(n); }

static struct block *lookup(void *p)
{
    unsigned long address = __builtin_capstone_cap_get_cursor(p);
    /* A stale free must fault before it can touch a replacement's handle. */
    (void)*(volatile unsigned char *)p;
    for (unsigned i = 0; i < block_capacity; ++i) {
        struct block *b = block_at(i);
        if (b->base == address && b->live) return b;
    }
    __builtin_trap();
}
static void return_block(struct block *b)
{
    sublet_give(&b->slot);
    ++frees; --live_objects;
    b->live = 0; --arena_at(b->arena)->used;
    b->next_free = arena_at(b->arena)->free_head;
    arena_at(b->arena)->free_head = b->index + 1;
}
/* Existing nested allocators borrow a whole linear block and return its
 * scalar base. The retained senior handle revokes every derived lifetime.
 * This is an internal allocator API, not the public free(pointer) contract. */
unsigned long __capstone_sublet_malloc_linear(size_t n, sublet_cap *out)
{
    cap_vm_heap_lock();
    struct block *b = reserve_block(n ? n : 1, 16);
    unsigned long base = 0;
    if (b) {
        b->live = 2;
        base = sublet_take_linear(&b->slot, out);
    } else sublet_clear(out);
    cap_vm_heap_unlock();
    return base;
}
void __capstone_sublet_free_linear(unsigned long base)
{
    cap_vm_heap_lock();
    for (unsigned i = 0; i < block_capacity; ++i) {
        struct block *b = block_at(i);
        if (b->base == base && b->live == 2) {
            return_block(b);
            cap_vm_heap_unlock();
            return;
        }
    }
    __builtin_trap();
}
void __capstone_sublet_heap_stats(unsigned long out[9])
{
    cap_vm_heap_lock();
    out[0] = allocations; out[1] = frees;
    out[2] = 0; /* size-class VM arenas do not perform buddy merges */
    out[3] = peak_objects;
    out[4] = sublet_stats.split; out[5] = sublet_stats.mrev;
    out[6] = sublet_stats.delin; out[7] = sublet_stats.revoke;
    out[8] = sublet_stats.init;
    cap_vm_heap_unlock();
}
static void free_locked(void *p)
{
    if (!p) return;
    struct block *b = lookup(p);
    if (b->live != 1) __builtin_trap();
    memset(p, 0, __builtin_capstone_cap_get_end(p) - b->base);
    return_block(b);
}
void free(void *p)
{
    if (!p) return;
    cap_vm_heap_lock();
    free_locked(p);
    cap_vm_heap_unlock();
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
    if (!n) { free(p); return NULL; }
    cap_vm_heap_lock();
    struct block *b = lookup(p);
    void *q = allocate(n, 16);
    if (q) {
        memcpy(q, p, n < b->requested ? n : b->requested);
        free_locked(p);
    }
    cap_vm_heap_unlock();
    return q;
}
size_t malloc_usable_size(void *p)
{
    if (!p) return 0;
    cap_vm_heap_lock();
    struct block *b = lookup(p);
    size_t n = __builtin_capstone_cap_get_end(p) - b->base;
    cap_vm_heap_unlock();
    return n;
}
void *aligned_alloc(size_t align, size_t n)
{
    if (!align || (align & (align - 1)) || align < sizeof(void *) || n % align) {
        errno = EINVAL; return NULL;
    }
    cap_vm_heap_lock();
    void *p = allocate(n ? n : 1, align);
    cap_vm_heap_unlock();
    return p;
}
int posix_memalign(void **out, size_t align, size_t n)
{
    if (!align || (align & (align - 1)) || align < sizeof(void *)) return EINVAL;
    cap_vm_heap_lock();
    void *p = allocate(n ? n : 1, align);
    cap_vm_heap_unlock();
    if (!p) return ENOMEM;
    *out = p; return 0;
}
void *memalign(size_t align, size_t n)
{
    void *p; int error = posix_memalign(&p, align, n);
    if (error) { errno = error; return NULL; }
    return p;
}
/* Return every wholly free arena to Linux after retiring all descendants.
 * Its old tags are cleared before any retired identity can be recycled. */
int malloc_trim(size_t pad)
{
    int released = 0; (void)pad;
    cap_vm_heap_lock();
    for (unsigned a = 0; a < arena_capacity; ++a) {
        if (!arena_at(a)->bytes || arena_at(a)->used) continue;
        if (__capstone_vm_unmap(arena_at(a)->base, arena_at(a)->bytes)) continue;
        for (unsigned i = 0; i < block_capacity; ++i) {
            struct block *b = block_at(i);
            if (b->base && b->arena == a) {
                /* Scalar clearing drops dead tags without loading them. */
                unsigned long *p = (unsigned long *)b;
                for (unsigned k = 0; k < sizeof(*b) / sizeof(*p); ++k) p[k] = 0;
            }
        }
        memset(arena_at(a), 0, sizeof(struct arena)); released = 1;
    }
    cap_vm_heap_unlock();
    return released;
}
