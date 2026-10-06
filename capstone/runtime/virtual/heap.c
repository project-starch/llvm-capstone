/* Grow through Linux mappings; keep object lifetimes in the existing Sublet
 * slot discipline. Power-of-two blocks keep stored bounds representable.
 * One-hart profile: metadata is static and virtual contexts are serialized by
 * the adapter; no allocator recursion. */
#include <errno.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "sublet.h"

#define ARENAS 256
#define BLOCKS 8192
#define MIN_BLOCK 256UL
#define MIN_ARENA 65536UL
struct arena { unsigned long base, bytes, block; unsigned used; };
struct block { sublet_cap slot; unsigned long base, requested; unsigned arena, live; };
static struct arena arenas[ARENAS];
static struct block blocks[BLOCKS];
extern void *__capstone_vm_map(unsigned long bytes);
extern long __capstone_vm_unmap(unsigned long address, unsigned long bytes);
static unsigned long free_ids(void)
{
    unsigned long n;
    __asm__ volatile("csrr %0, 0xcc0" : "=r"(n));
    return n;
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
    for (a = 0; a < ARENAS && arenas[a].bytes; ++a) {}
    if (a == ARENAS) return -1;
    need = bytes / size;
    /* Two root identities, need-1 splits and the first object's MREV.
     * The one-hart adapter serializes virtual contexts, so this count cannot
     * race another allocator in this namespace. Retirement itself allocates
     * no nodes. */
    if (free_ids() < need + 2) return -1;
    for (unsigned i = 0; i < BLOCKS && count < need; ++i)
        if (!blocks[i].base) ids[count++] = i;
    if (count != need) return -1;
    sublet_store(&rest, __capstone_vm_map(bytes));
    /* LCC reads a scalar zero as untagged; never read the linear grant into
     * a C variable merely to test it. */
    unsigned long base;
    __asm__ volatile("ld %0, 0(%1)" : "=r"(base) : "r"(&rest) : "memory");
    if (!base) return -1;
    base = sublet_base(&rest);
    arenas[a] = (struct arena){base, bytes, size, 0};
    for (unsigned i = 0; i < need; ++i) {
        struct block *b = &blocks[ids[i]];
        b->base = base + i * size; b->arena = a;
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
static void *allocate(size_t n, size_t alignment)
{
    unsigned long size = power(n > alignment ? n : alignment);
    if (!size || !free_ids()) { errno = ENOMEM; return NULL; }
    for (unsigned retry = 0; retry < 2; ++retry) {
        for (unsigned i = 0; i < BLOCKS; ++i) {
            struct block *b = &blocks[i];
            if (b->base && !b->live && arenas[b->arena].block == size) {
                b->live = 1; b->requested = n; ++arenas[b->arena].used;
                void *p = sublet_take(&b->slot);
                unsigned long length = n ? n : 1;
                if (length >= 4096) {
                    unsigned long grain = 1UL << (63 - __builtin_clzl(length) - 9);
                    length = (length + grain - 1) & ~(grain - 1);
                }
                return __builtin_capstone_cap_shrink(p, b->base, b->base + length);
            }
        }
        if (grow(size)) break;
    }
    errno = ENOMEM; return NULL;
}
void *malloc(size_t n) { return allocate(n ? n : 1, 16); }
void *__libc_malloc(size_t n) { return malloc(n); }
void *__simple_malloc(size_t n) { return malloc(n); }

static struct block *lookup(void *p)
{
    unsigned long address = __builtin_capstone_cap_get_cursor(p);
    /* A stale free must fault before it can touch a replacement's handle. */
    (void)*(volatile unsigned char *)p;
    for (unsigned i = 0; i < BLOCKS; ++i)
        if (blocks[i].base == address && blocks[i].live) return &blocks[i];
    __builtin_trap();
}
void free(void *p)
{
    if (!p) return;
    struct block *b = lookup(p);
    memset(p, 0, __builtin_capstone_cap_get_end(p) - b->base);
    sublet_give(&b->slot);
    b->live = 0; --arenas[b->arena].used;
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
    struct block *b = lookup(p);
    void *q = malloc(n);
    if (!q) return NULL;
    memcpy(q, p, n < b->requested ? n : b->requested);
    free(p); return q;
}
size_t malloc_usable_size(void *p)
{
    if (!p) return 0;
    struct block *b = lookup(p);
    return __builtin_capstone_cap_get_end(p) - b->base;
}
void *aligned_alloc(size_t align, size_t n)
{
    if (!align || (align & (align - 1)) || align < sizeof(void *) || n % align) {
        errno = EINVAL; return NULL;
    }
    return allocate(n ? n : 1, align);
}
int posix_memalign(void **out, size_t align, size_t n)
{
    if (!align || (align & (align - 1)) || align < sizeof(void *)) return EINVAL;
    void *p = allocate(n ? n : 1, align);
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
    for (unsigned a = 0; a < ARENAS; ++a) {
        if (!arenas[a].bytes || arenas[a].used) continue;
        if (__capstone_vm_unmap(arenas[a].base, arenas[a].bytes)) continue;
        for (unsigned i = 0; i < BLOCKS; ++i) {
            if (blocks[i].base && blocks[i].arena == a) {
                /* Scalar clearing drops dead tags without loading them. */
                unsigned long *p = (unsigned long *)&blocks[i];
                for (unsigned k = 0; k < sizeof(blocks[i]) / sizeof(*p); ++k) p[k] = 0;
            }
        }
        memset(&arenas[a], 0, sizeof(arenas[a])); released = 1;
    }
    return released;
}
