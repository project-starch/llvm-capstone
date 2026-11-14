#ifndef MALLOC_HEAP_SUBLET_H
#define MALLOC_HEAP_SUBLET_H
#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#ifndef CAPSTONE_HEAP_LIFETIMES
#define CAPSTONE_HEAP_LIFETIMES 1
#endif
#ifndef CAPSTONE_HEAP_OBJECT_BOUNDS
#define CAPSTONE_HEAP_OBJECT_BOUNDS 1
#endif
#if CAPSTONE_HEAP_LIFETIMES && !CAPSTONE_HEAP_OBJECT_BOUNDS
#error "object lifetimes require object bounds"
#endif
#if CAPSTONE_HEAP_SCRUB_FREE
#error "the minimal allocator does not scrub freed payloads"
#endif
#define HEAP_POLICY_NAME_(l,b) __malloc_policy_##l##b
#define HEAP_POLICY_NAME(l,b) HEAP_POLICY_NAME_(l,b)
#define HEAP_POLICY_SYMBOL HEAP_POLICY_NAME(CAPSTONE_HEAP_LIFETIMES,CAPSTONE_HEAP_OBJECT_BOUNDS)
__attribute__((visibility("hidden"))) void HEAP_POLICY_SYMBOL(void);

/* Same compressed-bounds rule as runtime/include/capstone/representable.h.
 * Requests are checked for overflow before this helper is called. */
static inline size_t heap_granule(size_t n)
{
    return n < 4096 ? 1 : (size_t)1 << (63-__builtin_clzl(n)-9);
}
static inline size_t heap_round(size_t n)
{
    if (!n) return 1;
    size_t g = heap_granule(n), r = (n+g-1) & -g;
    g = heap_granule(r);
    return (r+g-1) & -g;
}


/* Mapping index for the Sv39 virtual profile. No object or group records. */
struct heap_leaf { void *owner[512]; };
struct heap_mid { struct heap_leaf *leaf[512]; };
__attribute__((visibility("hidden"))) extern struct heap_mid *__malloc_page_root[512];
__attribute__((visibility("hidden"))) void *__heap_mapping(size_t);
__attribute__((visibility("hidden"))) int __heap_unmapping(void *, size_t);

static inline void *heap_issue(void *at, size_t n)
{
    void *child;
    /* at already has the parent's identity and the desired cursor. Hardware
     * validates tag, liveness, bounds, size and node availability. */
    __asm__ volatile(".insn r 0x5b, 0x1, 0x51, %0, %1, %2"
                     : "=r"(child) : "r"(at), "r"(n ? n : 1) : "memory");
    return child;
}
static inline void heap_end(void *object)
{
    /* CREVOKE takes the object alone. Hardware validates its tag and liveness
     * and ends everything derived inside it. */
    __asm__ volatile(".insn r 0x5b, 0x1, 0x52, x0, %0, x0"
                     : : "r"(object) : "memory");
}
static inline void *heap_renew(void *object)
{
    void *renewed;
    /* CRENEW: the object's lifetime ends and a new one begins in the same
     * place, with the cursor, bounds and rights of the operand. */
    __asm__ volatile(".insn r 0x5b, 0x1, 0x54, %0, %1, x0"
                     : "=r"(renewed) : "r"(object) : "memory");
    return renewed;
}
/* malloc/free's compile-time controls; nested allocators always use the direct
 * primitives above, independently of the outer heap's measurement arm.
 *
 * The failure result: a CDERIVE that finds no free slot traps with cause 30,
 * and the kernel grows the task's lifetime table. When it cannot, it sets
 * NOGROW (ensure_nodes, kernel patch 0011), and the resumed CDERIVE writes an
 * untagged 0 instead of trapping again. malloc then returns NULL with ENOMEM.
 * heap_alloc_issue,
 * for a slot that malloc or aligned_alloc just took, gives that slot back
 * (__malloc_unissue: mallocng's free without a lifetime to end).
 * heap_alloc_reissue, for realloc's in-place paths, returns 0 and leaves the
 * slot and the object's old lifetime as they are. realloc ends the old object
 * only after the new one exists: in place after heap_alloc_reissue, and a
 * resize that would move a mapping takes malloc, memcpy and free, so a failed
 * realloc returns NULL with the original intact (realloc.c, patch 0009). */
__attribute__((visibility("hidden"))) void __malloc_unissue(void *);
#if CAPSTONE_HEAP_DERIVE_INJECTION
/* Test builds only: the derive whose 1-based count equals
 * __heap_derive_fail_at gives the failure result without executing CDERIVE. */
__attribute__((__weak__, visibility("hidden"))) unsigned long __heap_derive_calls;
__attribute__((__weak__, visibility("hidden"))) unsigned long __heap_derive_fail_at;
#endif
static inline void *heap_alloc_derive(void *p, size_t n)
{
#if CAPSTONE_HEAP_DERIVE_INJECTION
    if (++__heap_derive_calls == __heap_derive_fail_at) return 0;
#endif
    return heap_issue(p, n);
}
static inline void *heap_alloc_reissue(void *p, size_t n)
{
    if (!p) return p;
#if CAPSTONE_HEAP_LIFETIMES
    void *child = heap_alloc_derive(p, n);
    if (!child) errno = ENOMEM;
    return child;
#elif CAPSTONE_HEAP_OBJECT_BOUNDS
    size_t base = __builtin_capstone_cap_get_cursor(p);
    return __builtin_capstone_cap_shrink(p, base, base+(n ? n : 1));
#else
    return p;
#endif
}
static inline void *heap_alloc_issue(void *p, size_t n)
{
    if (!p) return p;
#if CAPSTONE_HEAP_LIFETIMES
    void *child = heap_alloc_derive(p, n);
    if (!child) {
        __malloc_unissue(p);
        errno = ENOMEM;
    }
    return child;
#elif CAPSTONE_HEAP_OBJECT_BOUNDS
    size_t base = __builtin_capstone_cap_get_cursor(p);
    return __builtin_capstone_cap_shrink(p, base, base+(n ? n : 1));
#else
    return p;
#endif
}
static inline void heap_alloc_end(void *object)
{
#if CAPSTONE_HEAP_LIFETIMES
    heap_end(object);
#else
    (void)object;
#endif
}
static inline void *heap_access(void *p)
{
    /* Keep the supplied capability live and at its allocation start. An
     * integer address alone must never let a stale free affect a new object. */
    __asm__ volatile("lbu zero, 0(%0)" : : "r"(p) : "memory");
    size_t a = __builtin_capstone_cap_get_cursor(p);
    if (a >> 39) __builtin_trap();
#if CAPSTONE_HEAP_OBJECT_BOUNDS
    if (a != __builtin_capstone_cap_get_base(p)) __builtin_trap();
#endif
    struct heap_mid *mid = __malloc_page_root[a >> 30];
    if (!mid) __builtin_trap();
    struct heap_leaf *leaf = mid->leaf[(a >> 21) & 511];
    if (!leaf) __builtin_trap();
    void *parent = leaf->owner[(a >> 12) & 511];
    if (!parent) __builtin_trap();
    void *at;
    __asm__(".insn r 0x5b, 0x1, 0x05, %0, %1, %2"
            : "=r"(at) : "r"(parent), "r"(a));
    return at;
}
#endif
