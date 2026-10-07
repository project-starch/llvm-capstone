/* Capability ABI only. Allocation policy executes in native upstream musl. */
#include <errno.h>
#include <stdint.h>
#include <stddef.h>
#include "vm.h"
extern void *__capstone_vm_heap(void *, size_t, unsigned, size_t);
__attribute__((noinline)) static void validate(void *p)
{
    __asm__ volatile(".global cap_malloc_validate\ncap_malloc_validate:\nlbu zero, 0(%0)"
                     : : "r"(p) : "memory");
}
__attribute__((noinline, noreturn)) static void invalid_free(void)
{
    __asm__ volatile(".global cap_malloc_invalid_free\ncap_malloc_invalid_free:");
    __builtin_trap();
}
static void *heap_call(void *old, size_t bytes, unsigned op, size_t alignment)
{
    sublet_cap result;
    unsigned long raw;
    /* Validate before exposing a cursor to the trusted scalar allocator. */
    if (old) validate(old);
    sublet_store(&result, __capstone_vm_heap(old, bytes, op, alignment));
    __asm__ volatile("ld %0, 0(%1)" : "=r"(raw) : "r"(&result) : "memory");
    if ((long)raw < 0) { errno = -(long)raw; sublet_clear(&result); return NULL; }
    void *p;
    __asm__ volatile("ldc %0, 0(%1)" : "=r"(p) : "r"(&result) : "memory");
    return p;
}
void *malloc(size_t n) { return heap_call(NULL, n, CV_HEAP_MALLOC, 0); }
void *__libc_malloc(size_t n) { return malloc(n); }
void *__simple_malloc(size_t n) { return malloc(n); }
void __libc_free(void *p);
void free(void *p)
{
    if (!p) return;
    int saved = errno;
    errno = 0;
    heap_call(p, 0, CV_HEAP_FREE, 0);
    if (errno) invalid_free();
    errno = saved;
}
void __libc_free(void *p) { free(p); }
void *calloc(size_t n, size_t size)
{
    if (size && n > SIZE_MAX / size) { errno = ENOMEM; return NULL; }
    return heap_call(NULL, n * size, CV_HEAP_CALLOC, 0);
}
void *realloc(void *p, size_t n) { return heap_call(p, n, CV_HEAP_REALLOC, 0); }
void *aligned_alloc(size_t a, size_t n) { return heap_call(NULL, n, CV_HEAP_ALIGNED, a); }
int posix_memalign(void **out, size_t a, size_t n)
{
    if (!a || (a & (a - 1)) || a < sizeof(void *)) return EINVAL;
    int saved = errno;
    void *p = aligned_alloc(a, n);
    int error = p ? 0 : errno;
    errno = saved;
    if (!error) *out = p;
    return error;
}
void *memalign(size_t a, size_t n) { return aligned_alloc(a, n); }
size_t malloc_usable_size(void *p)
{
    if (!p) return 0;
    __asm__ volatile("lbu zero, 0(%0)" : : "r"(p) : "memory");
    return __builtin_capstone_cap_get_end(p) - __builtin_capstone_cap_get_base(p);
}
/* musl already decides when to unmap groups; there is no extra trim policy. */
int malloc_trim(size_t pad) { (void)pad; return 0; }
unsigned long __capstone_sublet_malloc_linear(size_t n, sublet_cap *out)
{
    sublet_store(out, heap_call(NULL, n, CV_HEAP_LINEAR, 0));
    unsigned long raw;
    __asm__ volatile("ld %0, 0(%1)" : "=r"(raw) : "r"(out) : "memory");
    return raw;
}
void __capstone_sublet_free_linear(unsigned long address)
{
    unsigned long status;
    __asm__ volatile("mv a0, %1\nli a1, 0\nli a2, %2\nli a3, 0\nli a7, %3\necall\nmv %0, a0"
                     : "=r"(status) : "r"(address), "i"(CV_HEAP_FREE_LINEAR), "i"(CV_SERVICE_HEAP)
                     : "a0", "a1", "a2", "a3", "a7", "memory");
    if ((long)status < 0) __builtin_trap();
}
void __capstone_sublet_heap_stats(unsigned long out[9])
{ heap_call(out, 72, CV_HEAP_STATISTICS, 0); }
