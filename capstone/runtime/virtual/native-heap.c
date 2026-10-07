/* Native-side bridge. mallocng owns every allocation decision. Linux, this
 * bridge and the module are trusted; no allocator metadata is exposed as a
 * capability. Link against the verified native musl build, not host glibc. */
#define _GNU_SOURCE
#include "wire.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdint.h>
#include <stdarg.h>

static int heap_device = -1;
/* memcpy is also called during musl's initial TLS construction, before tp is
 * usable. Never read a TLS variable from this wrapper. The service lock
 * serializes owners; another native worker follows its ordinary memcpy path. */
static int heap_native_owner;
static void *heap_old_pointer;
void capstone_native_heap_init(int fd) { heap_device = fd; }
extern void *__real_memcpy(void *, const void *, size_t);
extern int __real___munmap(void *, size_t);
extern void *__real___mremap(void *, size_t, size_t, int, ...);
extern void __real___libc_free(void *);

void __wrap___libc_free(void *p)
{
    if (__atomic_load_n(&heap_native_owner, __ATOMIC_ACQUIRE) &&
        syscall(SYS_gettid) == __atomic_load_n(&heap_native_owner, __ATOMIC_RELAXED) &&
        p && p == heap_old_pointer) {
        uint64_t address = (uintptr_t)p;
        if (ioctl(heap_device, CV_HEAP_RELEASE, &address)) _exit(126);
        heap_old_pointer = NULL;
    }
    __real___libc_free(p);
}

void *__wrap_memcpy(void *dst, const void *src, size_t bytes)
{
    if (!__atomic_load_n(&heap_native_owner, __ATOMIC_ACQUIRE) ||
        syscall(SYS_gettid) != __atomic_load_n(&heap_native_owner, __ATOMIC_RELAXED) ||
        !bytes || ((uintptr_t)dst | (uintptr_t)src) & 15)
        return __real_memcpy(dst, src, bytes);
    struct cv_heap_copy r = {(uintptr_t)dst, (uintptr_t)src, bytes};
    if (ioctl(heap_device, CV_HEAP_COPY, &r)) _exit(126);
    return dst;
}
int __wrap___munmap(void *address, size_t bytes)
{
    /* Pins keep physical tags inspectable even after Linux removes the VMA. */
    int result = __real___munmap(address, bytes);
    if (!result && heap_device >= 0) {
        struct cv_heap_range r = {(uintptr_t)address, (bytes + 4095) & -4096UL, 0, 0};
        if (ioctl(heap_device, CV_HEAP_RANGE, &r)) _exit(126);
    }
    return result;
}
int __wrap_munmap(void *address, size_t bytes)
{ return __wrap___munmap(address, bytes); }
void *__wrap___mremap(void *address, size_t old, size_t bytes, int flags, ...)
{
    void *target = NULL;
    if (flags & MREMAP_FIXED) {
        va_list args; va_start(args, flags); target = va_arg(args, void *); va_end(args);
    }
    void *result = __real___mremap(address, old, bytes, flags, target);
    if (result != MAP_FAILED && heap_device >= 0) {
        struct cv_heap_range r = {(uintptr_t)address, (old + 4095) & -4096UL,
                                 (uintptr_t)result, (bytes + 4095) & -4096UL};
        if (ioctl(heap_device, CV_HEAP_RANGE, &r)) _exit(126);
    }
    return result;
}
long capstone_native_heap(uint64_t tid, struct cv_step *step)
{
    void *old = (void *)(uintptr_t)step->args[0], *result = NULL;
    size_t bytes = step->args[1], alignment = step->args[3];
    unsigned op = step->args[2];
    struct cv_heap r = {tid, op, (uintptr_t)old, bytes};
    int error = 0;
    if (op == CV_HEAP_STATISTICS) {
        struct cv_heap_stats stats = {tid, (uintptr_t)old};
        return ioctl(heap_device, CV_HEAP_STATS, &stats) ? -errno : 0;
    }
    if (ioctl(heap_device, CV_HEAP_BEGIN, &r)) return -errno;
    heap_old_pointer = old;
    __atomic_store_n(&heap_native_owner, syscall(SYS_gettid), __ATOMIC_RELEASE);
    errno = 0;
    switch (op) {
    case CV_HEAP_MALLOC: case CV_HEAP_LINEAR: result = malloc(bytes); break;
    case CV_HEAP_CALLOC: result = calloc(1, bytes); break;
    case CV_HEAP_ALIGNED: result = aligned_alloc(alignment, bytes); break;
    case CV_HEAP_REALLOC: result = realloc(old, bytes); break;
    case CV_HEAP_FREE: case CV_HEAP_FREE_LINEAR: free(old); break;
    default: error = EINVAL; break;
    }
    if (!result && op != CV_HEAP_FREE && op != CV_HEAP_FREE_LINEAR)
        error = errno ? errno : ENOMEM;
    __atomic_store_n(&heap_native_owner, 0, __ATOMIC_RELEASE);
    heap_old_pointer = NULL;
    r.address = (uintptr_t)result;
    if (ioctl(heap_device, CV_HEAP_COMMIT, &r)) _exit(126);
    return error ? -error : 0; /* Successful allocations use the tagged reply. */
}
