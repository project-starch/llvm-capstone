#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdarg.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <syscall.h>
__attribute__((__weak__)) void __vm_wait(void) {}

static void *mapping_result(void *p)
{
    /* Inspect the address only for Linux's error range. Return the original
     * reference on success, without rebuilding it from that address. */
    unsigned long r = (unsigned long)p;
    if (r >= (unsigned long)-4095) {
        errno = -(long)r;
        return MAP_FAILED;
    }
    return p;
}
void *__mmap(void *addr, size_t n, int prot, int flags, int fd, off_t off)
{
    return mapping_result(__capstone_linux_syscall_ref(SYS_mmap, addr,
        __scc(n), __scc(prot), __scc(flags), __scc(fd), __scc(off)));
}
void *mmap(void *a, size_t n, int p, int f, int fd, off_t off)
{ return __mmap(a, n, p, f, fd, off); }
int __munmap(void *p, size_t n)
{ return __syscall_ret(__syscall(SYS_munmap, p, n)); }
int munmap(void *p, size_t n) { return __munmap(p, n); }
int __mprotect(void *p, size_t n, int prot)
{ return __syscall_ret(__syscall(SYS_mprotect, p, n, prot)); }
int mprotect(void *p, size_t n, int prot) { return __mprotect(p, n, prot); }
void *mremap(void *p, size_t old, size_t n, int flags, ...)
{
    void *to = 0;
    if (flags & MREMAP_FIXED) {
        va_list ap;
        va_start(ap, flags);
        to = va_arg(ap, void *);
        va_end(ap);
    }
    return mapping_result(__capstone_linux_syscall_ref(SYS_mremap, p,
        __scc(old), __scc(n), __scc(flags), to, 0));
}
/* Shared address spaces are outside the initial trap-route contract. */
int shmget(key_t key, size_t n, int flags) { errno = ENOSYS; return -1; }
void *shmat(int id, const void *p, int flags) { errno = ENOSYS; return (void *)-1; }
int shmdt(const void *p) { errno = ENOSYS; return -1; }
int shmctl(int id, int cmd, struct shmid_ds *p) { errno = ENOSYS; return -1; }
