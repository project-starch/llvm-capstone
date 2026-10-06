#include <errno.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include "../../sublet/sublet.h"
extern void *__capstone_vm_map(unsigned long bytes);
extern long __capstone_vm_unmap(unsigned long address, unsigned long bytes);
__attribute__((__weak__)) void __vm_wait(void) {}
void *mmap(void *addr, size_t n, int prot, int flags, int fd, off_t off)
{
    if (addr || !n || n > (64UL << 20) || prot != (PROT_READ | PROT_WRITE) ||
        flags != (MAP_PRIVATE | MAP_ANONYMOUS) || fd != -1 || off) {
        errno = EINVAL; return MAP_FAILED;
    }
    unsigned long size = 4096;
    while (size < n) size <<= 1;
    sublet_cap slot;
    sublet_store(&slot, __capstone_vm_map(size));
    unsigned long raw;
    __asm__ volatile("ld %0, 0(%1)" : "=r"(raw) : "r"(&slot) : "memory");
    if (!raw) { errno = ENOMEM; return MAP_FAILED; }
    /* Keep the kernel's ancestor as retirement authority, return copyable
     * application authority. No additional MREV is necessary for mmap. */
    void *p;
    __asm__ volatile("ldc %0, 0(%1)\ndelin %0"
                     : "=&r"(p) : "r"(&slot) : "memory");
    return p;
}
void *__mmap(void *a, size_t n, int p, int f, int fd, off_t off)
{ return mmap(a, n, p, f, fd, off); }
int munmap(void *p, size_t n)
{
    if (!p || !n || n > (64UL << 20)) { errno = EINVAL; return -1; }
    unsigned long size = 4096;
    while (size < n) size <<= 1;
    (void)*(volatile unsigned char *)p;
    long rc = __capstone_vm_unmap(__builtin_capstone_cap_get_cursor(p), size);
    if (rc) { errno = -rc; return -1; }
    return 0;
}
int __munmap(void *p, size_t n) { return munmap(p, n); }
/* Sharing and file-backed mappings have no tag-lifecycle contract yet. */
void *mremap(void *p, size_t old, size_t n, int flags, ...)
{ (void)p; (void)old; (void)n; (void)flags; errno = ENOSYS; return MAP_FAILED; }
int shmget(key_t k, size_t n, int flags)
{ (void)k; (void)n; (void)flags; errno = ENOSYS; return -1; }
void *shmat(int id, const void *p, int flags)
{ (void)id; (void)p; (void)flags; errno = ENOSYS; return (void *)-1; }
int shmdt(const void *p) { (void)p; errno = ENOSYS; return -1; }
int shmctl(int id, int cmd, struct shmid_ds *p)
{ (void)id; (void)cmd; (void)p; errno = ENOSYS; return -1; }
