#include <errno.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include "../../sublet/sublet.h"
extern void *__capstone_vm_map(unsigned long bytes);
extern long __capstone_vm_unmap(unsigned long address, unsigned long bytes);
__attribute__((__weak__)) void __vm_wait(void) {}
void *mmap(void *addr, size_t n, int prot, int flags, int fd, off_t off)
{
    if (flags & MAP_HUGETLB) { errno = ENOMEM; return MAP_FAILED; }
    /* Preserve the physical runtime's process-local MAP_SHARED compatibility.
     * The backing stays private: fork and shared tagged pages are unsupported. */
    if (addr || !n || n > (256UL << 20) || prot != (PROT_READ | PROT_WRITE) ||
        (flags != (MAP_PRIVATE | MAP_ANONYMOUS) && flags != (MAP_SHARED | MAP_ANONYMOUS)) || fd != -1 || off) {
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
    if (!p || !n || n > (256UL << 20)) { errno = EINVAL; return -1; }
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

#define L0_PAGE 4096UL
#define L0_MAX_SEGS 16
struct l0_seg {
	key_t key;
	void *base, *block;
	size_t len;              /* as requested; the block is whole pages */
	unsigned long nattch;
	int used, removed;
};
static struct l0_seg segs[L0_MAX_SEGS];

static size_t pages(size_t len)
{
	return (len + L0_PAGE - 1) & ~(L0_PAGE - 1);
}

/* A zeroed, page-aligned block of at least len bytes from level0. The base is
   the raw block moved up to the boundary -- pointer arithmetic on what malloc
   returned, so the capability survives -- and the raw block is what free()
   gets back. Only the address feeds the alignment computation. */
static void *page_block(size_t len, void **block)
{
	/* Account for both page rounding and the extra alignment page before
	   arithmetic can wrap a huge mapping into a small allocation. */
	if (len >= PTRDIFF_MAX || len > SIZE_MAX - (2 * L0_PAGE - 1))
		return 0;
	size_t want = pages(len);
	char *raw = malloc(want + L0_PAGE);
	if (!raw)
		return 0;
	size_t skew = (size_t)((uintptr_t)raw & (L0_PAGE - 1));
	char *base = raw + (skew ? L0_PAGE - skew : 0);
	memset(base, 0, want);
	*block = raw;
	return base;
}

static struct l0_seg *seg_of(int id)
{
	if (id < 1 || id > L0_MAX_SEGS || !segs[id - 1].used)
		return 0;
	return &segs[id - 1];
}

static void seg_drop(struct l0_seg *s)
{
	free(s->block);
	memset(s, 0, sizeof *s);
}

int shmget(key_t key, size_t size, int flag)
{
	int i;
	if (key != IPC_PRIVATE) {
		for (i = 0; i < L0_MAX_SEGS; i++) {
			struct l0_seg *s = &segs[i];
			if (!s->used || s->removed || s->key != key)
				continue;
			if ((flag & IPC_CREAT) && (flag & IPC_EXCL)) {
				errno = EEXIST;
				return -1;
			}
			if (size > s->len) {
				errno = EINVAL;
				return -1;
			}
			return i + 1;
		}
		if (!(flag & IPC_CREAT)) {
			errno = ENOENT;
			return -1;
		}
	}
	if (size == 0) {
		errno = EINVAL;
		return -1;
	}
	for (i = 0; i < L0_MAX_SEGS && segs[i].used; i++)
		;
	if (i == L0_MAX_SEGS) {
		errno = ENOSPC;
		return -1;
	}
	void *block, *base = page_block(size, &block);
	if (!base) {
		errno = ENOMEM;
		return -1;
	}
	segs[i].key = key;
	segs[i].base = base;
	segs[i].block = block;
	segs[i].len = size;
	segs[i].nattch = 0;
	segs[i].used = 1;
	segs[i].removed = 0;
	return i + 1;
}

void *shmat(int id, const void *addr, int flag)
{
	(void)flag;
	struct l0_seg *s = seg_of(id);
	if (!s || addr) {
		errno = EINVAL;
		return (void *)-1;
	}
	s->nattch++;
	return s->base;
}

int shmdt(const void *addr)
{
	for (int i = 0; i < L0_MAX_SEGS; i++) {
		struct l0_seg *s = &segs[i];
		if (!s->used || s->base != addr)
			continue;
		if (s->nattch)
			s->nattch--;
		if (s->removed && s->nattch == 0)
			seg_drop(s);
		return 0;
	}
	errno = EINVAL;
	return -1;
}

int shmctl(int id, int cmd, struct shmid_ds *buf)
{
	struct l0_seg *s = seg_of(id);
	if (!s) {
		errno = EINVAL;
		return -1;
	}
	if (cmd == IPC_STAT) {
		if (!buf) {
			errno = EFAULT;
			return -1;
		}
		memset(buf, 0, sizeof *buf);
		buf->shm_segsz = s->len;
		buf->shm_nattch = s->nattch;
		return 0;
	}
	if (cmd == IPC_RMID) {
		s->removed = 1;
		if (s->nattch == 0)
			seg_drop(s);
		return 0;
	}
	errno = EINVAL;
	return -1;
}
