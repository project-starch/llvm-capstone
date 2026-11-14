#include <errno.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/syscall.h>
#include "vm.h"
__attribute__((__weak__)) void __vm_wait(void) {}
static int mapping_length(size_t n, unsigned long *bytes)
{
    if (!n || n > CV_MAX_REGION_BYTES) { errno = EINVAL; return -1; }
    *bytes = (n + 4095) & ~4095UL;
    return 0;
}
void *mmap(void *addr, size_t n, int prot, int flags, int fd, off_t off)
{
    int kind = flags & ~MAP_FIXED;
    if (flags & MAP_HUGETLB) { errno = ENOMEM; return MAP_FAILED; }
    /* The existing MAP_SHARED compatibility remains process-local. */
    if ((prot & ~(PROT_READ | PROT_WRITE | PROT_EXEC)) ||
        (kind != (MAP_PRIVATE | MAP_ANONYMOUS) && kind != (MAP_SHARED | MAP_ANONYMOUS)) ||
        fd != -1 || off) {
        errno = EINVAL; return MAP_FAILED;
    }
    unsigned long bytes;
    if (flags & MAP_FIXED) {
        /* Stage 4e: MAP_FIXED only inside a mapping the program holds, through a reference of that
         * mapping (not of an object inside it) that covers the range, with its cursor at the first
         * page: the kernel gives the range zero pages with prot (removed pages included), and only
         * while the program derived no object below that reference (the refill rule; with root
         * grants, EOPNOTSUPP without). The reference
         * comes back, as Linux returns the address. MAP_FIXED by a plain address, outside every
         * mapping or across two is refused (EINVAL). */
        if (!addr || (__builtin_capstone_cap_get_cursor(addr) & 4095)) {
            errno = EINVAL; return MAP_FAILED;
        }
        if (mapping_length(n, &bytes)) return MAP_FAILED;
        long rc = __capstone_vm_refill(addr, bytes, prot);
        if (rc) { errno = -rc; return MAP_FAILED; }
        return addr;
    }
    if (addr) { errno = EINVAL; return MAP_FAILED; }
    if (mapping_length(n, &bytes)) return MAP_FAILED;
    sublet_cap slot;
    /* The page table enforces the protection, which mprotect changes later; the reference carries no
     * rights (block A). ENOMEM also when the lifetime table has no slot left and cannot grow. */
    if (cap_vm_acquire(&slot, bytes, 4096, prot, CAP_VM_MAPPING))
        return MAP_FAILED;
    void *p = cap_vm_copyable(&slot);
    unsigned long base = __builtin_capstone_cap_get_cursor(p);
    return __builtin_capstone_cap_shrink(p, base, base + bytes);
}
void *__mmap(void *a, size_t n, int p, int f, int fd, off_t off)
{ return mmap(a, n, p, f, fd, off); }
int munmap(void *p, size_t n)
{
    unsigned long bytes, address = __builtin_capstone_cap_get_cursor(p);
    if ((address & 4095) || mapping_length(n, &bytes)) {
        errno = EINVAL; return -1;
    }
    /* VM operations must work on PROT_NONE, without probing its contents.
     * The trusted service resolves the range against its mapping registry,
     * and with root_grants the reference p is the authority for the unmap. */
    long rc = __capstone_vm_unmap(p, bytes);
    if (rc) { errno = -rc; return -1; }
    return 0;
}
int __munmap(void *p, size_t n) { return munmap(p, n); }
int mprotect(void *p, size_t n, int prot)
{
    unsigned long bytes, address = __builtin_capstone_cap_get_cursor(p);
    if ((address & 4095) || (prot & ~(PROT_READ | PROT_WRITE | PROT_EXEC))) {
        errno = EINVAL; return -1;
    }
    if (!n) return 0;
    if (mapping_length(n, &bytes)) return -1;
    /* p goes to the service as a reference, as for munmap: with root_grants the kernel changes
     * the protection only if p is a live reference to the mapping that covers [p, p + bytes). */
    long rc = __capstone_vm_protect(p, bytes, prot);
    if (rc) { errno = -rc; return -1; }
    return 0;
}
/* musl's pthread stack setup calls the hidden symbol directly. */
int __mprotect(void *p, size_t n, int prot) { return mprotect(p, n, prot); }
/* madvise(MADV_DONTNEED) and MADV_FREE destroy the contents of the range, so p goes to the service as
 * a reference, as for mprotect: with root_grants the kernel acts only if p is a live reference to the
 * mapping that covers [p, p + bytes). The range reads zero afterwards. Every other advice answers
 * ENOSYS and is reported as unserved, as before this service existed. */
#ifndef MADV_DONTNEED
#define MADV_DONTNEED 4    /* Linux's values; <sys/mman.h> declares them only under _BSD_SOURCE */
#define MADV_FREE 8
#endif
void __capstone_hc_note_unserved(long n);
int madvise(void *p, size_t n, int advice)
{
    unsigned long bytes, address = __builtin_capstone_cap_get_cursor(p);
    if (advice != MADV_DONTNEED && advice != MADV_FREE) {
        __capstone_hc_note_unserved(SYS_madvise);
        errno = ENOSYS; return -1;
    }
    if (address & 4095) { errno = EINVAL; return -1; }
    if (!n) return 0;
    if (mapping_length(n, &bytes)) return -1;
    long rc = __capstone_vm_advise(p, bytes, advice);
    if (rc) { errno = -rc; return -1; }
    return 0;
}
int __madvise(void *p, size_t n, int advice) { return madvise(p, n, advice); }
/* Public mremap (2026-10-22: cpython's mmap.resize, which corpus cell lt0 18 needs before its bug
 * runs; stage 4e: in place where it can; stage 4f: one validator first). p is the block's own
 * reference with its cursor at the first page of the old range [p, p + old), any page range of one
 * block. Every path validates that old range before it changes anything, the same way: the range lies
 * inside one block, every page of it is logically mapped (EFAULT otherwise, as Linux answers for an
 * unmapped part of the old range), and with root grants p is a reference of the block (not of an
 * object inside it) that covers the old range (EINVAL otherwise).
 *   - Same page count: the mapping check (CV_SERVICE_MAPPING_CHECK) and p back, as Linux returns the
 *     same address.
 *   - Shrink or growth in place: one service (CV_SERVICE_RESIZE) validates the old range and then, in
 *     the same step, removes the tail [p + n, p + old) as a partial munmap does (the block keeps the
 *     pages of the new length, so the shrink never ends it, and p keeps working), or refills the new
 *     tail [p + old, p + n) when every page of it is a removed page inside the block's original extent
 *     (its visible length) that p covers and no object is derived below p (the refill rule of
 *     MAP_FIXED): those
 *     pages read zero afterwards with the protection of the range's last page, and p comes back.
 *     Where growth in place is not possible the kernel answers ENOMEM; without MREMAP_MAYMOVE that is
 *     the answer (as Linux answers when there is no room), with it the range moves.
 *   - Move: the mapping check with CV_CHECK_MOVE (the validator, and every page read-write: ENOTSUP
 *     otherwise; nothing changed, never a fault in the copy), then a new read-write private anonymous
 *     mapping, a copy of the old range's bytes with their tags (memcpy), and a munmap of [p, p + old)
 *     through p, which ends the block with CREVOKE on p if that range held every page still mapped
 *     (every capability to it is then revoked, the lifetime contract a moved mapping must keep: an
 *     address of the old block goes to another mapping only after the block has ended) and removes
 *     those pages otherwise.
 * The move's check states what holds when it answers, not what holds at the copy: another thread of
 * the program may end the old mapping or change its protection between the check and the copy, and
 * the copy then faults instead of answering EINVAL. That affects only the program itself, as a
 * concurrent free does for realloc. The copy reads at most the old range, which p covers, and writes
 * at most the new mapping, which mmap's reference covers. Without root grants the address is the
 * authority, as for munmap, so a reference that does not cover the range faults in the copy, and
 * growth in place is never possible (the refill rule needs the reference): it moves.
 * MREMAP_FIXED and MREMAP_DONTUNMAP stay ENOSYS. The copy is not atomic with respect to the program's
 * other threads: a write to the old range during the copy may be lost, as for realloc. If the munmap
 * of the old range fails after the copy (another thread changed it in between), the new mapping is
 * unmapped again and the error returned; that unmap of a fresh mapping cannot fail, and a failure
 * traps. */
#ifndef MREMAP_MAYMOVE
#define MREMAP_MAYMOVE 1   /* Linux's value; <sys/mman.h> declares it only under _GNU_SOURCE */
#endif
void *mremap(void *p, size_t old, size_t n, int flags, ...)
{
    unsigned long old_bytes, new_bytes, address = __builtin_capstone_cap_get_cursor(p);
    if (flags & ~MREMAP_MAYMOVE) { errno = ENOSYS; return MAP_FAILED; }
    if ((address & 4095) || mapping_length(old, &old_bytes) || mapping_length(n, &new_bytes)) {
        errno = EINVAL; return MAP_FAILED;
    }
    long rc;
    if (new_bytes == old_bytes) {
        rc = __capstone_vm_mapping_check(p, old_bytes, 0);
        if (rc) { errno = -rc; return MAP_FAILED; }
        return p;
    }
    rc = __capstone_vm_resize(p, old_bytes, new_bytes);
    if (!rc) return p;
    if (new_bytes < old_bytes || rc != -ENOMEM || !(flags & MREMAP_MAYMOVE)) {
        errno = -rc; return MAP_FAILED;
    }

    rc = __capstone_vm_mapping_check(p, old_bytes, CV_CHECK_MOVE);
    if (rc) { errno = -rc; return MAP_FAILED; }
    void *q = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (q == MAP_FAILED) return MAP_FAILED;
    memcpy(q, p, old_bytes);

    if (munmap(p, old)) {
        int error = errno;
        if (munmap(q, n)) __builtin_trap();
        errno = error;
        return MAP_FAILED;
    }
    return q;
}

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
