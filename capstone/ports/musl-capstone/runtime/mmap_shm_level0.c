/* mmap, munmap and the System V shared-memory calls for a domain, served from
 * the domain's own allocator (level0.c).
 *
 * WHY AN OVERRIDE AND NOT A CASE IN __capstone_hostcall. musl's mmap() and
 * shmat() return the syscall's long as a pointer:
 *
 *     return (void *)__syscall_ret(ret);            src/mman/mmap.c
 *     return (void *)syscall(SYS_shmat, id, addr, flag);   src/ipc/shmat.c
 *
 * A long holds an address, not a capability, so a mapping served through the
 * syscall layer would arrive untagged and fault on its first use -- the round
 * trip that keeps level0.c from using musl's malloc in the first place. So the
 * functions themselves are replaced, the way atexit_capability_safe.c replaces
 * atexit, and every pointer they return is derived from level0's arena by
 * pointer arithmetic, never rebuilt from an integer.
 *
 * WHAT IS SERVED. A domain is one process in one address space, so the
 * distinctions the kernel keeps do not arise: MAP_SHARED and MAP_PRIVATE are
 * the same memory, a System V segment is a block with an id, attaching is
 * handing out its pointer. Served: anonymous mappings (zeroed, page-aligned),
 * munmap of a whole mapping, and shmget/shmat/shmdt/shmctl with IPC_PRIVATE or
 * a key, IPC_CREAT and IPC_EXCL, IPC_STAT and IPC_RMID (a removed segment goes
 * when its last attachment does, as on Linux). Refused, each with the errno a
 * kernel gives for the nearest condition: a file mapping (ENODEV: there is no
 * file service behind mmap), MAP_FIXED and an shmat address (EINVAL),
 * MAP_HUGETLB (ENOMEM, which is what PostgreSQL expects before it retries
 * without), a partial munmap (EINVAL), any other shmctl command (EINVAL).
 * The memory is level0's, so a domain that maps megabytes sizes
 * CAPSTONE_LEVEL0_ARENA_BYTES for them, as every port already does for malloc.
 *
 * First consumer is PostgreSQL's single-user backend: one MAP_SHARED|MAP_ANONYMOUS
 * mapping for its shared memory and one small System V segment as its
 * data-directory interlock (capstone/ports/postgres/single-user/).
 *
 * __mmap and __munmap are defined too: musl's own objects call those names, and
 * a reference to one would otherwise pull musl's mmap.o in beside this file.
 */
#define _GNU_SOURCE /* MAP_ANONYMOUS, MAP_HUGETLB */
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>

#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif

#define L0_PAGE 4096UL
#define L0_MAX_MAPS 32
#define L0_MAX_SEGS 16

struct l0_map {
	void *base;   /* what the caller holds: page-aligned */
	void *block;  /* what free() takes back */
	size_t len;   /* whole pages */
};
static struct l0_map maps[L0_MAX_MAPS];

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

void *__mmap(void *start, size_t len, int prot, int flags, int fd, off_t off)
{
	(void)start; (void)prot; (void)fd; (void)off;
	if (len == 0 || (flags & MAP_FIXED)) {
		errno = EINVAL;
		return MAP_FAILED;
	}
	if (!(flags & MAP_ANONYMOUS)) {
		errno = ENODEV;
		return MAP_FAILED;
	}
	if (flags & MAP_HUGETLB) {
		errno = ENOMEM;
		return MAP_FAILED;
	}
	int i;
	for (i = 0; i < L0_MAX_MAPS && maps[i].base; i++)
		;
	if (i == L0_MAX_MAPS) {
		errno = ENOMEM;
		return MAP_FAILED;
	}
	void *block, *base = page_block(len, &block);
	if (!base) {
		errno = ENOMEM;
		return MAP_FAILED;
	}
	maps[i].base = base;
	maps[i].block = block;
	maps[i].len = pages(len);
	return base;
}

void *mmap(void *start, size_t len, int prot, int flags, int fd, off_t off)
{
	return __mmap(start, len, prot, flags, fd, off);
}

int __munmap(void *start, size_t len)
{
	for (int i = 0; i < L0_MAX_MAPS; i++) {
		if (!maps[i].base || maps[i].base != start)
			continue;
		if (pages(len) != maps[i].len) {
			errno = EINVAL; /* a partial unmap has no service here */
			return -1;
		}
		free(maps[i].block);
		maps[i].base = maps[i].block = 0;
		maps[i].len = 0;
		return 0;
	}
	errno = EINVAL;
	return -1;
}

int munmap(void *start, size_t len)
{
	return __munmap(start, len);
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
