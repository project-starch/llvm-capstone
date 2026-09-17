/* The domain's own allocator, because musl's cannot be the one here.
 *
 * WHY NOT musl's. lite_malloc gets memory with brk and then does
 *
 *     brk = __syscall(SYS_brk, 0);         ... a uintptr_t
 *     ...
 *     return (void *)(brk - req);          ... back to a pointer
 *
 * and on this target uintptr_t is 64 bits, so that round trip returns an
 * untagged capability. Implementing SYS_brk would not help: the allocator would
 * still hand out addresses it reconstructed from integers, and the first
 * dereference would fault. Measured 2026-09-16 as a null FILE* arriving in
 * __stdio_read after fopen, with the fault one instruction into the struct.
 *
 * That is not a musl defect. It is the ordinary assumption that a pointer is a
 * number, and it is exactly why every other port in this repository brings its
 * own level 0 rather than using the libc's: a domain's heap arrives as a
 * capability from the monitor, and only an allocator that derives from that
 * capability can hand out usable pointers. Defining malloc here means the
 * linker never pulls musl's, so there is no duplicate symbol and no ordering
 * subtlety, only the one definition.
 *
 * WHAT THIS IS. First fit over one static arena, with coalescing of adjacent
 * free blocks. Deliberately small and deliberately not clever: it exists so
 * that stdio has somewhere to put FILE buffers, and a domain that needs a
 * serious allocator should be running the one under test rather than this.
 * Every pointer it returns is derived from `arena` by pointer arithmetic, never
 * rebuilt from an integer, which is the whole property that makes it work.
 */
#include <stddef.h>
#include <stdint.h>

#ifndef CAPSTONE_LEVEL0_ARENA_BYTES
#define CAPSTONE_LEVEL0_ARENA_BYTES (256 * 1024)
#endif

/* 16, not 8: a capability is sixteen bytes and has to be stored aligned, so an
   allocation that will hold one must start on that boundary. Getting this wrong
   shows up as a tag silently not surviving a store, which is worse than a fault. */
#define L0_ALIGN 16

struct l0_block {
	size_t size;             /* payload bytes, aligned up */
	struct l0_block *next;   /* next block in address order */
	int free;
	char _pad[L0_ALIGN - (sizeof(int) % L0_ALIGN)];
};

static char l0_arena[CAPSTONE_LEVEL0_ARENA_BYTES] __attribute__((aligned(L0_ALIGN)));
static struct l0_block *l0_head;

static size_t l0_round(size_t n)
{
	return (n + (L0_ALIGN - 1)) & ~(size_t)(L0_ALIGN - 1);
}

static void l0_init(void)
{
	l0_head = (struct l0_block *)l0_arena;
	l0_head->size = CAPSTONE_LEVEL0_ARENA_BYTES - sizeof(struct l0_block);
	l0_head->next = 0;
	l0_head->free = 1;
}

void *malloc(size_t n)
{
	if (!l0_head)
		l0_init();
	if (n == 0)
		n = 1;
	size_t want = l0_round(n);

	for (struct l0_block *b = l0_head; b; b = b->next) {
		if (!b->free || b->size < want)
			continue;
		/* Split only when the tail can hold a header and something useful. */
		if (b->size >= want + sizeof(struct l0_block) + L0_ALIGN) {
			struct l0_block *tail =
			    (struct l0_block *)((char *)b + sizeof(struct l0_block) + want);
			tail->size = b->size - want - sizeof(struct l0_block);
			tail->next = b->next;
			tail->free = 1;
			b->size = want;
			b->next = tail;
		}
		b->free = 0;
		return (char *)b + sizeof(struct l0_block);
	}
	return 0;
}

void free(void *p)
{
	if (!p)
		return;
	struct l0_block *b =
	    (struct l0_block *)((char *)p - sizeof(struct l0_block));
	b->free = 1;
	/* Coalesce forward. One pass is enough because every free does it, so a
	   run of adjacent free blocks can only ever be two long at rest. */
	for (struct l0_block *c = l0_head; c; c = c->next) {
		while (c->free && c->next && c->next->free) {
			c->size += sizeof(struct l0_block) + c->next->size;
			c->next = c->next->next;
		}
	}
}

void *calloc(size_t n, size_t m)
{
	if (m && n > (size_t)-1 / m)
		return 0;
	size_t total = n * m;
	char *p = malloc(total);
	if (p)
		for (size_t i = 0; i < total; i++)
			p[i] = 0;
	return p;
}

void *realloc(void *p, size_t n)
{
	if (!p)
		return malloc(n);
	if (n == 0) {
		free(p);
		return 0;
	}
	struct l0_block *b =
	    (struct l0_block *)((char *)p - sizeof(struct l0_block));
	if (b->size >= l0_round(n))
		return p;
	char *q = malloc(n);
	if (!q)
		return 0;
	const char *s = p;
	for (size_t i = 0; i < b->size; i++)
		q[i] = s[i];
	free(p);
	return q;
}
