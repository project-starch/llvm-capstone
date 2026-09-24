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
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

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

/* CAPSTONE_LEVEL0_SHRINK: per-object heap bounds, opt-in. Without it every pointer this
 * allocator returns carries the bounds of the WHOLE ARENA, so an overflow from one object
 * into the next is not a fault -- which is the default, and which is what every port built
 * on this file has had. With it, malloc narrows the returned capability to exactly the n
 * bytes asked for (the rv8 allocators' shrink, benchmarks/rv8/adapted/rv8_malloc.c).
 *
 * Two things follow, and both are the reason this is a macro and not a one-line change:
 *  - the header sits BELOW the payload, outside a narrowed pointer, so free and realloc
 *    recover it through the arena capability from the pointer's ADDRESS (read as a number,
 *    never turned back into a pointer), not through the pointer;
 *  - realloc may copy only what the old pointer's bounds cover, and must re-narrow the
 *    pointer it returns in place, or a block grown within its slack keeps the old bounds.
 * What it does NOT give: temporal safety. free still only marks the block free, so a stale
 * pointer keeps working and reads whatever occupies the memory next. Bounds are exact in a
 * register. On silicon, a narrowed capability of 4 KiB or more that is stored and reloaded is
 * rounded outward to its representable granule (the RTL's encoder), since block bases here are
 * only 16-aligned; capstone-qemu keeps full precision for stored capabilities (cap_mem_map.h)
 * and does not show that. */
#if defined(CAPSTONE_LEVEL0_SHRINK) && CAPSTONE_LEVEL0_SHRINK
static void *l0_narrow(void *p, size_t n)
{
	unsigned long c = __builtin_capstone_cap_get_cursor(p);
	return __builtin_capstone_cap_shrink(p, c, c + n);
}
static struct l0_block *l0_header(void *p)
{
	unsigned long off = __builtin_capstone_cap_get_cursor(p) -
	                    __builtin_capstone_cap_get_cursor(l0_arena);
	return (struct l0_block *)(l0_arena + off - sizeof(struct l0_block));
}
static size_t l0_readable(void *p, size_t block)
{
	unsigned long have = __builtin_capstone_cap_get_end(p) - __builtin_capstone_cap_get_cursor(p);
	return have < block ? have : block;
}
#define L0_RETURN(p, n) l0_narrow((p), (n))
#define L0_REALLOC_IN_PLACE(p, b, n) l0_narrow((char *)(b) + sizeof(struct l0_block), (n))
#else
static struct l0_block *l0_header(void *p)
{
	return (struct l0_block *)((char *)p - sizeof(struct l0_block));
}
#define L0_RETURN(p, n) (p)
#define L0_REALLOC_IN_PLACE(p, b, n) (p)
#define l0_readable(p, block) (block)
#endif

/* CAPSTONE_LEVEL0_STATS: what a port needs to size CAPSTONE_LEVEL0_ARENA_BYTES, opt-in and
 * compiled out otherwise. Bytes are whole blocks, header included, since that is what the
 * arena gives up. `peak_end` is the furthest a live block has ever reached into the arena:
 * first fit leaves holes, so it is the arena size a run needed, and at least `peak_in_use`. */
#ifdef CAPSTONE_LEVEL0_STATS
static size_t l0_in_use, l0_peak_in_use, l0_peak_end;
static void l0_note_alloc(struct l0_block *b)
{
	size_t end = (size_t)((char *)b - l0_arena) + sizeof(struct l0_block) + b->size;
	l0_in_use += sizeof(struct l0_block) + b->size;
	if (l0_in_use > l0_peak_in_use)
		l0_peak_in_use = l0_in_use;
	if (end > l0_peak_end)
		l0_peak_end = end;
}
size_t __capstone_level0_in_use(void) { return l0_in_use; }
size_t __capstone_level0_peak_in_use(void) { return l0_peak_in_use; }
size_t __capstone_level0_peak_end(void) { return l0_peak_end; }
size_t __capstone_level0_arena_bytes(void) { return CAPSTONE_LEVEL0_ARENA_BYTES; }
#define L0_NOTE_ALLOC(b) l0_note_alloc(b)
#define L0_NOTE_FREE(b) (l0_in_use -= sizeof(struct l0_block) + (b)->size)
#else
#define L0_NOTE_ALLOC(b) ((void)0)
#define L0_NOTE_FREE(b) ((void)0)
#endif

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
		L0_NOTE_ALLOC(b);
		return L0_RETURN((char *)b + sizeof(struct l0_block), n);
	}
	/* POSIX: a failed allocation sets errno. libc-test's search_hsearch is
	   what asked: hcreate((size_t)-1) must fail with ENOMEM, and musl's
	   hcreate reports whatever calloc left in errno, which was nothing. */
	errno = ENOMEM;
	return 0;
}

void free(void *p)
{
	if (!p)
		return;
	struct l0_block *b = l0_header(p);
	L0_NOTE_FREE(b);
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
	if (m && n > (size_t)-1 / m) {
		errno = ENOMEM;
		return 0;
	}
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
	struct l0_block *b = l0_header(p);
	if (b->size >= l0_round(n))
		return L0_REALLOC_IN_PLACE(p, b, n);
	char *q = malloc(n);
	if (!q)
		return 0;
	/* memmove, not a byte loop: a byte loop drops the tag of every pointer
	   stored in the block, and the whole point of moving a block is that its
	   contents keep meaning what they meant. See string_bounds_safe.c. */
	memmove(q, p, l0_readable(p, b->size));
	free(p);
	return q;
}

/* musl calls its own allocator by five names, not one. The public malloc is a
   weak alias, so defining it is not enough: everything under src/locale,
   src/time/__tz.c and src/stdio/ofl_add.c calls __libc_malloc or __libc_calloc
   directly, which in a build without mallocng resolve to lite_malloc.c's
   __simple_malloc. That function keeps its heap in uintptr_t and turns integers
   back into pointers, so the first call traps: libc-test's mbc and swprintf
   both died on `movc` of a zero cur pointer inside it. Defining the internal
   names here means one allocator answers to all of them and lite_malloc.c is
   never pulled out of the archive at all. */
void *__libc_malloc(size_t n) { return malloc(n); }
void *__libc_malloc_impl(size_t n) { return malloc(n); }
void *__libc_calloc(size_t n, size_t m) { return calloc(n, m); }
void *__libc_realloc(void *p, size_t n) { return realloc(p, n); }
void __libc_free(void *p) { free(p); }
