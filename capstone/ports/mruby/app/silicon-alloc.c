/* mruby's allocator for the SILICON build (B5, docs/plans/b0-silicon-delegated-runtime.md).
 *
 * Every allocation mruby makes goes through mrb_basic_alloc_func (src/allocf.c, which documents redefining it); this
 * definition is linked ahead of libmruby.a, so allocf.o is never pulled from the archive. Without it mruby's
 * allocations go to the runtime's level0 heap (ports/musl-capstone/runtime/level0.c), a first-fit list on which a
 * malloc walks every block before the free tail and a free walks the whole list to coalesce: both O(live blocks),
 * so a program holding n objects costs O(n^2). smoke.rb's M6 (20,000 arrays, each with a hash) did not finish in
 * minutes under QEMU on it, where the native build takes milliseconds.
 *
 * Small requests (up to B5_SMALL_MAX bytes) come from size classes in a static slab: 4 KiB pages, each assigned to
 * one class when first needed, and a free list per class, so allocation and release are O(1). Larger requests, and
 * any small one once the slab is exhausted, stay on level0 (mruby's heap pages, big strings and arrays: few blocks).
 *
 * Bounds: as level0's (CAPSTONE_LEVEL0_OBJECT_BOUNDS), every returned capability is narrowed to exactly the bytes
 * requested, so an overflow from one object into the next faults. The allocator never reaches a block through the
 * pointer it handed out: the block's offset is read from that pointer's cursor (a number) and the block is addressed
 * through the slab's own capability. Free lists hold offsets, not capabilities. No temporal safety: a freed block is
 * reused, as level0's are. No lock: the B5 build has one thread. */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef B5_SLAB_BYTES
#define B5_SLAB_BYTES (16u << 20)
#endif
#define B5_PAGE 4096u
#define B5_PAGES (B5_SLAB_BYTES / B5_PAGE)
#define B5_SMALL_MAX 2048u

/* Multiples of 16, so every block of a page stays capability-aligned. */
static const uint16_t class_size[] = {16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512,
                                      640, 768, 1024, 1360, 2048};
#define NCLASS (sizeof class_size / sizeof class_size[0])

static char b5_slab[B5_SLAB_BYTES] __attribute__((aligned(B5_PAGE)));
static uint8_t page_class[B5_PAGES];            /* class + 1; 0 = not yet assigned */
static uint64_t free_head[NCLASS];              /* offset + 1 of the first free block; 0 = none */
static uint32_t pages_used;
static uint8_t class_of_16[B5_SMALL_MAX / 16 + 1]; /* (size + 15) / 16 -> class */
static uint64_t n_small, n_fallback;
static int ready;

static char *b5_put(char *o, const char *s)
{
	while (*s)
		*o++ = *s++;
	return o;
}

static char *b5_num(char *o, uint64_t x)
{
	char t[24];
	int n = 0;
	do { t[n++] = (char)('0' + x % 10); x /= 10; } while (x);
	while (n)
		*o++ = t[--n];
	return o;
}

/* One line on stderr at exit, which the B5 oracle does not compare: what the slab needed. */
static void b5_report(void)
{
	char buf[128], *o = buf;
	o = b5_num(b5_put(o, "B5 alloc: pages="), pages_used);
	o = b5_num(b5_put(o, "/"), B5_PAGES);
	o = b5_num(b5_put(o, " small="), n_small);
	o = b5_num(b5_put(o, " fallback="), n_fallback);
	*o++ = '\n';
	write(2, buf, (size_t)(o - buf));
}

static void b5_init(void)
{
	unsigned c = 0;
	for (unsigned k = 0; k <= B5_SMALL_MAX / 16; k++) {
		while (class_size[c] < k * 16)
			c++;
		class_of_16[k] = (uint8_t)c;
	}
	atexit(b5_report);
	ready = 1;
}

static void *b5_narrow(uint64_t off, size_t n)
{
	char *p = b5_slab + off;
	unsigned long c = __builtin_capstone_cap_get_cursor(p);
	return __builtin_capstone_cap_shrink(p, c, c + n);
}

/* The block's offset in the slab, or -1 if p is not a slab block (a level0 one). */
static int64_t b5_offset(void *p)
{
	unsigned long off = __builtin_capstone_cap_get_cursor(p) - __builtin_capstone_cap_get_cursor(b5_slab);
	return off < (unsigned long)B5_SLAB_BYTES ? (int64_t)off : -1;
}

static int b5_refill(unsigned c)
{
	if (pages_used == B5_PAGES)
		return 0;
	uint32_t page = pages_used++;
	page_class[page] = (uint8_t)(c + 1);
	uint64_t base = (uint64_t)page * B5_PAGE, size = class_size[c];
	/* Pushed from the top down, so the page is handed out in ascending order. */
	for (uint64_t off = base + (B5_PAGE / size - 1) * size;; off -= size) {
		*(uint64_t *)(b5_slab + off) = free_head[c];
		free_head[c] = off + 1;
		if (off == base)
			break;
	}
	return 1;
}

static void *b5_alloc(size_t n)
{
	if (n > B5_SMALL_MAX)
		return malloc(n);
	unsigned c = class_of_16[(n + 15) / 16];
	if (!free_head[c] && !b5_refill(c)) {
		n_fallback++;
		return malloc(n);
	}
	uint64_t off = free_head[c] - 1;
	free_head[c] = *(uint64_t *)(b5_slab + off);
	n_small++;
	return b5_narrow(off, n ? n : 1);
}

static void b5_release(void *p, int64_t off)
{
	unsigned c = page_class[off / B5_PAGE] - 1;
	*(uint64_t *)(b5_slab + off) = free_head[c];
	free_head[c] = (uint64_t)off + 1;
}

void *mrb_basic_alloc_func(void *p, size_t size)
{
	if (!ready)
		b5_init();
	if (!p)
		return size ? b5_alloc(size) : NULL;
	int64_t off = b5_offset(p);
	if (size == 0) {
		if (off < 0)
			free(p);
		else
			b5_release(p, off);
		return NULL;
	}
	if (off < 0)
		return realloc(p, size);        /* a level0 block stays on level0 */
	size_t have = class_size[page_class[off / B5_PAGE] - 1];
	if (size <= have)
		return b5_narrow((uint64_t)off, size);
	void *q = b5_alloc(size);
	if (!q)
		return NULL;                    /* p is left as it was, as realloc does */
	memcpy(q, b5_slab + off, have);  /* the whole old block: it lies inside q's size, which is larger */
	b5_release(p, off);
	return q;
}
