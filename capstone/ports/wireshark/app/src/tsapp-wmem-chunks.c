/* The wmem chunk port inside real tshark: what the ported block allocator and the port's own
 * src/allocators/sublet/chunks.c need from below, on the Sublet heap arm.
 *
 * The chunk port (ports/wireshark/wmem, patches 0001+0002) is compiled here unchanged: the same
 * wmem_allocator_block.c the replay harness tests, with this port's block size (patch 0007's one
 * macro), and the harness's chunks.c, which carves, takes, gives and revokes. This file is the
 * level below it, and replaces the harness's src/shared/backing.c:
 *
 *   wm_block_acquire   a block LINEAR from the Sublet heap (__capstone_sublet_malloc_linear): the
 *                      heap keeps its handle, the chunk port takes its own senior handle under it
 *   wm_block_release   the block must come back whole and LINEAR -- the port has revoked its senior
 *                      handle -- and goes back to the heap, whose revoke ends it
 *   wm_meta_alloc      the records and the index the headers moved into: ordinary heap objects
 *   wm_handback_probe  a pointer handed back to the allocator is read through its own authority
 *                      before its header is looked up
 *   wm_fail            one line on stderr, then exit 96: the oracle then DIFFERS, never passes
 *
 * The protected mode is switched on by a constructor, before main can make a wmem allocator.
 * Built only by host/build-domain.sh's chunks arm (TSAPP_HEAP=chunks). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chunks.h"
#include "regions.h"

unsigned long __capstone_sublet_malloc_linear(size_t n, capstone_cap_slot *out);
void __capstone_sublet_free_linear(unsigned long base);
void __capstone_sublet_heap_stats(unsigned long out[9]);

_Noreturn void wm_fail(unsigned code)
{
	fprintf(stderr, "TSAPP-WMEM fail %u\n", code);
	fflush(stderr);
	_Exit(96);
}

/* The heap lends at most 16 blocks of 1 MiB from its 16 MiB pool, and fewer in practice. */
#define TSAPP_WMEM_BLOCKS 64
static struct {
	size_t base, size;
} blocks[TSAPP_WMEM_BLOCKS];

size_t wm_block_acquire(size_t n, capstone_cap_slot *out)
{
	unsigned long base = __capstone_sublet_malloc_linear(n, out);

	if (!base)
		wm_fail(206);
	for (unsigned i = 0; i < TSAPP_WMEM_BLOCKS; ++i)
		if (!blocks[i].base) {
			blocks[i].base = base;
			blocks[i].size = n;
			return base;
		}
	wm_fail(207);
}

void wm_block_release(size_t base, capstone_cap_slot *in)
{
	for (unsigned i = 0; i < TSAPP_WMEM_BLOCKS; ++i) {
		if (blocks[i].base != base)
			continue;
		if (capstone_cap_type(in) != CAPSTONE_CAP_LINEAR || capstone_cap_base(in) != base ||
		    capstone_cap_end(in) < base + blocks[i].size)
			wm_fail(213);
		/* The heap's revoke of its own handle ends this region with the rest of the block. */
		capstone_cap_clear(in);
		__capstone_sublet_free_linear(base);
		blocks[i].base = 0;
		return;
	}
	wm_fail(211);
}

void *wm_meta_alloc(size_t n)
{
	void *p = calloc(1, n);

	if (!p)
		wm_fail(212);
	return p;
}

void wm_handback_probe(const void *p)
{
	(void)*(const volatile unsigned char *)p;
}

/* The spatial mode's system blocks: never used here, since the constructor turns the protected
 * mode on, but chunks.c links against them. */
void *wm_sys_alloc(size_t n)
{
	void *p = malloc(n);

	if (!p)
		wm_fail(208);
	return p;
}

void wm_sys_free(void *p)
{
	free(p);
}

/* chunks.c reports a second translation unit's counters beside its own. Here that unit is the
 * heap: its revokes include every jumbo chunk's free, which wmem sends to g_free. */
void wm_region_counts(uint64_t *revokes, uint64_t *inits)
{
	unsigned long st[9];

	__capstone_sublet_heap_stats(st);
	*revokes = st[7];
	*inits = st[8];
}

__attribute__((constructor)) static void tsapp_wmem_chunks_on(void)
{
	wm_chunks_init(1);
}

/* For the exit hook (src/tsapp-heap.c): the chunk port's counts, appended to the TSAPP-HEAP
 * line. nodes= is the chunk layer's revocation-node spend, split + mrev, which chunks.c's
 * structure fixes: an epoch (open or reset) is one handle and one carve, a chunk split one split,
 * an issue one take. */
int tsapp_wmem_report(char *line, size_t cap)
{
	static unsigned char page[128 + sizeof(struct wm_chunk_counts)];
	struct wm_chunk_counts c;

	wm_chunk_report(page);
	memcpy(&c, page + 128, sizeof c);
	if (c.magic != WM_CHUNK_COUNTS_MAGIC)
		return snprintf(line, cap, " wmem=absent");
	return snprintf(line, cap,
	                " wmem opens=%llu resets=%llu closes=%llu reset_revokes=%llu close_revokes=%llu"
	                " dropped=%llu retires=%llu splits=%llu issues=%llu revokes=%llu inits=%llu"
	                " nodes=%llu",
	                (unsigned long long)c.opens, (unsigned long long)c.resets,
	                (unsigned long long)c.closes, (unsigned long long)c.reset_revokes,
	                (unsigned long long)c.close_revokes, (unsigned long long)c.dropped,
	                (unsigned long long)c.retires, (unsigned long long)c.splits,
	                (unsigned long long)c.issues, (unsigned long long)c.revokes,
	                (unsigned long long)c.inits,
	                (unsigned long long)(2 * (c.opens + c.resets) + c.splits + c.issues));
}
