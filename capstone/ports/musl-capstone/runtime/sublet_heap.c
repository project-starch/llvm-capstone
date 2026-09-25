/* A domain heap with per-object bounds AND revocation on free: the level0 replacement for a
 * program whose safety, not only its correctness, is being measured.
 *
 * WHY level0 IS NOT ENOUGH. level0 (level0.c) hands out pointers carrying the whole arena's
 * bounds and frees by marking a block free, so an overflow into the next object and a use
 * after free both simply work. Measured on the FFmpeg app port, 2026-09-23
 * (ports/ffmpeg/app/host/safety-expect.txt).
 *
 * WHAT THIS IS. A binary buddy allocator over one LINEAR region the host transfers to the
 * domain (hostcall.c, CAPSTONE_PROGRAM_REGIONS), following the Sublet recipe exactly as the
 * SQLite memsys5 patch applies it (ports/sqlite/sublet/sublet-3530300.patch):
 *   - every block has its own revocation node: split with sublet_split, and a handle senior
 *     to both halves taken first (sublet_handle), so that one revoke merges them again;
 *   - malloc checks a block out with sublet_take: the slot keeps the handle, the caller gets
 *     the alias; free is sublet_give: one revoke, after which every copy of the alias is dead.
 * Plus two steps memsys5 does not take: the alias is SHRUNK to the requested size, so the
 * bounds are the object's and not the power-of-two block's; and free SCRUBS the object through
 * that alias before revoking it. A block reissued without a merge is not written through by the
 * revoke, so without the scrub the next owner would read the last owner's bytes, capabilities
 * included (audit, 2026-09-23).
 *
 * WHY A BUDDY and not first fit: on silicon a stored capability is compressed (the RTL's
 * encoder, ariane_pkg.sv; the kernel's capstone_repr_granule), and a region whose base or end is
 * not a multiple of its length's granule comes back WIDER than it was. A buddy block is a power
 * of two aligned to itself, so every block and every remainder is representable by construction.
 * First fit over 16-byte offsets would widen the remainder over the object just carved from it.
 * capstone-qemu does NOT show this: it keeps full-precision bounds for stored capabilities in a
 * side table (cap_mem_map.h, op_helper.c) -- so on the emulator the rounding in sh_narrow is
 * the only rounding there is, and a first-fit version would not have failed there.
 *
 * THE STALE FREE. free() reads one byte through the pointer it is given BEFORE anything is
 * revoked. On capstone-qemu a pointer to a freed block reloads untagged (ISSUES Q-11), so that
 * read faults, in free, and the stale free never reaches a revoke of somebody else's handle (the
 * nginx port's note: "a stale free is caught when the allocator touches the object"). It also
 * keeps the emulator alive: capstone-qemu without the revoke-raises fix ASSERTS on a revoke of an
 * untagged operand. ON SILICON THE PROBE DOES NOT PROTECT: the RTL forwards the stale capability
 * tagged and a data load through it retires (measurements §7r), so a stale free there would
 * revoke the new owner's handle. A silicon-effective check is the LCC validity query (selector
 * 0), which the RTL answers from the revocation node and capstone-qemu stubs to "valid" (Q-11);
 * it would need both.
 *
 * THE POOL is the largest power-of-two block, at most 2^CAPSTONE_SUBLET_HEAP_LOG, that is
 * aligned to its own size and lies inside the grant. A buddy region from the kernel is aligned
 * to its size; a CMA region only to CONFIG_CMA_ALIGNMENT (1 MiB), so a 4 MiB grant may yield a
 * 2 MiB pool (audit, 2026-09-23). What lies outside the pool is split off and dropped.
 *
 * WHAT IT COSTS. Tables beside the pool, in .bss (one capability slot per atom and one per
 * split): with the defaults, 4 MiB pool and 256-byte atoms, about 0.6 MiB. A merge writes the
 * merged block through before init (sublet.h, sublet_give_to), so freeing is O(block) when it
 * coalesces, and O(object) always, for the scrub. Revocation nodes: one per split and one per
 * allocation, 65,532 per boot. The resident bitstream (054cea69b) reclaims the nodes a revoke
 * walk invalidates but never a handle's own node, so how far that ceiling moves depends on the
 * workload's leak fraction (ISSUES R-12; bitstreams before 2026-09-17 reclaimed nothing). The
 * counts are exported (__capstone_sublet_heap_stats) for a program to report.
 *
 * Linear capabilities never sit in C variables here, only in sublet_cap slots, except the
 * grant between the one call that hands it over and the store that parks it.
 */
#include <errno.h>
#include <stddef.h>
#include <string.h>
#include <unistd.h>

#include "sublet.h"

#ifndef CAPSTONE_SUBLET_HEAP_LOG
#define CAPSTONE_SUBLET_HEAP_LOG 22    /* the pool: 4 MiB, one naturally aligned buddy block */
#endif
#ifndef CAPSTONE_SUBLET_ATOM_LOG
#define CAPSTONE_SUBLET_ATOM_LOG 8     /* the smallest block: 256 bytes */
#endif

#define SH_POOL   (1UL << CAPSTONE_SUBLET_HEAP_LOG)
#define SH_MAXORD (CAPSTONE_SUBLET_HEAP_LOG - CAPSTONE_SUBLET_ATOM_LOG)
#define SH_N      (1u << SH_MAXORD)
#define SH_NIL    0xFFFFFFFFu

#define SH_FREE 0x80u   /* sh_ctrl at a block's first atom: free, on its order's list */
#define SH_OUT  0x40u   /*                                  checked out to the program */
#define SH_ORD  0x3Fu

/* Handed over by hostcall.c: the first region the host shares after the two HostCall v0
   regions, or null if it shared none. Taken once. */
void *__capstone_region(unsigned index);

static sublet_cap sh_grant;
static sublet_cap sh_cap[SH_N];   /* block at atom i: the region while free, its handle while out */
static sublet_cap sh_par[SH_N];   /* handle senior to the halves of split block (i, k), at sh_paroff[k] + (i >> k) */
static unsigned char sh_ctrl[SH_N];
static unsigned sh_next[SH_N], sh_prev[SH_N];
static unsigned sh_head[SH_MAXORD + 1];
static unsigned sh_paroff[SH_MAXORD + 2];
static unsigned long sh_base;
static unsigned sh_maxord;        /* the pool's order: SH_MAXORD, or less for a smaller aligned block */
static int sh_state;              /* 0 not yet, 1 ready, -1 no usable grant */

/* counts a program can report: allocations, frees, merges, and the Sublet primitives */
static unsigned long sh_n_alloc, sh_n_free, sh_n_merge, sh_live, sh_peak_live;

static void sh_say(const char *msg)
{
	write(2, msg, strlen(msg));
}

static unsigned sh_paridx(unsigned i, unsigned k) { return sh_paroff[k] + (i >> k); }

static void sh_push(unsigned i, unsigned k)
{
	sh_ctrl[i] = SH_FREE | k;
	sh_prev[i] = SH_NIL;
	sh_next[i] = sh_head[k];
	if (sh_head[k] != SH_NIL)
		sh_prev[sh_head[k]] = i;
	sh_head[k] = i;
}

static void sh_unlink(unsigned i, unsigned k)
{
	if (sh_prev[i] != SH_NIL)
		sh_next[sh_prev[i]] = sh_next[i];
	else
		sh_head[k] = sh_next[i];
	if (sh_next[i] != SH_NIL)
		sh_prev[sh_next[i]] = sh_prev[i];
	sh_ctrl[i] = 0;
}

static void sh_init(void)
{
	sh_state = -1;
	for (unsigned k = 0; k <= SH_MAXORD; k++)
		sh_head[k] = SH_NIL;
	sh_paroff[0] = sh_paroff[1] = 0;
	for (unsigned k = 1; k <= SH_MAXORD; k++)
		sh_paroff[k + 1] = sh_paroff[k] + (SH_N >> k);

	/* Stored first and only then examined: a linear capability moves when it is loaded on
	   hardware that enforces linearity (ISSUES Q-12), so a null test on a C variable before
	   the store could leave the slot empty at -O0. An absent grant reads as type NONE. */
	sublet_store(&sh_grant, __capstone_region(0));
	unsigned long t = sublet_type(&sh_grant);
	if (t == SUBLET_TYPE_NONE) {
		sh_say("sublet-heap: no region was granted; every allocation fails\n");
		return;
	}
	if (t != SUBLET_TYPE_LIN) {
		sh_say("sublet-heap: the granted region is not LINEAR (share it REV_TRANSFERRED); every allocation fails\n");
		return;
	}
	unsigned long base = sublet_base(&sh_grant), end = sublet_end(&sh_grant);
	/* the largest self-aligned power-of-two block inside [base, end), at most the table size */
	unsigned ord = SH_MAXORD;
	unsigned long pool, abase;
	for (;;) {
		pool = 1UL << (ord + CAPSTONE_SUBLET_ATOM_LOG);
		abase = (base + pool - 1) & ~(pool - 1);
		if (abase >= base && abase + pool <= end && abase + pool > abase)
			break;
		if (ord == 0) {
			sh_say("sublet-heap: the granted region holds no aligned block; every allocation fails\n");
			return;
		}
		ord--;
	}
	sublet_cap rest;
	if (abase > base) {               /* the head below the aligned pool dies here */
		sublet_split(&sh_grant, abase, &rest);
		sublet_clear(&sh_grant);
		sublet_move(&rest, &sh_grant);
	}
	if (abase + pool < end) {         /* and so does the tail above it */
		sublet_split(&sh_grant, abase + pool, &rest);
		sublet_clear(&rest);
	}
	sh_base = abase;
	sh_maxord = ord;
	sublet_move(&sh_grant, &sh_cap[0]);
	sh_push(0, ord);
	sh_state = 1;
	if (ord < SH_MAXORD)
		sh_say("sublet-heap: the grant is not aligned to the full pool; using a smaller one\n");
}

/* Exactly n bytes when that is representable; above 4 KiB, n rounded up to the granule the
   compressed encoding keeps (capstone.c's capstone_repr_granule). The block is a power of two
   at least as long and aligned to itself, so the rounding never reaches another object. */
static void *sh_narrow(void *alias, size_t n)
{
	unsigned long c = __builtin_capstone_cap_get_cursor(alias);
	unsigned long len = n;
	if (len >= 4096) {
		unsigned lg = 63 - __builtin_clzl(len);
		unsigned long g = 1UL << (lg - 9);
		len = (len + g - 1) & ~(g - 1);
	}
	return __builtin_capstone_cap_shrink(alias, c, c + len);
}

/* Carve a block of at least n bytes and mark it handed out; the atom index lands in *idx.
   Returns 0, or -1 with errno set. This is malloc's body up to the hand-out: the linear lend
   below carves IDENTICALLY and differs only in how the block leaves, so both arms share one
   buddy policy and one set of counters. */
static int sh_carve_block(size_t n, unsigned *idx)
{
	if (sh_state == 0)
		sh_init();
	if (sh_state < 0 || n > (1UL << (sh_maxord + CAPSTONE_SUBLET_ATOM_LOG))) {
		errno = ENOMEM;
		return -1;
	}
	if (n == 0)
		n = 1;
	unsigned k = 0;
	while (((size_t)1 << (k + CAPSTONE_SUBLET_ATOM_LOG)) < n)
		k++;
	unsigned j = k;
	while (j <= sh_maxord && sh_head[j] == SH_NIL)
		j++;
	if (j > sh_maxord) {
		errno = ENOMEM;
		return -1;
	}
	unsigned i = sh_head[j];
	sh_unlink(i, j);
	while (j > k) {
		unsigned half = 1u << (j - 1);
		/* the handle senior to both halves, then the split: the lower half keeps the
		   block's node, the upper half gets a fresh one */
		sublet_handle(&sh_cap[i], &sh_par[sh_paridx(i, j)]);
		sublet_split(&sh_cap[i], sh_base + ((unsigned long)(i + half) << CAPSTONE_SUBLET_ATOM_LOG),
		             &sh_cap[i + half]);
		j--;
		sh_push(i + half, j);
	}
	sh_ctrl[i] = SH_OUT | k;
	sh_n_alloc++;
	if (++sh_live > sh_peak_live)
		sh_peak_live = sh_live;
	*idx = i;
	return 0;
}

void *malloc(size_t n)
{
	unsigned i;
	if (sh_carve_block(n, &i) < 0)
		return 0;
	return sh_narrow(sublet_take(&sh_cap[i]), n);
}

void free(void *p)
{
	if (!p)
		return;
	/* The stale-pointer probe. A freed object's alias is revoked, so this read faults here,
	   before anything below can revoke a handle that now belongs to someone else. */
	(void)*(volatile const char *)p;
	unsigned long a = __builtin_capstone_cap_get_cursor(p);
	unsigned i;
	if (sh_state <= 0 || a < sh_base || a - sh_base >= (1UL << (sh_maxord + CAPSTONE_SUBLET_ATOM_LOG)) ||
	    ((a - sh_base) & ((1UL << CAPSTONE_SUBLET_ATOM_LOG) - 1)) ||
	    !(sh_ctrl[i = (unsigned)((a - sh_base) >> CAPSTONE_SUBLET_ATOM_LOG)] & SH_OUT)) {
		sh_say("sublet-heap: free of a pointer this heap did not hand out; ignored\n");
		return;
	}
	unsigned k = sh_ctrl[i] & SH_ORD;
	/* the scrub, through the object's own alias and within its own bounds: the next owner of
	   this block must not find the last owner's bytes or capabilities in it */
	memset(p, 0, __builtin_capstone_cap_get_end(p) - a);
	sublet_give(&sh_cap[i]);          /* one revoke: every alias of the object is dead */
	sh_n_free++;
	sh_live--;
	while (k < sh_maxord) {
		unsigned b = i ^ (1u << k);
		if (sh_ctrl[b] != (SH_FREE | k))
			break;
		sh_unlink(b, k);
		unsigned lo = i < b ? i : b, hi = lo + (1u << k);
		/* the handle taken before the split: one revoke, the block is whole again,
		   and the upper half's capability died with it */
		sublet_give_to(&sh_par[sh_paridx(lo, k + 1)], &sh_cap[lo]);
		sublet_clear(&sh_cap[hi]);
		sh_ctrl[hi] = 0;
		sh_n_merge++;
		i = lo;
		k++;
	}
	sh_push(i, k);
}

/* --- lending a block to a NESTED allocator -------------------------------------------------
 * SQLite's memsys5 hands lookaside its block LINEAR and keeps the handle, so one revoke later
 * destroys the pool and every slot in it at once (ports/sqlite/sublet/README.md). FFmpeg's
 * AVBufferPool and AVRefStructPool need exactly that shape, and malloc cannot serve it: malloc
 * returns a DELINEARISED alias (sublet_take), and a nested allocator cannot carve an alias.
 *
 * sublet_malloc_linear carves the block the same way malloc does and hands it out with
 * sublet_take_linear instead: the region stays LINEAR in *out, and sh_cap[i] keeps the senior
 * handle. Everything the borrower splits out of *out therefore hangs BELOW that handle, so the
 * single revoke in sublet_free_linear reclaims the whole sub-pool -- the hierarchy property,
 * not merely a free.
 *
 * Reclaim is keyed by BASE, not by a pointer: the lender holds no alias to probe, and the
 * borrower's own aliases are exactly what the revoke is meant to kill. The caller must have
 * released nothing else from the region first; a revoke of a handle whose region still has live
 * borrowers is the point of the operation, not an error.
 */
unsigned long __capstone_sublet_malloc_linear(size_t n, sublet_cap *out)
{
	unsigned i;
	if (sh_carve_block(n, &i) < 0)
		return 0;
	/* Reads the base before moving the region out, as the header requires. */
	return sublet_take_linear(&sh_cap[i], out);
}

/* Reclaim a block lent by __capstone_sublet_malloc_linear. One revoke kills the lent region and
   every capability the borrower carved from it; the buddy merge below is free's, unchanged. */
void __capstone_sublet_free_linear(unsigned long base)
{
	unsigned i;
	if (sh_state <= 0 || base < sh_base ||
	    base - sh_base >= (1UL << (sh_maxord + CAPSTONE_SUBLET_ATOM_LOG)) ||
	    ((base - sh_base) & ((1UL << CAPSTONE_SUBLET_ATOM_LOG) - 1)) ||
	    !(sh_ctrl[i = (unsigned)((base - sh_base) >> CAPSTONE_SUBLET_ATOM_LOG)] & SH_OUT)) {
		sh_say("sublet-heap: linear free of a base this heap did not lend; ignored\n");
		return;
	}
	unsigned k = sh_ctrl[i] & SH_ORD;
	/* No scrub here, and that is deliberate: the lender holds no alias to write through, and
	   sublet_give's write-through IS the scrub when the revoke returns UNINIT -- which is
	   precisely the case a linear child produces. */
	sublet_give(&sh_cap[i]);
	sh_n_free++;
	sh_live--;
	while (k < sh_maxord) {
		unsigned b = i ^ (1u << k);
		if (sh_ctrl[b] != (SH_FREE | k))
			break;
		sh_unlink(b, k);
		unsigned lo = i < b ? i : b, hi = lo + (1u << k);
		sublet_give_to(&sh_par[sh_paridx(lo, k + 1)], &sh_cap[lo]);
		sublet_clear(&sh_cap[hi]);
		sh_ctrl[hi] = 0;
		sh_n_merge++;
		i = lo;
		k++;
	}
	sh_push(i, k);
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
		memset(p, 0, total);     /* a reused block holds its last owner's bytes */
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
	(void)*(volatile const char *)p;  /* same probe as free, before anything is copied */
	size_t old = __builtin_capstone_cap_get_end(p) - __builtin_capstone_cap_get_cursor(p);
	char *q = malloc(n);
	if (!q)
		return 0;
	/* memmove, not a byte loop: the block may hold capabilities (see level0.c) */
	memmove(q, p, old < n ? old : n);
	free(p);
	return q;
}

void *__libc_malloc(size_t n) { return malloc(n); }
void *__libc_malloc_impl(size_t n) { return malloc(n); }
void *__libc_calloc(size_t n, size_t m) { return calloc(n, m); }
void *__libc_realloc(void *p, size_t n) { return realloc(p, n); }
void __libc_free(void *p) { free(p); }

/* out: allocations, frees, merges, peak live objects, then the Sublet primitives this file
   ran: split, mrev, delin, revoke, init. split + mrev is the revocation-node spend. */
void __capstone_sublet_heap_stats(unsigned long out[9])
{
	out[0] = sh_n_alloc;
	out[1] = sh_n_free;
	out[2] = sh_n_merge;
	out[3] = sh_peak_live;
	out[4] = sublet_stats.split;
	out[5] = sublet_stats.mrev;
	out[6] = sublet_stats.delin;
	out[7] = sublet_stats.revoke;
	out[8] = sublet_stats.init;
}
