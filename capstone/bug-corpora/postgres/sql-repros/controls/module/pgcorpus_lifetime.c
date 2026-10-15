/*
 * Parent, sibling and successor controls at PostgreSQL's memory contexts,
 * run inside the backend (pre-registration:
 * capstone/docs/plans/2026-10-11-nested-lifetime-controls.md).
 *
 * Not defects. Each control drives the nested allocator through its own API --
 * palloc and pfree on a dedicated AllocSet, the context type patch 0003
 * protects -- and then accesses memory through a reference the allocator
 * itself handed out or keeps. Predictions, on the protected build
 * (PGSU_NESTED=sublet, memory-contexts patch 0003) and on the release build
 * (the same source without it):
 *
 *   parent           read and write A's former range through the AllocBlock
 *                    that holds it, reached as aset.c reaches it, through the
 *                    context's block list: completes on both
 *   sibling          A and B from one block, pfree A, read and write B:
 *                    completes on both, B intact
 *   successor        pfree A, palloc until the context hands back A's address
 *                    (C), read and write through C: completes on both
 *   successor-stale  the same, then one read through the old A: rejected with
 *                    cause 25 on protected, completes on release and reads
 *                    C's data
 *
 * Every access goes through one of the lc_probe_* functions below, so a fault
 * is attributed by the pc's function and a completed access is accounted for
 * by name. Each control prints its addresses and checks to stderr with
 * write(2) BEFORE the access it describes: the stand-alone backend's own
 * output is buffered and dies with the process, stderr is not.
 *
 * The dedicated context is an AllocSet created with sizes (0, 8192, 8192),
 * which match neither ALLOCSET_DEFAULT_SIZES nor ALLOCSET_SMALL_SIZES, so
 * aset.c never recycles it from its context freelists: every run gets a
 * freshly malloc'd context whose keeper block holds every chunk below. All
 * requests are 64 bytes, one size class (fidx 2: 16 << 2), well below the
 * keeper block's 8 KiB.
 */
#include "postgres.h"

#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>

#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "utils/memutils_memorychunk.h"

PG_MODULE_MAGIC;

#define LC_REQUEST 64
#define LC_FIDX 2				/* 64 == (1 << ALLOC_MINBITS) << 2 with patch 0002 */
#define LC_NUM_FREELISTS 10		/* ALLOCSET_NUM_FREELISTS with patch 0002 */
#define LC_MAX_TRIES 64

/*
 * aset.c's private layout, the leading fields only, as patched for this port
 * (app patch 0002 sets ALLOCSET_NUM_FREELISTS to 10; patch 0003 appends a
 * SubletBlock to AllocBlockData, after the fields read here). Checked at run
 * time rather than trusted: the block reached through `blocks` must name this
 * context as its owner and must contain every chunk the control touches.
 */
typedef struct LcBlock
{
	void	   *aset;
	struct LcBlock *prev;
	struct LcBlock *next;
	char	   *freeptr;
	char	   *endptr;
} LcBlock;

typedef struct LcAllocSet
{
	MemoryContextData header;
	LcBlock    *blocks;
	MemoryChunk *freelist[LC_NUM_FREELISTS];
} LcAllocSet;

static unsigned long
lc_addr(const void *p)
{
	return (unsigned long) (uintptr_t) p;
}

static void
lc_say(const char *fmt,...)
{
	char		buf[512];
	va_list		ap;
	int			n;

	va_start(ap, fmt);
	n = vsnprintf(buf, sizeof buf - 1, fmt, ap);
	va_end(ap);
	if (n < 0)
		return;
	if (n > (int) sizeof buf - 2)
		n = sizeof buf - 2;
	buf[n++] = '\n';
	(void) write(2, buf, n);
}

/* A control that cannot set itself up is no reading, never a result. */
#define LC_REQUIRE(cond, ...) \
	do { \
		if (!(cond)) \
		{ \
			lc_say("LIFETIME-CONTROL-FAILED " __VA_ARGS__); \
			elog(ERROR, "lifetime control setup failed: %s", #cond); \
		} \
	} while (0)

static void
lc_bounds(const char *what, void *p)
{
	lc_say("LIFETIME %s addr=0x%lx base=0x%lx end=0x%lx tag=%d", what, lc_addr(p),
		   (unsigned long) __builtin_capstone_cap_get_base(p),
		   (unsigned long) __builtin_capstone_cap_get_end(p),
		   (int) __builtin_capstone_cap_get_tag(p));
}

/* ---- the probes: every access a control makes, one function each ---------- */

static __attribute__((noinline)) void *
lc_probe_parent_read_link(void *const volatile *slot)
{
	return *slot;				/* parent: load of A's former free-list link */
}

static __attribute__((noinline)) uint64
lc_probe_parent_read_header(const volatile uint64 *hdrmask)
{
	return *hdrmask;			/* parent: load of A's former header word */
}

static __attribute__((noinline)) void
lc_probe_parent_write(volatile unsigned char *p, size_t n, unsigned char v)
{
	for (size_t i = 0; i < n; i++)
		p[i] = v;				/* parent: store into A's former range */
}

static __attribute__((noinline)) int
lc_probe_parent_read(const volatile unsigned char *p, size_t n, unsigned char v)
{
	int			bad = 0;

	for (size_t i = 0; i < n; i++)
		bad += p[i] != v;		/* parent: load from A's former range */
	return bad;
}

static __attribute__((noinline)) int
lc_probe_sibling_read(const volatile unsigned char *p, size_t n, unsigned char seed)
{
	int			bad = 0;

	for (size_t i = 0; i < n; i++)
		bad += p[i] != (unsigned char) (seed ^ i);	/* sibling: load from B */
	return bad;
}

static __attribute__((noinline)) void
lc_probe_sibling_write(volatile unsigned char *p, size_t n, unsigned char seed)
{
	for (size_t i = 0; i < n; i++)
		p[i] = (unsigned char) (seed ^ i);	/* sibling: store into B */
}

static __attribute__((noinline)) void
lc_probe_successor_write(volatile unsigned char *p, size_t n, unsigned char seed)
{
	for (size_t i = 0; i < n; i++)
		p[i] = (unsigned char) (seed ^ i);	/* successor: store through C */
}

static __attribute__((noinline)) int
lc_probe_successor_read(const volatile unsigned char *p, size_t n, unsigned char seed)
{
	int			bad = 0;

	for (size_t i = 0; i < n; i++)
		bad += p[i] != (unsigned char) (seed ^ i);	/* successor: load through C */
	return bad;
}

static __attribute__((noinline)) uint64
lc_probe_stale_read(const volatile uint64 *p)
{
	return *p;					/* successor-stale: load through the old A */
}

/* ---- setup ---------------------------------------------------------------- */

static MemoryContext
lc_context(void)
{
	return AllocSetContextCreate(CurrentMemoryContext, "lifetime-control",
								 0, 8192, 8192);
}

static void
lc_fill(unsigned char *p, size_t n, unsigned char seed)
{
	for (size_t i = 0; i < n; i++)
		p[i] = (unsigned char) (seed ^ i);
}

/* The block on the context's list that holds [p, p + n), checked. */
static LcBlock *
lc_block_of(MemoryContext cxt, const void *p, size_t n)
{
	LcAllocSet *set = (LcAllocSet *) cxt;

	for (LcBlock *b = set->blocks; b != NULL; b = b->next)
		if (lc_addr(b->aset) == lc_addr(set) &&
			lc_addr(p) >= lc_addr(b) && lc_addr(p) + n <= lc_addr(b->endptr))
			return b;
	return NULL;
}

/* ---- the controls --------------------------------------------------------- */

static void
lc_parent(StringInfo out)
{
	MemoryContext cxt = lc_context();
	LcAllocSet *set = (LcAllocSet *) cxt;
	unsigned char *d = MemoryContextAlloc(cxt, LC_REQUEST);
	unsigned char *a = MemoryContextAlloc(cxt, LC_REQUEST);
	LcBlock    *block = lc_block_of(cxt, a - sizeof(MemoryChunk), sizeof(MemoryChunk) + LC_REQUEST);
	unsigned long off;
	uint64		hdr_before;
	unsigned char *former;
	void	   *link;
	uint64		hdr_after;
	int			bad;
	unsigned char *again;

	LC_REQUIRE(block != NULL, "parent: no block of this context holds A");
	off = lc_addr(a) - lc_addr(block);
	hdr_before = PointerGetMemoryChunk(a)->hdrmask;
	lc_fill(a, LC_REQUEST, 0xA1);
	lc_say("LIFETIME parent context=0x%lx block=0x%lx block_end=0x%lx chunk_header_bytes=%zu "
		   "A=0x%lx D=0x%lx A_offset_in_block=%lu", lc_addr(cxt), lc_addr(block),
		   lc_addr(block->endptr), sizeof(MemoryChunk), lc_addr(a), lc_addr(d), off);
	lc_bounds("parent A-live", a);
	lc_bounds("parent block-ref", block);

	pfree(d);
	pfree(a);
	LC_REQUIRE(lc_addr(set->freelist[LC_FIDX]) == lc_addr(a) - sizeof(MemoryChunk),
			   "parent: freelist[%d] is 0x%lx, not A's chunk", LC_FIDX,
			   lc_addr(set->freelist[LC_FIDX]));

	/* A's former range, reached from the block, as the allocator reaches it */
	former = (unsigned char *) block + off;
	lc_say("LIFETIME parent mark access=read-link via=block+%lu", off);
	link = lc_probe_parent_read_link((void *const volatile *) former);
	lc_say("LIFETIME parent read-link=0x%lx expected D-chunk=0x%lx", lc_addr(link),
		   lc_addr(d) - sizeof(MemoryChunk));
	lc_say("LIFETIME parent mark access=read-header via=block+%lu", off - sizeof(uint64));
	hdr_after = lc_probe_parent_read_header((const volatile uint64 *) (former - sizeof(uint64)));
	lc_say("LIFETIME parent read-header=0x%lx before-free=0x%lx", (unsigned long) hdr_after,
		   (unsigned long) hdr_before);
	/* past the free-list link, so the write leaves the allocator's state alone */
	lc_say("LIFETIME parent mark access=write via=block+%lu bytes=%d", off + 16, LC_REQUEST - 16);
	lc_probe_parent_write(former + 16, LC_REQUEST - 16, 0x5A);
	lc_say("LIFETIME parent mark access=read via=block+%lu bytes=%d", off + 16, LC_REQUEST - 16);
	bad = lc_probe_parent_read(former + 16, LC_REQUEST - 16, 0x5A);
	lc_say("LIFETIME parent read-back mismatches=%d", bad);
	again = MemoryContextAlloc(cxt, LC_REQUEST);
	lc_say("LIFETIME parent next-palloc=0x%lx (A=0x%lx)", lc_addr(again), lc_addr(a));

	appendStringInfo(out, "parent link_ok=%d header_ok=%d readback_ok=%d allocator_intact=%d",
					 lc_addr(link) == lc_addr(d) - sizeof(MemoryChunk),
					 hdr_after == hdr_before, bad == 0, lc_addr(again) == lc_addr(a));
	MemoryContextDelete(cxt);
}

static void
lc_sibling(StringInfo out)
{
	MemoryContext cxt = lc_context();
	unsigned char *a = MemoryContextAlloc(cxt, LC_REQUEST);
	unsigned char *b = MemoryContextAlloc(cxt, LC_REQUEST);
	LcBlock    *ba = lc_block_of(cxt, a - sizeof(MemoryChunk), sizeof(MemoryChunk) + LC_REQUEST);
	LcBlock    *bb = lc_block_of(cxt, b - sizeof(MemoryChunk), sizeof(MemoryChunk) + LC_REQUEST);
	int			bad1,
				bad2;

	LC_REQUIRE(ba != NULL && ba == bb, "sibling: A and B are not in one block");
	lc_fill(a, LC_REQUEST, 0xA1);
	lc_fill(b, LC_REQUEST, 0xB2);
	lc_say("LIFETIME sibling context=0x%lx block=0x%lx block_end=0x%lx chunk_header_bytes=%zu "
		   "A=0x%lx B=0x%lx", lc_addr(cxt), lc_addr(ba), lc_addr(ba->endptr),
		   sizeof(MemoryChunk), lc_addr(a), lc_addr(b));
	lc_bounds("sibling A-live", a);
	lc_bounds("sibling B-live", b);
	pfree(a);
	lc_say("LIFETIME sibling mark access=read via=B");
	bad1 = lc_probe_sibling_read(b, LC_REQUEST, 0xB2);
	lc_say("LIFETIME sibling contents-after-free-of-A mismatches=%d", bad1);
	lc_say("LIFETIME sibling mark access=write via=B");
	lc_probe_sibling_write(b, LC_REQUEST, 0x3C);
	lc_say("LIFETIME sibling mark access=read via=B");
	bad2 = lc_probe_sibling_read(b, LC_REQUEST, 0x3C);
	lc_say("LIFETIME sibling read-back mismatches=%d", bad2);
	appendStringInfo(out, "sibling intact=%d readback_ok=%d", bad1 == 0, bad2 == 0);
	MemoryContextDelete(cxt);
}

static void
lc_successor(StringInfo out, bool stale)
{
	const char *name = stale ? "successor-stale" : "successor";
	MemoryContext cxt = lc_context();
	unsigned char *a = MemoryContextAlloc(cxt, LC_REQUEST);
	unsigned char *c = NULL;
	int			tries;
	int			bad;
	uint64		got,
				want;

	lc_fill(a, LC_REQUEST, 0xA1);
	lc_bounds(stale ? "successor-stale A-live" : "successor A-live", a);
	pfree(a);
	for (tries = 1; tries <= LC_MAX_TRIES; tries++)
	{
		c = MemoryContextAlloc(cxt, LC_REQUEST);
		if (lc_addr(c) == lc_addr(a))
			break;
	}
	lc_say("LIFETIME %s context=0x%lx chunk_header_bytes=%zu A=0x%lx C=0x%lx pallocs=%d equal=%d",
		   name, lc_addr(cxt), sizeof(MemoryChunk), lc_addr(a), lc_addr(c), tries,
		   lc_addr(c) == lc_addr(a));
	LC_REQUIRE(lc_addr(c) == lc_addr(a), "%s: no palloc of %d returned A's address", name,
			   LC_MAX_TRIES);
	lc_bounds(stale ? "successor-stale C-live" : "successor C-live", c);
	lc_say("LIFETIME %s mark access=write via=C", name);
	lc_probe_successor_write(c, LC_REQUEST, 0xC5);
	lc_say("LIFETIME %s mark access=read via=C", name);
	bad = lc_probe_successor_read(c, LC_REQUEST, 0xC5);
	lc_say("LIFETIME %s read-back mismatches=%d", name, bad);
	if (!stale)
	{
		appendStringInfo(out, "successor same_address=1 readback_ok=%d", bad == 0);
		MemoryContextDelete(cxt);
		return;
	}
	memcpy(&want, c, sizeof want);
	lc_say("LIFETIME successor-stale mark access=stale-read via=A C-data=0x%lx",
		   (unsigned long) want);
	got = lc_probe_stale_read((const volatile uint64 *) a);
	lc_say("LIFETIME successor-stale stale-read=0x%lx C-data=0x%lx equal=%d",
		   (unsigned long) got, (unsigned long) want, got == want);
	appendStringInfo(out, "successor-stale same_address=1 readback_ok=%d stale_read_completed=1 "
					 "stale_equals_C=%d", bad == 0, got == want);
	MemoryContextDelete(cxt);
}

PG_FUNCTION_INFO_V1(lifetime_control);

Datum
lifetime_control(PG_FUNCTION_ARGS)
{
	char	   *name = text_to_cstring(PG_GETARG_TEXT_PP(0));
	StringInfoData out;

	initStringInfo(&out);
	lc_say("LIFETIME-BEGIN %s", name);
	if (strcmp(name, "parent") == 0)
		lc_parent(&out);
	else if (strcmp(name, "sibling") == 0)
		lc_sibling(&out);
	else if (strcmp(name, "successor") == 0)
		lc_successor(&out, false);
	else if (strcmp(name, "successor-stale") == 0)
		lc_successor(&out, true);
	else
		elog(ERROR, "unknown lifetime control \"%s\"", name);
	lc_say("LIFETIME-RETURNED %s %s", name, out.data);
	PG_RETURN_TEXT_P(cstring_to_text(out.data));
}
