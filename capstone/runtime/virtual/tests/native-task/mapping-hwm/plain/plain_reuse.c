/* Plain baseline's counterpart of CAPSTONE_HEAP_MAPPING_REUSE (capstone/runtime/virtual/malloc-mapping.c): mallocng's
 * group and large mappings are kept on release and handed back to the next mapping of exactly the same length, zeroed
 * with memset. Same bounds: 512 cached mappings, 64 MiB; past either, munmap. At mmap failure the cache is unmapped
 * and the mmap retried once. */
#include <stddef.h>
#include <string.h>
#include <sys/mman.h>
#include "glue.h"

#define PR_FREE 512U
#define PR_FREE_BYTES (64UL << 20)
static struct { void *base; size_t len; } pr_free[PR_FREE];
static unsigned pr_n; static size_t pr_bytes; static volatile int pr_lock_word;
unsigned long __plain_reuse_hits, __plain_reuse_puts, __plain_reuse_evicted;
static void pr_lock(void) { while (__atomic_exchange_n(&pr_lock_word, 1, __ATOMIC_ACQUIRE)) ; }
static void pr_unlock(void) { __atomic_store_n(&pr_lock_word, 0, __ATOMIC_RELEASE); }

/* PLAIN_REUSE_HWM_SLACK (bytes; undefined = off) is the Capstone arm's CAPSTONE_HEAP_MAPPING_REUSE_HWM rule with
 * its CAPSTONE_HEAP_MAPPING_REUSE_HWM_SLACK (llvm-capstone mapping-cache-hwm, mapping-hwm/PREREG.md).
 * - pr_live counts the page-rounded bytes of every mapping this file has handed out and not taken back; pr_peak is
 *   its largest value.
 * - Before a fresh mmap, cached mappings are given back until live + cached + the new mapping fit under
 *   max(peak, live + new) + slack. Each step gives back the smallest mapping that covers the excess, else the
 *   largest.
 * - mallocng's realloc grows a large mapping with the kernel's mremap, outside this file, and __plain_remapped
 *   counts that change. The Capstone arm's __heap_remapping instead maps, copies and unmaps through its own counted
 *   calls.
 * - This file's lock is held from the rule to the mmap, so the rule and the count see one state. The Capstone arm
 *   holds its index lock the same way.
 * PLAIN_REUSE_TEST_LOG prints every event on fd 2 in the Capstone arm's TEST_LOG format,
 * "@@HM <op> <bytes> <live> <cached> <peak>", plus "@@HM X <new> <live> <cached> <peak> <old>" for a remap.
 * With neither defined, the object is the one without the rule. */
#if defined(PLAIN_REUSE_HWM_SLACK) || defined(PLAIN_REUSE_TEST_LOG)
#define PR_COUNT 1
static size_t pr_live, pr_peak;
static void pr_live_add(size_t len)
{
	pr_live += len;
	if (pr_live > pr_peak) pr_peak = pr_live;
}
#ifdef PLAIN_REUSE_TEST_LOG
static void pr_log(char op, size_t bytes, size_t old, int with_old)
{
	char b[160]; unsigned n = 0;
	size_t v[5] = {bytes, pr_live, pr_bytes, pr_peak, old};
	b[n++] = '@'; b[n++] = '@'; b[n++] = 'H'; b[n++] = 'M'; b[n++] = ' '; b[n++] = op;
	for (int k = 0; k < 4 + with_old; ++k) {
		char d[24]; int m = 0; size_t x = v[k];
		do d[m++] = '0' + x % 10; while (x /= 10);
		b[n++] = ' ';
		while (m) b[n++] = d[--m];
	}
	b[n++] = '\n';
	__syscall(SYS_write, 2, b, n);
}
#else
#define pr_log(op, bytes, old, with_old) ((void)0)
#endif
#else
#define PR_COUNT 0
#define pr_live_add(len) ((void)0)
#define pr_log(op, bytes, old, with_old) ((void)0)
#endif
#define pr_event(op, bytes) pr_log(op, bytes, 0, 0)

#ifdef PLAIN_REUSE_HWM_SLACK
unsigned long __plain_reuse_hwm_evicted;
static void pr_fit_peak(size_t len)
{
	size_t cap = (pr_live + len > pr_peak ? pr_live + len : pr_peak) + (size_t)(PLAIN_REUSE_HWM_SLACK);
	while (pr_n && pr_live + pr_bytes + len > cap) {
		size_t need = pr_live + pr_bytes + len - cap;
		unsigned pick = 0;
		for (unsigned i = 1; i < pr_n; ++i) {
			size_t a = pr_free[i].len, c = pr_free[pick].len;
			if (a >= need ? (c < need || a < c) : (c < need && a > c)) pick = i;
		}
		void *b = pr_free[pick].base; size_t l = pr_free[pick].len;
		memmove(pr_free + pick, pr_free + pick + 1, (--pr_n - pick) * sizeof *pr_free);
		pr_bytes -= l; munmap(b, l); ++__plain_reuse_evicted; ++__plain_reuse_hwm_evicted;
		pr_event('H', l);
	}
}
#else
#define pr_fit_peak(len) ((void)0)
#endif

void *__plain_map(size_t len)
{
	/* free() releases an individually mmapped chunk with its page-rounded length (maplen*4096), while malloc maps
	 * it with the exact request: round here, so the cache can hand such a chunk back (the Capstone arm's
	 * __heap_mapping rounds both). */
	len = (len + 4095) & -4096UL;
	pr_lock();
	for (unsigned i = pr_n; i-- > 0;) {
		if (pr_free[i].len != len) continue;
		void *p = pr_free[i].base;
		pr_free[i] = pr_free[--pr_n]; pr_bytes -= len; ++__plain_reuse_hits;
		pr_live_add(len); pr_event('U', len);
		pr_unlock();
		memset(p, 0, len);
		return p;
	}
#if PR_COUNT
	pr_fit_peak(len);
	void *p = mmap(0, len, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
	if (p != MAP_FAILED) { pr_live_add(len); pr_event('M', len); pr_unlock(); return p; }
	unsigned had = pr_n;
	while (pr_n) {
		--pr_n; munmap(pr_free[pr_n].base, pr_free[pr_n].len); ++__plain_reuse_evicted;
		pr_bytes -= pr_free[pr_n].len; pr_event('E', pr_free[pr_n].len);
	}
	pr_bytes = 0;
	p = had ? mmap(0, len, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0) : MAP_FAILED;
	if (p != MAP_FAILED) { pr_live_add(len); pr_event('M', len); }
	pr_unlock();
	return p;
#else
	pr_unlock();
	void *p = mmap(0, len, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
	if (p != MAP_FAILED) return p;
	pr_lock();
	unsigned had = pr_n;
	while (pr_n) { --pr_n; munmap(pr_free[pr_n].base, pr_free[pr_n].len); ++__plain_reuse_evicted; }
	pr_bytes = 0;
	pr_unlock();
	return had ? mmap(0, len, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0) : MAP_FAILED;
#endif
}

void __plain_unmap(void *p, size_t len)
{
	pr_lock();
	/* The Capstone arm's rule (malloc-mapping.c mr_make_room, llvm-capstone nat-realloc-evict 3e70344d8c3f): a
	 * mapping that fits the byte bound by itself gives cached mappings back, from the front of the table, until
	 * it fits. */
	if (len <= PR_FREE_BYTES)
		while (pr_n && (pr_n >= PR_FREE || pr_bytes + len > PR_FREE_BYTES)) {
			void *b = pr_free[0].base; size_t l = pr_free[0].len;
			memmove(pr_free, pr_free + 1, --pr_n * sizeof *pr_free);
			pr_bytes -= l; munmap(b, l); ++__plain_reuse_evicted;
			pr_event('R', l);
		}
	if (pr_n < PR_FREE && pr_bytes + len <= PR_FREE_BYTES) {
		pr_free[pr_n].base = p; pr_free[pr_n].len = len; ++pr_n; pr_bytes += len; ++__plain_reuse_puts;
#if PR_COUNT
		/* Every caller passes a page multiple except malloc.c's unmap after a failed alloc_meta; the rounded
		 * count keeps live exact there, and its C line then reads the unrounded length. */
		pr_live -= (len + 4095) & -4096UL;
#endif
		pr_event('C', len);
		pr_unlock();
		return;
	}
#if PR_COUNT
	pr_live -= (len + 4095) & -4096UL;
	pr_event('F', (len + 4095) & -4096UL);
#endif
	pr_unlock();
	munmap(p, len);
}

#if PR_COUNT
/* realloc.c, after a successful mremap of a large mapping from old to new bytes (both page multiples). */
void __plain_remapped(size_t old, size_t new)
{
	pr_lock();
	pr_live -= old;
	pr_live_add(new);
	pr_log('X', new, old, 1);
	pr_unlock();
}
#endif
