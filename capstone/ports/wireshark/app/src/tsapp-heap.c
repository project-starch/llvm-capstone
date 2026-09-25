/* The tshark domain's heap report: one line on stderr when the program exits, whether from a
 * staged stop (patch 0006) or at the end of a full run.
 *
 *   TSAPP-HEAP status=<n> in_use=<bytes> peak_in_use=<bytes> peak_end=<bytes> arena=<bytes>
 *              unserved=<nr>[x<times>],... stdout=open|closed
 *
 * The numbers are level0's (runtime/level0.c, built with CAPSTONE_LEVEL0_STATS by
 * host/build-domain.sh), in whole blocks with their headers. peak_end is what sizes the arena:
 * first fit leaves holes, so a run needs an arena of at least peak_end.
 *
 * On the sublet heap arm (-DTSAPP_SUBLET_HEAP, runtime/sublet_heap.c) the numbers are that heap's
 * instead, and the line reads
 *   TSAPP-HEAP status=<n> sublet alloc=<n> free=<n> merge=<n> peak_live=<objects> split=<n>
 *              mrev=<n> delin=<n> revoke=<n> init=<n> unserved=... stdout=open|closed
 * where split + mrev is the revocation-node spend (sublet_heap.c).
 *
 * The unserved syscalls are reported HERE, on fd 2, as well as by the runtime. The runtime's own
 * line goes to fd 1 after the program has finished, and a program that closed fd 1 loses it
 * without a trace: the full tshark run reported none while its first four stages reported fifteen
 * (2026-09-24). `stdout=` says whether fd 1 was still open when the program exited.
 *
 * It is the runtime's __capstone_at_exit (hostcall.c), which runs inside the exit syscall, after
 * musl's exit() has flushed stdout. So the line comes after all of tshark's own output, and
 * host/domain-stdout.py takes it out before the oracle compares. Linked only into the images
 * build-domain.sh makes, never through deps/capstone-cc.
 */
#include <stddef.h>
#include <stdio.h>
#include <unistd.h>

#ifdef TSAPP_SUBLET_HEAP
void __capstone_sublet_heap_stats(unsigned long out[9]);
#else
size_t __capstone_level0_in_use(void);
size_t __capstone_level0_peak_in_use(void);
size_t __capstone_level0_peak_end(void);
size_t __capstone_level0_arena_bytes(void);
#endif
unsigned long __capstone_unserved_count(void);
long __capstone_unserved_at(unsigned long i);

int __capstone_at_exit(int status)
{
	char line[512];
	size_t cap = sizeof line - 2;
#ifdef TSAPP_SUBLET_HEAP
	unsigned long st[9];
	__capstone_sublet_heap_stats(st);
	int n = snprintf(line, cap,
	                 "TSAPP-HEAP status=%d sublet alloc=%lu free=%lu merge=%lu peak_live=%lu split=%lu"
	                 " mrev=%lu delin=%lu revoke=%lu init=%lu unserved=",
	                 status, st[0], st[1], st[2], st[3], st[4], st[5], st[6], st[7], st[8]);
#else
	int n = snprintf(line, cap,
	                 "TSAPP-HEAP status=%d in_use=%zu peak_in_use=%zu peak_end=%zu arena=%zu unserved=",
	                 status, __capstone_level0_in_use(), __capstone_level0_peak_in_use(),
	                 __capstone_level0_peak_end(), __capstone_level0_arena_bytes());
#endif
	size_t p = n > 0 ? (size_t)n : 0;
	/* The runtime keeps the first 16 numbers asked for (hostcall.c, HC_UNSERVED_MAX) and counts
	 * the rest: __capstone_unserved_at() is -1 past what it kept. One entry per distinct number,
	 * first-seen order, with its count, as the runtime's own line; then "(of N)" if N > kept. */
	unsigned long total = __capstone_unserved_count(), kept = 0;
	while (kept < total && __capstone_unserved_at(kept) >= 0)
		kept++;
	if (total == 0 && p < cap)
		p += (size_t)snprintf(line + p, cap - p, "none");
	for (unsigned long i = 0; i < kept && p < cap; i++) {
		long v = __capstone_unserved_at(i), times = 0, seen = 0;
		for (unsigned long j = 0; j < kept; j++)
			if (__capstone_unserved_at(j) == v) {
				if (j < i)
					seen = 1;
				times++;
			}
		if (seen)
			continue;
		p += (size_t)snprintf(line + p, cap - p, times > 1 ? "%s%ldx%ld" : "%s%ld",
		                      i ? "," : "", v, times);
	}
	if (total > kept && p < cap)
		p += (size_t)snprintf(line + p, cap - p, "(of %lu)", total);
	if (p < cap)
		p += (size_t)snprintf(line + p, cap - p, " stdout=%s", write(1, "", 0) < 0 ? "closed" : "open");
	if (p > cap)
		p = cap;
	line[p++] = '\n';
	write(2, line, p);
	return status;
}
