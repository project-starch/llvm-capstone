/* The tshark domain's heap report: one line on stderr when the program exits, whether from a
 * staged stop (patch 0006) or at the end of a full run.
 *
 *   TSAPP-HEAP status=<n> in_use=<bytes> peak_in_use=<bytes> peak_end=<bytes> arena=<bytes>
 *
 * The numbers are level0's (runtime/level0.c, built with CAPSTONE_LEVEL0_STATS by
 * host/build-domain.sh), in whole blocks with their headers. peak_end is what sizes the arena:
 * first fit leaves holes, so a run needs an arena of at least peak_end.
 *
 * It is the runtime's __capstone_at_exit (hostcall.c), which runs inside the exit syscall, after
 * musl's exit() has flushed stdout. So the line comes after all of tshark's own output, and
 * host/domain-stdout.py takes it out before the oracle compares. Linked only into the images
 * build-domain.sh makes, never through deps/capstone-cc.
 */
#include <stddef.h>
#include <stdio.h>
#include <unistd.h>

size_t __capstone_level0_in_use(void);
size_t __capstone_level0_peak_in_use(void);
size_t __capstone_level0_peak_end(void);
size_t __capstone_level0_arena_bytes(void);

int __capstone_at_exit(int status)
{
	char line[160];
	int n = snprintf(line, sizeof line,
	                 "TSAPP-HEAP status=%d in_use=%zu peak_in_use=%zu peak_end=%zu arena=%zu\n",
	                 status, __capstone_level0_in_use(), __capstone_level0_peak_in_use(),
	                 __capstone_level0_peak_end(), __capstone_level0_arena_bytes());
	if (n > 0)
		write(2, line, (size_t)n < sizeof line ? (size_t)n : sizeof line - 1);
	return status;
}
